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

fn braced_item<'a>(source: &'a str, start: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing braced source boundary {start:?}"));
    let open_offset = source[start_index..]
        .find('{')
        .unwrap_or_else(|| panic!("missing opening brace after {start:?}"));
    let open_index = start_index + open_offset;
    let mut depth = 0_u64;
    for (offset, byte) in source.as_bytes()[open_index..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start_index..=open_index + offset];
                }
            }
            _ => {}
        }
    }
    panic!("missing closing brace after {start:?}")
}

fn without_optional_braced_item(source: &str, start: &str) -> String {
    let Some(start_index) = source.find(start) else {
        return source.to_owned();
    };
    let item = braced_item(source, start);
    let mut remainder = String::with_capacity(source.len() - item.len());
    remainder.push_str(&source[..start_index]);
    remainder.push_str(&source[start_index + item.len()..]);
    remainder
}

fn identifier_after<'a>(source: &'a str, marker: &str) -> &'a str {
    let tail = source
        .split_once(marker)
        .unwrap_or_else(|| panic!("missing identifier marker {marker:?}"))
        .1
        .trim_start();
    let end = tail
        .find(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .unwrap_or(tail.len());
    assert!(end > 0, "missing identifier after {marker:?}");
    &tail[..end]
}

fn require_by_value_type(signature: &str, type_name: &str, context: &str) {
    let open = signature
        .find('(')
        .unwrap_or_else(|| panic!("{context} has no parameter list"));
    let mut parenthesis_depth = 0_u64;
    let close = signature.as_bytes()[open..]
        .iter()
        .enumerate()
        .find_map(|(offset, byte)| match byte {
            b'(' => {
                parenthesis_depth += 1;
                None
            }
            b')' => {
                parenthesis_depth -= 1;
                (parenthesis_depth == 0).then_some(open + offset)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("{context} has an unclosed parameter list"));
    let mut delimiter_depth = 0_i64;
    let mut start = open + 1;
    let mut exact_matches = 0_u64;
    for end in (open + 1..=close).filter(|&index| {
        if index == close {
            return true;
        }
        match signature.as_bytes()[index] {
            b'(' | b'[' | b'{' | b'<' => delimiter_depth += 1,
            b')' | b']' | b'}' | b'>' => delimiter_depth -= 1,
            b',' if delimiter_depth == 0 => return true,
            _ => {}
        }
        false
    }) {
        let parameter = signature[start..end].trim();
        start = end + 1;
        let Some((pattern, parameter_type)) = parameter.split_once(':') else {
            continue;
        };
        let exact_type: String = parameter_type
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        if exact_type == type_name {
            assert!(
                !pattern.split_whitespace().any(|token| token == "ref") && !pattern.contains('&'),
                "{context} must move `{type_name}` through a plain binding"
            );
            exact_matches += 1;
        }
    }
    assert_eq!(
        exact_matches, 1,
        "{context} must have exactly one parameter whose type is exactly `{type_name}`; references, wrappers and aliases are rejected"
    );
}

fn require_move_only_type(source: &str, type_marker: &str) {
    let declaration = format!("struct {type_marker}");
    let type_index = source
        .find(&declaration)
        .unwrap_or_else(|| panic!("missing move-only type `{type_marker}`"));
    let prefix_start = source[..type_index]
        .rfind("\n\n")
        .map_or(0, |index| index + 2);
    let compact_attributes: String = source[prefix_start..type_index]
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    for trait_name in ["Clone", "Copy"] {
        let mut derives = compact_attributes.as_str();
        while let Some((_, after_derive)) = derives.split_once("derive(") {
            let (derive_list, rest) = after_derive
                .split_once(')')
                .unwrap_or_else(|| panic!("unclosed derive attribute for `{type_marker}`"));
            assert!(
                !derive_list
                    .split(',')
                    .any(|derived| derived.rsplit("::").next() == Some(trait_name)),
                "{type_marker} must not derive {trait_name}"
            );
            derives = rest;
        }
        assert!(
            !source.contains(&format!("impl {trait_name} for {type_marker}")),
            "{type_marker} must not implement {trait_name}"
        );
    }
}

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(
            source.contains(token),
            "resident trim/prefilter native V1 source is missing {token:?}"
        );
    }
}

fn normalized_abi_sha256(source: &str) -> String {
    use sha2::{Digest, Sha256};

    let normalized: String = source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    Sha256::digest(normalized.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn require_declaration_repr_c(source: &str, type_marker: &str) {
    let type_index = source
        .find(type_marker)
        .unwrap_or_else(|| panic!("missing Rust ABI type `{type_marker}`"));
    let declaration_line_start = source[..type_index]
        .rfind('\n')
        .map_or(0, |index| index + 1);
    let mut attribute_start = declaration_line_start;
    while attribute_start > 0 {
        let previous_line_end = attribute_start - 1;
        let previous_line_start = source[..previous_line_end]
            .rfind('\n')
            .map_or(0, |index| index + 1);
        let previous_line = source[previous_line_start..previous_line_end].trim();
        if !previous_line.starts_with("#[") || !previous_line.ends_with(']') {
            break;
        }
        attribute_start = previous_line_start;
    }
    let attributes = &source[attribute_start..declaration_line_start];
    let repr_attributes = attributes
        .lines()
        .filter(|line| line.trim() == "#[repr(C)]")
        .count();
    assert_eq!(
        repr_attributes, 1,
        "{type_marker} declaration attribute block must contain exactly one #[repr(C)]"
    );
}

fn require_c_and_rust_offsets(
    header: &str,
    rust: &str,
    c_type: &str,
    rust_type: &str,
    offsets: &[(&str, usize)],
) {
    let header: String = header
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let rust: String = rust
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    for (field, offset) in offsets {
        let c_assert = format!("static_assert(offsetof({c_type},{field})=={offset},");
        assert!(
            header.contains(&c_assert),
            "{c_type}.{field} lacks exact C offset {offset}"
        );
        let rust_assert =
            format!("const_:[();{offset}]=[();mem::offset_of!({rust_type},{field})];");
        assert!(
            rust.contains(&rust_assert),
            "{rust_type}.{field} lacks exact Rust offset {offset}"
        );
    }
}

fn require_c_and_rust_size(header: &str, rust: &str, c_type: &str, rust_type: &str, size: usize) {
    let header: String = header
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let rust: String = rust
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert!(
        header.contains(&format!("static_assert(sizeof({c_type})=={size},")),
        "{c_type} lacks exact C size {size}"
    );
    assert!(
        rust.contains(&format!(
            "const_:[();{size}]=[();mem::size_of::<{rust_type}>()];"
        )),
        "{rust_type} lacks exact Rust size {size}"
    );
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
fn published_v1_plan_allocation_and_views_keep_exact_type_order_repr_and_size() {
    let header = read_required("native/resident_trim_prefilter_v1_abi.cuh");
    let rust = read_required("src/resident_trim_prefilter_v1.rs");

    for (c_type, rust_type, c_sha256, rust_sha256, size) in [
        (
            "NeoResidentTrimPrefilterPlanV1",
            "RawResidentTrimPrefilterPlanV1",
            "5794e495f6a103987da748c7f9d0fb09da7ef5de82b2bc8fc4f66adb3d20dbe1",
            "e75ad34573416d21454c9d46c974f5ca6d8e42f01f23e999880e775c4a371c2b",
            608,
        ),
        (
            "NeoResidentTrimPrefilterAllocationReceiptV1",
            "RawResidentTrimPrefilterAllocationReceiptV1",
            "9b2e08fe5ba0edd3e069ed79562d4bad23ed895f98f80f197404709af8054bda",
            "717d234b9ae6ba0f1aeb3e61777113ae24b28810f5acb925c72e9cc74cb78638",
            200,
        ),
        (
            "NeoResidentTrimPrefilterViewsV1",
            "RawResidentTrimPrefilterViewsV1",
            "0722c6113f4ff15a16837ebeebed819e7001cbb3d7cbb1b3a153a196dc7022f8",
            "86e58596e2eab21f62771907080fc545491b1a8e59c9b555ded48ac019af473b",
            344,
        ),
    ] {
        let c_declaration = braced_item(&header, &format!("struct {c_type} {{"));
        let rust_declaration = braced_item(&rust, &format!("struct {rust_type} {{"));
        assert_eq!(
            normalized_abi_sha256(c_declaration),
            c_sha256,
            "{c_type} field type/order changed"
        );
        assert_eq!(
            normalized_abi_sha256(rust_declaration),
            rust_sha256,
            "{rust_type} field type/order changed"
        );
        require_declaration_repr_c(&rust, &format!("struct {rust_type} {{"));
        require_c_and_rust_size(&header, &rust, c_type, rust_type, size);
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
            !braced_item(&cuda, "bool resolve_absolute_view_ranges_v1(").contains(forbidden),
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
    let cuda_without_v2_bounded_readback = without_optional_braced_item(
        &cuda,
        "extern \"C\" std::int32_t read_resident_trim_prefilter_selected_map_v2(",
    );
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
            !cuda_without_v2_bounded_readback.contains(forbidden),
            "V1 exports and their private helpers escape via {forbidden:?}; only the exact additive V2 bounded-readback body is exempt"
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
    let v1_public_view = format!(
        "{}\n{}",
        braced_item(&rust, "pub struct SealedResidentTrimPrefilterDeviceViewsV1"),
        braced_item(&rust, "impl SealedResidentTrimPrefilterDeviceViewsV1")
    );
    for forbidden in [
        "pub selected_column_count: u64",
        "pub selected_columns:",
        "Vec<u32>",
        "Vec<usize>",
        "copy_selected",
    ] {
        assert!(
            !v1_public_view.contains(forbidden),
            "opaque V1 result leaks through {forbidden:?}"
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

#[test]
fn additive_v2_score_batches_bind_local_stride_and_global_parent_order() {
    let header = read_required("native/resident_trim_prefilter_v1_abi.cuh");
    let rust = read_required("src/resident_trim_prefilter_v1.rs");
    let cuda = read_required("native/resident_trim_prefilter_v1.cu");

    // V2 is additive.  The already-published V1 ABI number, layouts and
    // symbols remain the byte-stable identity understood by existing callers.
    require_all(
        &header,
        &[
            "constexpr std::uint32_t NEO_RESIDENT_TRIM_PREFILTER_ABI_V1 = 1U;",
            "static_assert(sizeof(NeoResidentTrimPrefilterImportV1) == 560",
            "static_assert(sizeof(NeoResidentTrimPrefilterPlanV1) == 608",
            "static_assert(sizeof(NeoResidentTrimPrefilterViewsV1) == 344",
            "query_resident_trim_prefilter_scratch_v1(",
            "query_resident_trim_prefilter_allocation_v1(",
            "create_resident_trim_prefilter_run_v1(",
            "enqueue_resident_trim_prefilter_stage_v1(",
            "seal_resident_trim_prefilter_views_v1(",
            "enqueue_resident_trim_prefilter_release_v1(",
        ],
    );
    require_all(
        &rust,
        &[
            "const ABI_VERSION_V1: u32 = 1;",
            "const _: [(); 560] = [(); mem::size_of::<RawResidentTrimPrefilterImportV1>()]",
            "const _: [(); 608] = [(); mem::size_of::<RawResidentTrimPrefilterPlanV1>()]",
            "const _: [(); 344] = [(); mem::size_of::<RawResidentTrimPrefilterViewsV1>()]",
        ],
    );
    let v1_import = section(&header, "struct NeoResidentTrimPrefilterImportV1 {", "\n};");
    for forbidden in [
        "batch_values_bar_major",
        "batch_validity_u4",
        "local_batch_stride",
        "global_parent_column_start",
        "global_parent_ordinals_device",
    ] {
        assert!(
            !v1_import.contains(forbidden),
            "V2 streaming field `{forbidden}` mutated the V1 import ABI"
        );
    }

    require_all(
        &header,
        &[
            "constexpr std::uint32_t NEO_RESIDENT_TRIM_PREFILTER_SCORE_BATCH_ABI_V2 = 2U;",
            "struct NeoResidentTrimPrefilterScoreBatchV2 {",
            "std::uint32_t abi_version;",
            "std::uint32_t reserved;",
            "const double* batch_values_bar_major;",
            "const unsigned char* batch_validity_u4;",
            "std::uint64_t batch_row_count;",
            "std::uint64_t batch_column_count;",
            "std::uint64_t local_batch_stride;",
            "std::uint64_t global_parent_column_start;",
            "const std::uint32_t* global_parent_ordinals_device;",
            "cudaEvent_t batch_ready_event;",
            "enqueue_resident_trim_prefilter_score_batch_v2(",
        ],
    );
    require_all(
        &rust,
        &[
            "const SCORE_BATCH_ABI_VERSION_V2: u32 = 2;",
            "struct RawResidentTrimPrefilterScoreBatchV2 {",
            "abi_version: u32",
            "reserved: u32",
            "batch_values_bar_major: *const f64",
            "batch_validity_u4: *const u8",
            "batch_row_count: u64",
            "batch_column_count: u64",
            "local_batch_stride: u64",
            "global_parent_column_start: u64",
            "global_parent_ordinals_device: *const u32",
            "batch_ready_event: *mut c_void",
            "abi_version: SCORE_BATCH_ABI_VERSION_V2",
            "enqueue_resident_trim_prefilter_score_batch_v2(",
        ],
    );
    require_declaration_repr_c(&rust, "struct RawResidentTrimPrefilterScoreBatchV2 {");
    require_c_and_rust_size(
        &header,
        &rust,
        "NeoResidentTrimPrefilterScoreBatchV2",
        "RawResidentTrimPrefilterScoreBatchV2",
        72,
    );
    require_c_and_rust_offsets(
        &header,
        &rust,
        "NeoResidentTrimPrefilterScoreBatchV2",
        "RawResidentTrimPrefilterScoreBatchV2",
        &[
            ("abi_version", 0),
            ("reserved", 4),
            ("batch_values_bar_major", 8),
            ("batch_validity_u4", 16),
            ("batch_row_count", 24),
            ("batch_column_count", 32),
            ("local_batch_stride", 40),
            ("global_parent_column_start", 48),
            ("global_parent_ordinals_device", 56),
            ("batch_ready_event", 64),
        ],
    );
    let score_enqueue = braced_item(
        &cuda,
        "extern \"C\" std::int32_t enqueue_resident_trim_prefilter_score_batch_v2(",
    );
    let score_kernel = braced_item(
        &cuda,
        "__global__ void resident_trim_prefilter_score_batch_kernel_v2(",
    );
    let checked_address = braced_item(
        &cuda,
        "__host__ __device__ bool checked_resident_trim_prefilter_score_batch_address_v2(",
    );
    require_all(
        braced_item(&cuda, "struct ResidentTrimPrefilterScoreBatchAddressV2 {"),
        &[
            "std::uint64_t local_offset;",
            "std::uint32_t global_parent_ordinal;",
        ],
    );
    require_all(
        score_enqueue,
        &["batch->abi_version != NEO_RESIDENT_TRIM_PREFILTER_SCORE_BATCH_ABI_V2"],
    );
    require_all(
        checked_address,
        &[
            "batch->batch_row_count",
            "batch->batch_column_count",
            "batch->local_batch_stride",
            "batch->global_parent_ordinals_device",
            "row * batch->local_batch_stride + local_column",
            "batch->global_parent_ordinals_device[local_column]",
            "ResidentTrimPrefilterScoreBatchAddressV2*",
            "->local_offset",
            "->global_parent_ordinal",
        ],
    );
    assert!(
        checked_address.contains("row >=")
            && checked_address.contains("local_column >=")
            && (checked_address.contains("numeric_limits<std::uint64_t>::max()")
                || checked_address.contains("__builtin_mul_overflow")
                || checked_address.contains("checked_mul")),
        "V2 address helper must reject row/column bounds and multiplication overflow"
    );
    assert_eq!(
        score_kernel
            .matches("checked_resident_trim_prefilter_score_batch_address_v2(")
            .count(),
        1,
        "V2 score kernel must resolve every cell and parent ordinal through one checked helper"
    );
    let address_binding =
        identifier_after(score_kernel, "ResidentTrimPrefilterScoreBatchAddressV2 ");
    let failed_address = braced_item(
        score_kernel,
        "if (!checked_resident_trim_prefilter_score_batch_address_v2(",
    );
    assert!(
        failed_address.contains("return;"),
        "V2 score kernel must fail closed when checked address resolution fails"
    );
    let compact_kernel: String = score_kernel
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert!(
        compact_kernel.contains(&format!(
            "batch->batch_values_bar_major[{address_binding}.local_offset]"
        )) && (compact_kernel.contains(&format!(
            "batch->batch_validity_u4,{address_binding}.local_offset"
        )) || compact_kernel.contains(&format!(
            "batch->batch_validity_u4[{address_binding}.local_offset]"
        ))) && compact_kernel.contains(&format!(
            "column_scores[{address_binding}.global_parent_ordinal]"
        )),
        "V2 kernel must use the checked output for values, validity, and global score destination"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn score_batch_addressing_honors_padded_stride_and_noncontiguous_parent_ordinals() {
    let resolved = neoethos_gpu_cuda::resident_trim_prefilter_v1::
        checked_resident_trim_prefilter_score_batch_address_v2(
            2,
            1,
            3,
            8,
            40,
            &[41, 3, 97],
        )
        .expect("valid padded score-batch address");

    assert_eq!(resolved.local_offset(), 17);
    assert_eq!(resolved.global_parent_ordinal(), 3);
}

#[test]
fn v2_selected_map_exposes_typed_seal_and_public_capacity_invariant() {
    let header = read_required("native/resident_trim_prefilter_v1_abi.cuh");
    let rust = read_required("src/resident_trim_prefilter_v1.rs");
    let cuda = read_required("native/resident_trim_prefilter_v1.cu");

    require_all(
        &header,
        &[
            "constexpr std::uint32_t NEO_RESIDENT_TRIM_PREFILTER_SELECTED_MAP_READ_ABI_V2 = 2U;",
            "struct NeoResidentTrimPrefilterSelectedMapReadV2 {",
            "std::uint32_t abi_version;",
            "std::uint32_t reserved;",
            "std::uint64_t selected_capacity;",
            "std::uint64_t selected_count;",
            "std::uint64_t selected_map_readback_bytes;",
            "std::uint32_t* selected_global_parent_ordinals_host;",
            "cudaEvent_t selected_map_readback_ready_event;",
            "seal_resident_trim_prefilter_selected_map_v2(",
            "read_resident_trim_prefilter_selected_map_v2(",
        ],
    );
    require_all(
        &rust,
        &[
            "const SELECTED_MAP_READ_ABI_VERSION_V2: u32 = 2;",
            "struct RawResidentTrimPrefilterSelectedMapReadV2 {",
            "abi_version: u32",
            "reserved: u32",
            "selected_capacity: u64",
            "selected_count: u64",
            "selected_map_readback_bytes: u64",
            "selected_global_parent_ordinals_host: *mut u32",
            "selected_map_readback_ready_event: *mut c_void",
            "abi_version: SELECTED_MAP_READ_ABI_VERSION_V2",
            "raw_read.abi_version != SELECTED_MAP_READ_ABI_VERSION_V2",
            "pub struct SealedResidentTrimPrefilterSelectedMapV2",
            "pub struct BoundedResidentTrimPrefilterSelectedMapReadV2",
            "seal_selected_map_v2(",
            "read_bounded_selected_map_v2(",
            "pub fn selected_capacity(&self) -> u64",
            "pub fn selected_count(&self) -> u64",
            "pub fn capacity_covers_actual_v2(&self) -> bool",
            "selected_capacity",
            "selected_count",
            "selected_map_sha256",
            "selected_map_readback_bytes",
        ],
    );
    require_declaration_repr_c(&rust, "struct RawResidentTrimPrefilterSelectedMapReadV2 {");
    require_c_and_rust_size(
        &header,
        &rust,
        "NeoResidentTrimPrefilterSelectedMapReadV2",
        "RawResidentTrimPrefilterSelectedMapReadV2",
        48,
    );
    require_c_and_rust_offsets(
        &header,
        &rust,
        "NeoResidentTrimPrefilterSelectedMapReadV2",
        "RawResidentTrimPrefilterSelectedMapReadV2",
        &[
            ("abi_version", 0),
            ("reserved", 4),
            ("selected_capacity", 8),
            ("selected_count", 16),
            ("selected_map_readback_bytes", 24),
            ("selected_global_parent_ordinals_host", 32),
            ("selected_map_readback_ready_event", 40),
        ],
    );
    require_all(
        &cuda,
        &["read->abi_version != NEO_RESIDENT_TRIM_PREFILTER_SELECTED_MAP_READ_ABI_V2"],
    );
    require_by_value_type(
        braced_item(&rust, "pub fn read_bounded_selected_map_v2("),
        "SealedResidentTrimPrefilterSelectedMapV2",
        "bounded selected-map read",
    );
    require_move_only_type(&rust, "SealedResidentTrimPrefilterSelectedMapV2");
    assert!(
        braced_item(&rust, "pub fn capacity_covers_actual_v2(&self) -> bool")
            .contains("selected_count <= self.selected_capacity"),
        "public selected-map receipt must expose capacity >= actual"
    );
}
