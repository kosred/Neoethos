use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
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
            "resident trim/prefilter V1 source is missing {token:?}"
        );
    }
}

fn resident_source() -> String {
    read_required("src/gpu_full_discovery/gpu_resident_trim_prefilter_view_v1.rs")
}

#[test]
fn v1_pins_the_exact_current_non_fallback_host_semantics() {
    let host = read_required("src/discovery.rs");
    let resident = resident_source();

    require_all(
        &host,
        &[
            "pub const DEFAULT_OOS_HOLDOUT_FRACTION: f64 = 0.2;",
            ".floor() as usize;",
            "split_at >= 64",
            "fn row_cap_for_config(config: &DiscoveryConfig) -> usize",
            "(global, tf) => global.min(tf)",
            "start_idx = available_rows - row_cap;",
            "fn rolling_atr_f64(ohlcv: &Ohlcv, period: usize)",
            "let long_tp = entry + take_distance + cost;",
            "let long_sl = entry - stop_distance + cost;",
            "let short_tp = entry - take_distance - cost;",
            "let short_sl = entry + stop_distance - cost;",
            "let (windows, folds_available) = prefilter_fit_windows(n_rows, spec);",
            "neoethos_data::core::stats_f64::pearson_pairwise(&xs, &ys)",
            "worst = worst.min(a);",
            "best = best.max(a);",
            "b.1.partial_cmp(&a.1)",
            ".then_with(|| a.0.cmp(&b.0))",
            "keep_indices.sort(); // Maintain original order",
        ],
    );
    require_all(
        &resident,
        &[
            "ResidentTrimCorrelationPrefilterSemanticsV1",
            "RESIDENT_TRIM_CORRELATION_PREFILTER_SEMANTICS_V1",
            "DEFAULT_OOS_HOLDOUT_FRACTION_BITS_V1",
            "MINIMUM_IN_SAMPLE_ROWS_V1",
            "MINIMUM_PAIRWISE_SAMPLES_V1",
            "MINIMUM_DECIDED_FIRST_PASSAGE_LABELS_V1",
            "MAXIMUM_REFIT_FOLDS_V1",
            "StrictMathFlagsV1",
            "ResearchOnly",
            "NotPromotionEligible",
        ],
    );
}

#[test]
fn degenerate_labels_fail_loud_and_never_change_the_target() {
    let source = resident_source();
    require_all(
        &source,
        &[
            "InsufficientDecidedFirstPassageLabels",
            "insufficient-decided-labels-invalidates-device-seal",
            "enqueue_invalidate_device_seal_if_insufficient_decisions_v1(",
        ],
    );
    for forbidden in [
        "label_fell_back_to_forward_return",
        "forward_return",
        "close[i + 1] - close[i]",
        "best_effort",
        "fallback_mode",
    ] {
        assert!(
            !source.contains(forbidden),
            "strict resident prefilter silently changes semantics through {forbidden:?}"
        );
    }
}

#[test]
fn one_shot_authority_binds_store_schema_device_build_context_and_stream() {
    let source = resident_source();
    let run = section(&source, "pub struct ResidentTrimPrefilterRunV1 {", "\n}");
    require_all(
        run,
        &[
            "device_run: ResidentTrimPrefilterDeviceRunV1",
            "resolved_plan: ResidentTrimPrefilterResolvedPlanV1",
            "selected_cuda_ordinal: u32",
            "primary_context_identity_sha256: [u8; 32]",
            "run_stream_identity_sha256: [u8; 32]",
            "cuda_build_manifest_sha256: [u8; 32]",
        ],
    );
    assert!(
        !run.contains("pub "),
        "native owner fields must stay private"
    );
    require_all(
        &source,
        &[
            "consume_same_run_parent_and_schema_v1(",
            "ResidentTrimPrefilterDeviceRunV1",
            "SealedResidentColumnClassificationV1",
            "ResidentTrimPrefilterParentImportV1",
            "begin_resident_trim_prefilter_device_run_v1(",
        ],
    );
    for forbidden in [
        "impl Clone for ResidentTrimPrefilterRunV1",
        "impl Default for ResidentTrimPrefilterRunV1",
        "pub fn from_raw",
        "pub fn raw_",
        "NonNull<NativeResidentTrimPrefilterRunV1>",
        "caller_schema_flags",
        "caller_selected_columns",
        "Device::get_device",
        "Context::new",
        "create_stream",
        "second_probe",
    ] {
        assert!(
            !source.contains(forbidden),
            "caller or second-route escape remains via {forbidden:?}"
        );
    }
}

#[test]
fn scopes_are_absolute_suffix_trimmed_and_share_one_compact_to_parent_map() {
    let source = resident_source();
    require_all(
        &source,
        &[
            "parent_row_count",
            "outer_split_at",
            "selection_row_start",
            "selection_row_end",
            "holdout_row_start",
            "holdout_row_end",
            "selection_row_start = outer_split_at - retained_selection_rows",
            "holdout_row_start = outer_split_at",
            "holdout_row_end = parent_row_count",
            "selected_compact_to_parent_columns_device",
            "selected_column_count_device",
            "same_selected_column_map_for_holdout",
            "canonical_search_input_receipt_sha256",
            "canonical_content_merkle_sha256",
            "normalization_fit_sha256",
            "feature_plan_sha256",
            "source_provenance_sha256",
        ],
    );
    for forbidden in [
        "FeatureFrame",
        "Ohlcv",
        "Cow<",
        "row_window(",
        "select_columns(",
        "rewrite_gene_indices_to_parent",
    ] {
        assert!(
            !source.contains(forbidden),
            "resident view identity crosses or rewrites host values through {forbidden:?}"
        );
    }
}

#[test]
fn sealed_schema_metadata_drives_state_template_and_timeframe_rules_on_device() {
    let source = resident_source();
    require_all(
        &source,
        &[
            "SealedResidentColumnClassificationV1",
            "column_class_flags_device",
            "timeframe_group_ids_device",
            "template_force_keep_flags_device",
            "column_classification_content_sha256",
            "ordered_feature_schema_sha256",
            "PREFILTER_STATE_FAMILY_SEMANTICS_V1",
            "TIMEFRAME_GROUP_SEMANTICS_V1",
            "TEMPLATE_FORCE_KEEP_SEMANTICS_V1",
            "MissingResidentSchemaClassificationAuthority",
            "SchemaClassificationIdentityMismatch",
        ],
    );
    for forbidden in [
        "ordered_feature_names()",
        "starts_with(\"regime_\")",
        "HashMap<",
        "HashSet<",
        "upload_schema",
        "copy_schema_to_device",
    ] {
        assert!(
            !source.contains(forbidden),
            "post-materialization CPU/schema upload escape via {forbidden:?}"
        );
    }
}

#[test]
fn prefilter_work_is_same_stream_resident_and_only_returns_an_opaque_handoff() {
    let source = resident_source();
    require_all(
        &source,
        &[
            "enqueue_first_passage_labels_v1(",
            "enqueue_exact_cpcv_fold_descriptors_v1(",
            "enqueue_pairwise_two_pass_f64_correlations_v1(",
            "enqueue_stable_score_index_rank_v1(",
            "enqueue_state_template_timeframe_quota_v1(",
            "enqueue_ascending_parent_column_map_v1(",
            "enqueue_trim_prefilter_device_seal_v1(",
            "same_stream_enqueue_count",
            "intermediate_host_wait_count() != 0",
            "intermediate_readback_count() != 0",
            "host_to_device_transfer_count() != 0",
            "device_to_host_transfer_count() != 0",
            "explicit_synchronization_count() != 0",
            "SealedResidentTrimPrefilterViewsV1",
        ],
    );
    for forbidden in [
        "cudaStreamSynchronize",
        "cudaEventSynchronize",
        ".synchronize(",
        "copy_to_host",
        "copy_from_host",
        "read_metrics",
        "Vec<f64>",
        "Vec<usize>",
        "rayon",
        "par_iter",
    ] {
        assert!(
            !source.contains(forbidden),
            "resident prefilter crosses to host via {forbidden:?}"
        );
    }
}

#[test]
fn every_retained_and_peak_buffer_is_charged_to_future_full_admission() {
    let source = resident_source();
    let receipt = section(
        &source,
        "pub(crate) struct ResidentTrimPrefilterMemoryReceiptV1 {",
        "\n}",
    );
    require_all(
        receipt,
        &[
            "long_labels_bytes: u64",
            "short_labels_bytes: u64",
            "label_census_bytes: u64",
            "fold_descriptor_bytes: u64",
            "column_score_bytes: u64",
            "column_instability_bytes: u64",
            "column_rankability_bytes: u64",
            "state_template_timeframe_metadata_bytes: u64",
            "radix_key_ping_pong_bytes: u64",
            "radix_index_ping_pong_bytes: u64",
            "timeframe_group_counter_bytes: u64",
            "selected_column_map_bytes: u64",
            "selected_column_count_bytes: u64",
            "cub_select_scratch_bytes: u64",
            "cub_radix_sort_scratch_bytes: u64",
            "device_seal_bytes: u64",
            "retained_device_bytes: u64",
            "peak_device_bytes: u64",
            "full_discovery_reserve_bytes: u64",
            "allocation_plan_sha256: [u8; 32]",
        ],
    );
    require_all(
        &source,
        &[
            "checked_trim_prefilter_memory_plan_v1(",
            "validate_against_full_discovery_admission_v1(",
            "MemoryPlanArithmeticOverflow",
            "FullDiscoveryAdmissionUndercharged",
        ],
    );
}

#[test]
fn source_is_feature_compiled_but_non_mintable_unwired_and_uncalled() {
    let source = resident_source();
    let lib = read_required("src/lib.rs");
    let prepared = read_required("src/prepared_discovery_run_input_v3.rs");
    let discovery = read_required("src/discovery.rs");
    let capability = read_required("src/gpu_native/capability.rs");

    require_all(
        &source,
        &[
            "ResidentTrimPrefilterIntegrationStateV1::Unwired",
            "MissingFullDiscoveryWorkspaceAdmission",
            "MissingResidentSchemaClassificationAuthority",
            "MissingPopulationCompactColumnMapConsumer",
            "MissingExactSelectedIndexDeviceParity",
        ],
    );
    require_all(
        &lib,
        &[
            "#[cfg(feature = \"gpu-b-native\")]",
            "#[path = \"gpu_full_discovery/gpu_resident_trim_prefilter_view_v1.rs\"]",
            "pub mod gpu_resident_trim_prefilter_view_v1;",
        ],
    );
    require_all(
        &source,
        &[
            "pub enum ResidentTrimPrefilterIntegrationStateV1",
            "pub enum ResidentTrimPrefilterErrorV1",
            "pub struct ResidentTrimPrefilterResolvedPlanV1",
            "pub struct ResidentTrimPrefilterRunV1",
            "pub struct SealedResidentTrimPrefilterViewsV1",
            "pub const fn resident_trim_prefilter_integration_state_v1()",
            "pub fn begin_gpu_resident_trim_prefilter_view_v1(",
            "pub fn execute_gpu_resident_trim_prefilter_view_v1(",
            "pub fn seal_gpu_resident_trim_prefilter_view_v1(",
            "pub const fn is_research_only(&self) -> bool",
            "pub const fn is_not_promotion_eligible(&self) -> bool",
            "pub const fn plan_identity_sha256(&self) -> [u8; 32]",
            "pub const fn allocation_plan_sha256(&self) -> [u8; 32]",
            "pub const fn has_zero_intermediate_host_boundary(&self) -> bool",
        ],
    );

    for type_name in [
        "ResidentTrimPrefilterResolvedPlanV1",
        "ResidentTrimPrefilterRunV1",
        "SealedResidentTrimPrefilterViewsV1",
    ] {
        let body = section(&source, &format!("pub struct {type_name} {{"), "\n}");
        assert!(
            !body.contains("pub "),
            "opaque exported type {type_name} exposes constructible authority fields"
        );
    }
    let compact_accessors = section(
        &source,
        "impl SealedResidentTrimPrefilterViewsV1 {",
        "impl ResidentTrimPrefilterSearchPlanV1",
    );
    for forbidden in [
        "from_raw",
        "as_ptr",
        "raw_pointer",
        "raw_event",
        "raw_buffer",
        "raw_context",
        "raw_stream",
        "-> &SealedResidentTrimPrefilterDeviceViewsV1",
    ] {
        assert!(
            !compact_accessors.contains(forbidden),
            "compact Search status surface leaks device authority via {forbidden:?}"
        );
    }
    for forbidden in [
        "impl Default for ResidentTrimPrefilter",
        "Serialize",
        "Deserialize",
        "pub fn from_raw",
        "allow(dead_code)",
        "expect(dead_code)",
    ] {
        assert!(
            !source.contains(forbidden),
            "unwired Search boundary becomes caller-mintable via {forbidden:?}"
        );
    }
    for caller in [&prepared, &discovery] {
        assert!(
            !caller.contains("begin_gpu_resident_trim_prefilter_view_v1"),
            "Discovery was switched before full workspace admission and device parity were sealed"
        );
    }
    assert!(
        !capability.contains("ResidentTrimPrefilter"),
        "feature compilation falsely promoted trim/prefilter to a StrictGpu stage"
    );
}
