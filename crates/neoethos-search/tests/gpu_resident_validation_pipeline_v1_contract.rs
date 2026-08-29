use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn pipeline_source() -> String {
    let path = manifest_dir().join("src/gpu_full_discovery/gpu_resident_validation_pipeline_v1.rs");
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
            "resident validation pipeline is missing {token:?}"
        );
    }
}

#[test]
fn exact_five_validation_stages_are_strict_gpu_and_identity_bound() {
    let source = pipeline_source();
    let stages = section(
        &source,
        "const GPU_RESIDENT_VALIDATION_STAGES_V1: [PipelineStage; 5] = [",
        "];",
    );
    for stage in [
        "PipelineStage::WalkForward",
        "PipelineStage::Cpcv",
        "PipelineStage::Pbo",
        "PipelineStage::RiskDiagnostics",
        "PipelineStage::CanonicalReplay",
    ] {
        assert_eq!(
            stages.matches(stage).count(),
            1,
            "missing exact stage {stage}"
        );
    }
    assert_eq!(stages.matches("PipelineStage::").count(), 5);
    require_all(
        &source,
        &[
            "StrictGpuOnlyFullDiscoveryPermitV2",
            "SealedResidentPortfolioConstraintsOutcomeV1",
            "StageGpuCapability::StrictGpu",
            "canonical_search_input_receipt_sha256",
            "resident_input_content_sha256",
            "validation_plan_semantics_sha256",
        ],
    );
}

#[test]
fn move_only_portfolio_outcome_is_consumed_into_one_build_bound_validation_run() {
    let source = pipeline_source();
    let run = section(&source, "pub struct GpuResidentValidationRunV1 {", "\n}");
    require_all(
        run,
        &[
            "selected_cuda_ordinal: u32",
            "cuda_device_identity_sha256: String",
            "cuda_device_uuid: String",
            "primary_context_identity_sha256: String",
            "cuda_build_manifest_sha256: String",
            "cuda_toolkit_version: String",
            "cccl_version: String",
            "native_validation_kernel_build_sha256: String",
            "cuda_compile_math_mode_sha256: String",
            "resident_parent_identity_sha256: String",
            "split_plan_store: GpuResidentValidationSplitPlanStoreV1",
            "metric_store: GpuResidentValidationMetricStoreV1",
            "scenario_trade_ledgers: ValidationScenarioTradeLedgerStoreV1",
            "cub_scratch_arena: CubScratchArenaV1",
            "run_stream: GpuRunStreamV1",
            "run_stream_identity_sha256: String",
            "run_identity_sha256: String",
        ],
    );
    assert!(
        !run.contains("pub "),
        "validation run fields must remain private"
    );
    require_all(
        &source,
        &[
            "portfolio_outcome: SealedResidentPortfolioConstraintsOutcomeV1",
            "consume_portfolio_outcome_into_validation_run_v1(",
            "validate_portfolio_outcome_identity_v1(",
            "validate_exact_validation_build_identity_v1(",
            "temporary_storage_plan_sha256",
            "validate_validation_run_against_permit_v1(",
        ],
    );
    for forbidden in [
        "&SealedResidentPortfolioConstraintsOutcomeV1",
        "impl Clone for SealedResidentPortfolioConstraintsOutcomeV1",
        "impl Copy for SealedResidentPortfolioConstraintsOutcomeV1",
        "impl Default for SealedResidentPortfolioConstraintsOutcomeV1",
        "cuda::execution::determinism::run_to_run",
        "run_to_run_determinism_environment_sha256",
    ] {
        assert!(
            !source.contains(forbidden),
            "validation input/build authority contains stale or mintable path {forbidden:?}"
        );
    }
}

#[test]
fn walkforward_uses_resident_contiguous_views_device_base_and_device_diagnostics() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "WALKFORWARD_GEOMETRY_SEMANTICS_V1",
            "walkforward_input_rows",
            "walkforward_split_count",
            "walkforward_train_ratio_f64_bits",
            "walkforward_embargo_bars",
            "walkforward_window_rows_floor_division_v1(",
            "MIN_WALKFORWARD_WINDOW_ROWS_V1: u64 = 80",
            "walkforward_train_end_floor_v1(",
            "walkforward_test_start_checked_add_embargo_v1(",
            "MIN_WALKFORWARD_TRAIN_ROWS_V1: u64 = 40",
            "MIN_WALKFORWARD_TEST_ROWS_V1: u64 = 40",
            "walkforward_qualifying_split_bitmap_sha256",
            "build_walkforward_split_descriptors_on_device_v1(",
            "bind_resident_contiguous_validation_views_v1(",
            "derive_window_adaptive_base_on_device_v1(",
            "launch_walkforward_population_metrics_v1(",
            "launch_device_risk_diagnostics_v1(",
            "walkforward_semantics_sha256",
            "walkforward_metric_rows_readback_count == 0",
            "walkforward_trade_rows_readback_count == 0",
        ],
    );
    for forbidden in [
        "embargoed_walkforward_population",
        "WindowEvaluation",
        "signals_per_gene: &[Vec<i8>]",
        "simulate_trades_core",
        "adaptive_upload_bytes",
    ] {
        assert!(
            !source.contains(forbidden),
            "walk-forward retains host/reupload path {forbidden:?}"
        );
    }
}

#[test]
fn cpcv_builds_ordered_indices_on_device_without_host_gathers_or_per_fold_uploads() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "CPCV_GEOMETRY_SEMANTICS_V1",
            "resolve_cpcv_tail_capped_rows_v1(",
            "cpcv_tail_offset_checked_sub_v1(",
            "cpcv_group_size_floor_division_v1(",
            "cpcv_last_group_absorbs_remainder_v1(",
            "cpcv_lexicographic_test_group_combinations_v1(",
            "cpcv_purge_rows_ceil_v1(",
            "cpcv_embargo_rows_ceil_v1(",
            "trim_each_train_group_against_every_test_group_v1(",
            "refuse_empty_cpcv_train_or_test_split_v1(",
            "build_cpcv_split_descriptors_on_device_v1(",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSelectFlagged",
            "bind_device_owned_ordered_indices_v1(",
            "launch_batched_cpcv_population_metrics_v1(",
            "cpcv_geometry_semantics_sha256",
            "ordered_index_host_to_device_bytes == 0",
            "gathered_parent_host_to_device_bytes == 0",
            "cpcv_metric_rows_readback_count == 0",
        ],
    );
    for forbidden in [
        "Array2::<f64>::zeros",
        "absolute_idx: Vec<usize>",
        "gathered_ind",
        "gathered_smc",
        "validation_genes_population_gathered",
    ] {
        assert!(
            !source.contains(forbidden),
            "CPCV retains host gather/reupload via {forbidden:?}"
        );
    }
}

#[test]
fn pbo_v2_refuses_nonfinite_and_matches_the_versioned_cpu_oracle_exactly() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "PBO_SEMANTICS_V2",
            "PboV2Error::NonFiniteMetric",
            "reject_nonfinite_pbo_metric_v2(",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceReduceArgMax",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSegmentedRadixSort",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceReduce",
            "stable_tie_break_candidate_identity_v1(",
            "canonical_f64_total_order_key_v1(",
            "pbo_candidate_order_identity_sha256",
            "pbo_tie_policy_sha256",
            "lower_middle_index_v2((candidate_count - 1) / 2)",
            "champion_oos_value <= lower_middle_oos_value",
            "pbo_cpu_oracle_v2(",
            "assert_gpu_pbo_matches_cpu_oracle_v2(",
            "pbo_semantics_v2_sha256",
            "pbo_metric_rows_readback_count == 0",
        ],
    );
    for forbidden in [
        "partial_cmp",
        "unwrap_or(std::cmp::Ordering::Equal)",
        "oos_perf.clone()",
        "sort_by(",
        "median_on_host",
    ] {
        assert!(
            !source.contains(forbidden),
            "PBO V2 retains host/weak/nonfinite ordering via {forbidden:?}"
        );
    }
}

#[test]
fn risk_and_canonical_replay_consume_exact_scenario_bound_device_trade_ledgers() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "ValidationScenarioTradeLedgerStoreV1",
            "ValidationScenarioIdentityV1",
            "walkforward_scenario_identity_sha256",
            "cpcv_scenario_identity_sha256",
            "canonical_replay_scenario_identity_sha256",
            "bind_trade_ledger_to_scenario_settings_content_v1(",
            "validate_trade_ledger_cost_conversion_identity_v1(",
            "validate_written_segments_against_count_scan_v1(",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceScan",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceRunLengthEncode",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSegmentedReduce",
            "launch_path_dependent_risk_state_v1(",
            "launch_canonical_replay_from_resident_parent_v1(",
            "risk_diagnostic_semantics_sha256",
            "canonical_replay_semantics_sha256",
            "risk_trade_rows_readback_count == 0",
            "canonical_replay_metric_rows_readback_count == 0",
        ],
    );
    for forbidden in [
        "walkforward_risk_diagnostics_from_trades",
        "fast_evaluate_strategy_core",
        "Vec::<f64>::new()",
        "Vec::<usize>::new()",
    ] {
        assert!(
            !source.contains(forbidden),
            "risk/canonical replay retains host or unbound diagnostics via {forbidden:?}"
        );
    }
}

#[test]
fn strict_validation_has_zero_cpu_fallback_full_readback_or_intermediate_sync() {
    let source = pipeline_source();
    let execute = section(
        &source,
        "fn execute_gpu_resident_validation_pipeline_v1(",
        "\n}\n\n",
    );
    for forbidden in [
        "FeatureFrame",
        "Vec<Vec<i8>>",
        "Vec<Vec<Trade>>",
        "rayon",
        ".par_iter",
        "copy_to_host",
        "read_metrics",
        "read_diagnostics",
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
        ".synchronize(",
        "cpu_forced",
        "FallbackPolicy::AllowCpu",
        "RecomputeOnCpu",
    ] {
        assert!(
            !execute.contains(forbidden),
            "strict validation crosses/fallbacks to host via {forbidden:?}"
        );
    }
    require_all(
        &source,
        &[
            "full_metric_rows_readback_count == 0",
            "full_trade_rows_readback_count == 0",
            "intermediate_explicit_synchronization_count == 0",
            "intermediate_host_decision_count == 0",
            "ValidationTransferAccountingMismatch",
        ],
    );
}

#[test]
fn validation_chunks_are_calibrated_and_handoff_stays_opaque_with_zero_readback() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "ExactDeviceBuildShapeCalibrationReceiptV1",
            "validate_calibration_against_validation_shape_v1(",
            "ConservativeObservableChunkPlanV1",
            "cudaEventQuery",
            "observe_cancellation_between_chunks_v1(",
            "publish_progress_between_chunks_v1(",
            "#[must_use = \"resident validation outcome must be consumed by robustness\"]",
            "pub struct SealedResidentValidationOutcomeV1",
            "resident_gate_summary_store: GpuResidentValidationGateSummaryStoreV1",
            "detailed_artifact_manifest_identity_sha256",
            "compact_control_readback_count == 0",
            "ResearchOnly",
            "NotPromotionEligible",
        ],
    );
    for forbidden in [
        "interpolate_from_other_device",
        "scale_from_reference_device",
        "Deserialize",
        "impl Clone for SealedResidentValidationOutcomeV1",
        "impl Default for SealedResidentValidationOutcomeV1",
        "pub fn from_hash",
        "bounded_gate_summaries",
        "compact_control_readback_count == 1",
        "pub folds:",
        "pub trades:",
        "pub metric_matrix:",
    ] {
        assert!(
            !source.contains(forbidden),
            "validation outcome/calibration weakens authority via {forbidden:?}"
        );
    }
}
