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
fn one_run_owned_validation_store_binds_device_build_algorithms_splits_and_stream() {
    let source = pipeline_source();
    let run = section(&source, "pub struct GpuResidentValidationRunV1 {", "\n}");
    require_all(
        run,
        &[
            "selected_cuda_ordinal: u32",
            "cuda_device_identity_sha256: String",
            "cuda_build_manifest_sha256: String",
            "resident_parent_identity_sha256: String",
            "split_plan_store: GpuResidentValidationSplitPlanStoreV1",
            "metric_store: GpuResidentValidationMetricStoreV1",
            "trade_ledger: CompactContiguousTradeOutcomeStoreV1",
            "cub_scratch_arena: CubScratchArenaV1",
            "run_stream: GpuRunStreamV1",
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
            "cccl_version",
            "temporary_storage_plan_sha256",
            "cuda::execution::determinism::run_to_run",
            "run_to_run_determinism_environment_sha256",
            "validate_validation_run_against_permit_v1(",
        ],
    );
}

#[test]
fn walkforward_uses_resident_contiguous_views_device_base_and_device_diagnostics() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
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
fn pbo_argmax_sort_median_and_verdict_remain_device_resident_and_deterministic() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceReduceArgMax",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSegmentedRadixSort",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceReduce",
            "stable_tie_break_candidate_identity_v1(",
            "canonical_f64_total_order_key_v1(",
            "pbo_semantics_sha256",
            "pbo_metric_rows_readback_count == 0",
        ],
    );
    for forbidden in [
        "partial_cmp",
        "oos_perf.clone()",
        "sort_by(",
        "median_on_host",
    ] {
        assert!(
            !source.contains(forbidden),
            "PBO retains host/weak ordering via {forbidden:?}"
        );
    }
}

#[test]
fn risk_and_canonical_replay_consume_the_same_compact_device_trade_ledger() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "CompactContiguousTradeOutcomeStoreV1",
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
            "risk/canonical replay retains host diagnostics via {forbidden:?}"
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
fn validation_chunks_are_exactly_calibrated_observable_cancellable_and_compact() {
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
            "pub struct SealedResidentValidationOutcomeV1",
            "bounded_gate_summaries",
            "detailed_artifact_manifest_identity_sha256",
            "compact_control_readback_count == 1",
            "ResearchOnly",
            "NotPromotionEligible",
        ],
    );
    for forbidden in [
        "interpolate_from_other_device",
        "scale_from_reference_device",
        "Deserialize",
        "pub fn from_hash",
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
