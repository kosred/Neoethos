use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn pipeline_source() -> String {
    let path = manifest_dir().join("src/gpu_full_discovery/gpu_robustness_tail_pipeline_v1.rs");
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
            "resident robustness/tail pipeline is missing {token:?}"
        );
    }
}

#[test]
fn exact_robustness_and_forward_tail_stages_cover_three_strict_device_plans() {
    let source = pipeline_source();
    let stages = section(
        &source,
        "const GPU_ROBUSTNESS_TAIL_STAGES_V1: [PipelineStage; 2] = [",
        "];",
    );
    for stage in [
        "PipelineStage::RobustnessPermutationPlateau",
        "PipelineStage::ForwardTailReplay",
    ] {
        assert_eq!(
            stages.matches(stage).count(),
            1,
            "missing exact stage {stage}"
        );
    }
    assert_eq!(stages.matches("PipelineStage::").count(), 2);
    require_all(
        &source,
        &[
            "PermutationRobustnessPlanV1",
            "ParameterPlateauPlanV1",
            "ForwardTailReplayPlanV1",
            "StageGpuCapability::StrictGpu",
        ],
    );
}

#[test]
fn run_binds_exact_validation_tail_holdout_device_build_algorithms_and_determinism() {
    let source = pipeline_source();
    let run = section(
        &source,
        "pub struct GpuResidentRobustnessTailRunV1 {",
        "\n}",
    );
    require_all(
        run,
        &[
            "validation_outcome: SealedResidentValidationOutcomeV1",
            "sealed_resident_tail_authority:",
            "selected_cuda_ordinal: u32",
            "cuda_device_identity_sha256: String",
            "cuda_build_manifest_sha256: String",
            "canonical_search_input_receipt_sha256: String",
            "holdout_scope_identity_sha256: String",
            "tail_resident_content_sha256: String",
            "run_stream: GpuRunStreamV1",
            "run_identity_sha256: String",
        ],
    );
    assert!(
        !run.contains("pub "),
        "robustness/tail run fields must remain private"
    );
    require_all(
        &source,
        &[
            "cccl_version",
            "temporary_storage_plan_sha256",
            "cuda::execution::determinism::run_to_run",
            "run_to_run_determinism_environment_sha256",
            "validate_tail_authority_against_validation_run_v1(",
        ],
    );
}

#[test]
fn permutation_uses_versioned_philox_counter_mapping_and_cub_segmented_sort() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "CounterRngAlgorithmV1::Philox4x32_10",
            "search_seed",
            "run_identity_sha256",
            "candidate_identity",
            "permutation_index",
            "draw_index",
            "checked_counter_mapping_v1(",
            "rng_mapping_identity_sha256",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSegmentedRadixSortPairs",
            "stable_random_key_tie_break_original_index_v1(",
            "permutation_semantics_sha256",
        ],
    );
    for forbidden in [
        "curandState",
        "curand_init",
        "curandGenerate",
        "thread_rng",
        "StdRng",
        "shuffle(&mut",
        "CustomDeviceSegmentedRadixSort",
    ] {
        assert!(
            !source.contains(forbidden),
            "permutation uses stateful/host RNG or replaces CUB via {forbidden:?}"
        );
    }
}

#[test]
fn plateau_variants_reuse_resident_evaluator_and_reduce_verdicts_on_device() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "build_parameter_plateau_scenarios_on_device_v1(",
            "launch_resident_parameter_plateau_evaluation_v1(",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSegmentedReduce",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSelectFlagged",
            "parameter_plateau_semantics_sha256",
            "plateau_metric_rows_readback_count == 0",
            "plateau_parent_reupload_count == 0",
        ],
    );
    for forbidden in [
        "signals_for_gene_full",
        "simulate_trades_core",
        "variant.clone()",
        "CustomDeviceSegmentedReduce",
    ] {
        assert!(
            !source.contains(forbidden),
            "plateau stage retains host/redundant primitive path {forbidden:?}"
        );
    }
}

#[test]
fn forward_tail_replay_consumes_sealed_resident_view_and_device_trade_ledger() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "bind_sealed_resident_forward_tail_view_v1(",
            "launch_forward_tail_replay_from_resident_parent_v1(",
            "CompactContiguousTradeOutcomeStoreV1",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceScan",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceRunLengthEncode",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSegmentedReduce",
            "forward_tail_replay_semantics_sha256",
            "tail_parent_reupload_count == 0",
            "tail_full_trade_rows_readback_count == 0",
            "tail_full_metric_rows_readback_count == 0",
        ],
    );
    for forbidden in [
        "select_columns(",
        "signals_for_gene_full",
        "simulate_trades_core",
        "compute_discovery_forward_test_artifacts",
        "compute_discovery_prop_firm_artifacts",
    ] {
        assert!(
            !source.contains(forbidden),
            "forward-tail replay retains host materialization via {forbidden:?}"
        );
    }
}

#[test]
fn strict_execution_has_zero_cpu_fallback_full_readback_or_intermediate_sync() {
    let source = pipeline_source();
    let execute = section(
        &source,
        "fn execute_gpu_resident_robustness_tail_pipeline_v1(",
        "\n}\n\n",
    );
    for forbidden in [
        "FeatureFrame",
        "Vec<Vec<i8>>",
        "Vec<Trade>",
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
            "strict robustness/tail crosses/fallbacks to host via {forbidden:?}"
        );
    }
    require_all(
        &source,
        &[
            "full_metric_rows_readback_count == 0",
            "full_trade_rows_readback_count == 0",
            "intermediate_explicit_synchronization_count == 0",
            "intermediate_host_decision_count == 0",
            "RobustnessTailTransferAccountingMismatch",
        ],
    );
}

#[test]
fn calibrated_chunks_are_observable_cancellable_and_artifacts_never_cross_as_full_arrays() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "ExactDeviceBuildShapeCalibrationReceiptV1",
            "validate_calibration_against_robustness_tail_shape_v1(",
            "ConservativeObservableChunkPlanV1",
            "cudaEventQuery",
            "observe_cancellation_between_chunks_v1(",
            "publish_progress_between_chunks_v1(",
            "persist_device_artifact_by_bounded_chunk_v1(",
            "detailed_artifact_manifest_identity_sha256",
            "bounded_artifact_chunk_bytes",
            "CalibrationDeviceBuildShapeMismatch",
        ],
    );
    for forbidden in [
        "SCENARIO_BARS_PER_SECOND",
        "OCCUPANCY_KNEE",
        "interpolate_from_other_device",
        "scale_from_reference_device",
        "return_full_trade_array",
        "return_full_metric_matrix",
    ] {
        assert!(
            !source.contains(forbidden),
            "robustness/tail chunking leaks or uses stale authority via {forbidden:?}"
        );
    }
}

#[test]
fn output_is_opaque_compact_research_only_and_bound_to_persisted_artifacts() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "pub struct SealedResidentRobustnessTailOutcomeV1",
            "bounded_robustness_summaries",
            "bounded_forward_tail_summaries",
            "serialized_byte_ceiling",
            "detailed_artifact_manifest_identity_sha256",
            "outcome_identity_sha256",
            "compact_control_readback_count == 1",
            "ResearchOnly",
            "NotPromotionEligible",
        ],
    );
    for forbidden in [
        "Deserialize",
        "impl Default for SealedResidentRobustnessTailOutcomeV1",
        "pub fn from_hash",
        "pub signals:",
        "pub trades:",
        "pub metric_matrix:",
    ] {
        assert!(
            !source.contains(forbidden),
            "robustness/tail outcome leaks/reconstructs authority via {forbidden:?}"
        );
    }
}
