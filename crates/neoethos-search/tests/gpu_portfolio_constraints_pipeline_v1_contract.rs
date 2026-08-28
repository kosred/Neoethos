use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn pipeline_source() -> String {
    let path =
        manifest_dir().join("src/gpu_full_discovery/gpu_portfolio_constraints_pipeline_v1.rs");
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
            "resident portfolio-constraints pipeline is missing {token:?}"
        );
    }
}

#[test]
fn exact_prop_firm_and_candidate_correlation_stages_are_strict_and_run_owned() {
    let source = pipeline_source();
    let stages = section(
        &source,
        "const GPU_PORTFOLIO_CONSTRAINT_STAGES_V1: [PipelineStage; 2] = [",
        "];",
    );
    for stage in [
        "PipelineStage::PropFirmWindow",
        "PipelineStage::CandidateCorrelation",
    ] {
        assert_eq!(
            stages.matches(stage).count(),
            1,
            "portfolio constraints must contain {stage} exactly once"
        );
    }
    assert_eq!(stages.matches("PipelineStage::").count(), 2);

    let run = section(
        &source,
        "pub struct GpuResidentPortfolioConstraintsRunV1 {",
        "\n}",
    );
    require_all(
        run,
        &[
            "post_ga_outcome: SealedResidentPostGaOutcomeV1",
            "candidate_store: GpuResidentPortfolioCandidateStoreV1",
            "trade_state_store: GpuResidentTradeStateStoreV1",
            "correlation_workspace: GpuResidentCorrelationWorkspaceV1",
            "cub_scratch_arena: CubScratchArenaV1",
            "run_stream: GpuRunStreamV1",
            "run_identity_sha256: String",
        ],
    );
    assert!(
        !run.contains("pub "),
        "run-owned fields must remain private"
    );
}

#[test]
fn prop_firm_windows_use_device_state_and_official_segmented_primitives() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "launch_prop_firm_path_state_v1(",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceScan",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceRunLengthEncode",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSegmentedReduce",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSelectFlagged",
            "prop_firm_rule_semantics_sha256",
            "prop_firm_window_plan_sha256",
            "prop_firm_full_trade_readback_count == 0",
        ],
    );
    for forbidden in [
        "CustomDeviceScan",
        "CustomDeviceRunLengthEncode",
        "CustomDeviceSegmentedReduce",
        "plan_prop_firm_windows",
        "compute_prop_firm_pass_rate",
        "simulate_trades_core",
    ] {
        assert!(
            !source.contains(forbidden),
            "prop-firm stage retains host work or replaces official primitive via {forbidden:?}"
        );
    }
}

#[test]
fn correlation_uses_cublas_only_with_exact_authority_otherwise_exact_integer_statistics() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "enum CandidateCorrelationArithmeticAuthorityV1",
            "PinnedCublasDgemmF64",
            "ExactTernaryIntegerSufficientStatistics",
            "validate_pinned_cublas_authority_against_exact_run_v1(",
            "launch_exact_ternary_integer_statistics_v1(",
            "UnprovenCublasCorrelationAuthority",
            "correlation_semantics_sha256",
            "stable_tie_break_candidate_identity_v1(",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSelectFlagged",
        ],
    );
    let cublas = section(
        &source,
        "pub struct PinnedCublasCorrelationReceiptV1 {",
        "\n}",
    );
    require_all(
        cublas,
        &[
            "selected_cuda_ordinal: u32",
            "cuda_device_identity_sha256: String",
            "cuda_build_manifest_sha256: String",
            "cublas_version: String",
            "cublas_algorithm_id: String",
            "cublas_math_mode: String",
            "cublas_atomics_mode: String",
            "workspace_bytes: u64",
            "stream_identity_sha256: String",
            "receipt_identity_sha256: String",
        ],
    );
    assert!(
        !cublas.contains("pub "),
        "cuBLAS receipt fields must remain private"
    );
    for forbidden in [
        "pearson_corr_i8",
        "spearman_corr_i8",
        "partial_cmp",
        "approximately_equal",
        "UnpinnedCublas",
    ] {
        assert!(
            !source.contains(forbidden),
            "correlation retains host/unproven arithmetic via {forbidden:?}"
        );
    }
}

#[test]
fn algorithm_receipt_binds_exact_device_build_stream_cccl_and_determinism_environment() {
    let source = pipeline_source();
    let receipt = section(
        &source,
        "pub struct PortfolioConstraintAlgorithmReceiptV1 {",
        "\n}",
    );
    require_all(
        receipt,
        &[
            "selected_cuda_ordinal: u32",
            "cuda_device_identity_sha256: String",
            "cuda_build_manifest_sha256: String",
            "cccl_version: String",
            "stream_identity_sha256: String",
            "temporary_storage_plan_sha256: String",
            "run_to_run_determinism_environment_sha256: String",
            "prop_firm_algorithm_sha256: String",
            "candidate_correlation_algorithm_sha256: String",
            "receipt_identity_sha256: String",
        ],
    );
    assert!(
        !receipt.contains("pub "),
        "algorithm receipt fields must remain private"
    );
    require_all(
        &source,
        &[
            "cuda::execution::determinism::run_to_run",
            "reuse_run_owned_temporary_storage_v1(",
            "validate_algorithm_receipt_against_permit_v1(",
        ],
    );
    for forbidden in [
        "CrossDeviceBitwiseDeterminism",
        "device_agnostic_determinism",
    ] {
        assert!(
            !source.contains(forbidden),
            "receipt overclaims determinism through {forbidden:?}"
        );
    }
}

#[test]
fn strict_execution_has_zero_cpu_fallback_full_matrix_readback_or_intermediate_sync() {
    let source = pipeline_source();
    let execute = section(
        &source,
        "fn execute_gpu_resident_portfolio_constraints_v1(",
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
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
        ".synchronize(",
        "cpu_forced",
        "FallbackPolicy::AllowCpu",
        "RecomputeOnCpu",
    ] {
        assert!(
            !execute.contains(forbidden),
            "strict portfolio constraints cross/fallback to host via {forbidden:?}"
        );
    }
    require_all(
        &source,
        &[
            "full_signal_matrix_readback_count == 0",
            "full_correlation_matrix_readback_count == 0",
            "full_metric_rows_readback_count == 0",
            "full_trade_rows_readback_count == 0",
            "intermediate_explicit_synchronization_count == 0",
            "intermediate_host_decision_count == 0",
            "PortfolioConstraintTransferAccountingMismatch",
        ],
    );
}

#[test]
fn chunks_are_calibrated_observable_and_cancellable_without_stale_device_interpolation() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "ExactDeviceBuildShapeCalibrationReceiptV1",
            "validate_calibration_against_portfolio_shape_v1(",
            "ConservativeObservableChunkPlanV1",
            "cudaEventQuery",
            "observe_cancellation_between_chunks_v1(",
            "publish_progress_between_chunks_v1(",
            "CalibrationDeviceBuildShapeMismatch",
        ],
    );
    for forbidden in [
        "SCENARIO_BARS_PER_SECOND",
        "OCCUPANCY_KNEE",
        "interpolate_from_other_device",
        "scale_from_reference_device",
    ] {
        assert!(
            !source.contains(forbidden),
            "portfolio chunk plan uses stale/unbound calibration via {forbidden:?}"
        );
    }
}

#[test]
fn output_is_opaque_resident_and_only_bounded_compact_control_data_can_cross() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "pub struct SealedResidentPortfolioConstraintsOutcomeV1",
            "resident_outcome_identity_sha256",
            "bounded_selected_candidate_count",
            "serialized_byte_ceiling",
            "compact_control_readback_count == 1",
            "ResearchOnly",
            "NotPromotionEligible",
        ],
    );
    for forbidden in [
        "Deserialize",
        "impl Default for SealedResidentPortfolioConstraintsOutcomeV1",
        "pub fn from_hash",
        "pub signals:",
        "pub trades:",
        "pub correlation_matrix:",
    ] {
        assert!(
            !source.contains(forbidden),
            "portfolio outcome leaks/reconstructs authority via {forbidden:?}"
        );
    }
}
