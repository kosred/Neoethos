use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn read_benchmark_source() -> String {
    let path = manifest_dir().join("src/bin/full_discovery_paired_benchmark_v1.rs");
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
            "paired full-discovery benchmark is missing {token:?}"
        );
    }
}

#[test]
fn controller_launches_exactly_one_isolated_cpu_reference_and_one_strict_gpu_candidate() {
    let benchmark_source = read_benchmark_source();
    require_all(
        &benchmark_source,
        &[
            "const BENCHMARK_CHILD_PROCESS_COUNT_V1: usize = 2;",
            "enum FullDiscoveryBenchmarkProcessRoleV1",
            "Controller",
            "CpuReference",
            "StrictGpuCandidate",
            "std::process::Command",
            "spawn_isolated_child_v1(",
            "--benchmark-process-role",
        ],
    );

    let controller = section(&benchmark_source, "fn run_controller_v1(", "\n}");
    require_all(
        controller,
        &[
            "FullDiscoveryBenchmarkProcessRoleV1::CpuReference",
            "FullDiscoveryBenchmarkProcessRoleV1::StrictGpuCandidate",
            "BENCHMARK_CHILD_PROCESS_COUNT_V1",
        ],
    );
}

#[test]
fn both_processes_consume_one_immutable_whole_cycle_semantic_case_identity() {
    let benchmark_source = read_benchmark_source();
    let case = section(
        &benchmark_source,
        "struct FullDiscoveryPairedBenchmarkCaseV1 {",
        "\n}",
    );
    require_all(
        case,
        &[
            "scope: FullDiscoveryBenchmarkScopeV1",
            "source_generation_set_identity_sha256: String",
            "dataset_recipe_semantic_identity_sha256: String",
            "feature_schema_semantic_identity_sha256: String",
            "feature_plan_semantic_identity_sha256: String",
            "resolved_search_semantic_config_sha256: String",
            "search_seed: u64",
            "selected_cuda_ordinal: u32",
            "semantic_case_identity_sha256: String",
            "cache_policy: BenchmarkCachePolicyV1",
        ],
    );
    require_all(
        &benchmark_source,
        &[
            "enum FullDiscoveryBenchmarkScopeV1",
            "WholeCycleEndToEnd",
            "write_immutable_semantic_case_once_v1(",
            "validate_semantic_case_identity_before_run_v1(",
            "require_same_semantic_case_identity_v1(",
            "materialize_lane_local_canonical_search_input_v1(",
            "require_exact_semantic_content_root_v1(",
        ],
    );
    for forbidden in [
        "cpu_case.search_seed =",
        "gpu_case.search_seed =",
        "cpu_case.resolved_search_semantic_config_sha256 =",
        "gpu_case.resolved_search_semantic_config_sha256 =",
        "canonical_search_input_receipt_sha256: String",
        "reuse_prebuilt_search_input_for_both_lanes_v1(",
        "SearchOnlyOverPrebuiltFeatures",
    ] {
        assert!(
            !benchmark_source.contains(forbidden),
            "paired whole-cycle benchmark weakens lane-local materialization through {forbidden:?}"
        );
    }
}

#[test]
fn child_reports_bind_all_sixteen_stage_receipts_and_complete_transfer_counters() {
    let benchmark_source = read_benchmark_source();
    let report = section(
        &benchmark_source,
        "struct FullDiscoveryBenchmarkChildReportV1 {",
        "\n}",
    );
    require_all(
        report,
        &[
            "semantic_case_identity_sha256: String",
            "source_generation_set_identity_sha256: String",
            "dataset_recipe_semantic_identity_sha256: String",
            "feature_schema_semantic_identity_sha256: String",
            "feature_plan_semantic_identity_sha256: String",
            "canonical_content_root_sha256: String",
            "lane_input_execution_receipt_sha256: String",
            "resolved_search_semantic_config_sha256: String",
            "search_seed: u64",
            "cache_policy: BenchmarkCachePolicyV1",
            "semantic_result_identity_sha256: String",
            "benchmark_only_cpu_semantic_result_evidence_sha256: Option<String>",
            "gpu_only_compact_discovery_receipt_identity_sha256: Option<String>",
            "stage_receipts: [FullDiscoveryStageBenchmarkReceiptV1; 16]",
        ],
    );

    let stage_receipt = section(
        &benchmark_source,
        "struct FullDiscoveryStageBenchmarkReceiptV1 {",
        "\n}",
    );
    require_all(
        stage_receipt,
        &[
            "semantic_input_identity_sha256: String",
            "semantic_output_identity_sha256: String",
            "semantic_result_identity_sha256: String",
            "execution_receipt_identity_sha256: String",
            "execution_engine_identity_sha256: String",
            "execution_device_identity_sha256: String",
            "execution_build_manifest_sha256: String",
            "transfer_counters: FullDiscoveryStageTransferCountersV1",
        ],
    );

    let counters = section(
        &benchmark_source,
        "struct FullDiscoveryStageTransferCountersV1 {",
        "\n}",
    );
    require_all(
        counters,
        &[
            "host_to_device_transfer_count: u64",
            "host_to_device_bytes: u64",
            "device_to_host_transfer_count: u64",
            "device_to_host_bytes: u64",
            "explicit_synchronization_count: u64",
            "metric_rows_readback_count: u64",
            "accepted_trade_total_readback_count: u64",
            "compact_result_readback_count: u64",
            "compact_result_readback_bytes: u64",
        ],
    );
    require_all(
        &benchmark_source,
        &[
            "require_exactly_sixteen_stage_receipts_v1(",
            "require_complete_transfer_accounting_v1(",
        ],
    );
}

#[test]
fn comparison_requires_exact_semantics_and_distinct_truthful_execution_identities() {
    let benchmark_source = read_benchmark_source();
    require_all(
        &benchmark_source,
        &[
            "require_same_semantic_case_identity_v1(",
            "require_exact_semantic_source_recipe_schema_plan_v1(",
            "require_exact_semantic_content_root_v1(",
            "require_distinct_truthful_input_execution_receipts_v1(",
            "require_exact_stage_semantic_identities_v1(",
            "require_distinct_truthful_stage_execution_identities_v1(",
            "require_exact_semantic_result_identity_v1(",
            "require_cpu_benchmark_only_semantic_result_evidence_v1(",
            "require_gpu_only_compact_discovery_receipt_v1(",
            "require_compact_receipt_binds_semantic_result_v1(",
            "PairedBenchmarkIdentityMismatch",
            "PairedBenchmarkExecutionIdentityAliased",
        ],
    );
    for forbidden in [
        "require_exact_stage_receipt_identities_v1(",
        "require_exact_compact_receipt_identity_v1(",
        "require_exact_input_receipt_identity_v1(",
        "abs() <",
        "epsilon",
        "approximately_equal",
        "allow_result_mismatch",
    ] {
        assert!(
            !benchmark_source.contains(forbidden),
            "benchmark weakens exact parity through {forbidden:?}"
        );
    }
}

#[test]
fn cold_and_warm_cache_results_are_explicit_separate_policies() {
    let benchmark_source = read_benchmark_source();
    require_all(
        &benchmark_source,
        &[
            "enum BenchmarkCachePolicyV1",
            "Cold",
            "Warm",
            "--cache-policy",
            "prepare_cold_cache_policy_v1(",
            "prepare_warm_cache_policy_v1(",
            "require_same_cache_policy_v1(",
        ],
    );
    assert!(
        !benchmark_source.contains("unwrap_or(BenchmarkCachePolicyV1::Warm)"),
        "cache temperature must be explicit rather than silently defaulting warm"
    );
}

#[test]
fn cpu_reference_is_created_by_process_isolation_and_a_sealed_no_gpu_probe() {
    let benchmark_source = read_benchmark_source();
    require_all(
        &benchmark_source,
        &[
            "isolate_cpu_reference_from_cuda_devices_v1(",
            "probe_after_process_isolation_v1(",
            "require_sealed_no_compatible_gpu_probe_v1(",
            "require_exact_gpu_ordinal_and_build_v1(",
        ],
    );
    for forbidden in [
        "cpu_forced",
        "cpu-forced",
        "GPU_PREFERRED",
        "FallbackPolicy::AllowCpu",
        "NEOETHOS_REQUIRE_GPU=0",
        "allow_cpu: true",
    ] {
        assert!(
            !benchmark_source.contains(forbidden),
            "benchmark reuses a production CPU/fallback escape {forbidden:?}"
        );
    }
}

#[test]
fn benchmark_output_is_evidence_only_and_cannot_mint_production_authority() {
    let benchmark_source = read_benchmark_source();
    require_all(
        &benchmark_source,
        &[
            "BenchmarkOnly",
            "ResearchOnly",
            "NotPromotionEligible",
            "write_paired_benchmark_evidence_v1(",
        ],
    );
    for forbidden in [
        "impl From<FullDiscoveryBenchmarkChildReportV1",
        "StrictGpuOnlyFullDiscoveryPermitV2 {",
        "SealedCompactGpuOnlyDiscoveryReceiptV1 {",
        "promotion_permit",
        "live_trading_permit",
    ] {
        assert!(
            !benchmark_source.contains(forbidden),
            "benchmark evidence can mint production authority via {forbidden:?}"
        );
    }
}

#[test]
fn launch_sizing_uses_exact_measured_calibration_or_conservative_observable_chunks() {
    let benchmark_source = read_benchmark_source();
    require_all(
        &benchmark_source,
        &[
            "ExactDeviceBuildShapeCalibrationReceiptV1",
            "selected_cuda_ordinal",
            "cuda_device_identity_sha256",
            "cuda_build_manifest_sha256",
            "workload_shape_identity_sha256",
            "measured_scenario_bars_per_second",
            "measured_occupancy_knee",
            "calibration_identity_sha256",
            "validate_calibration_against_exact_run_v1(",
            "ConservativeObservableChunkPlanV1",
            "CalibrationDeviceBuildShapeMismatch",
            "observe_cancellation_between_chunks_v1(",
            "publish_progress_between_chunks_v1(",
        ],
    );
    for forbidden in [
        "SCENARIO_BARS_PER_SECOND",
        "843_000_000",
        "OCCUPANCY_KNEE",
        "16_384",
        "RTX 3090",
        "scale_from_reference_device",
        "interpolate_from_other_device",
    ] {
        assert!(
            !benchmark_source.contains(forbidden),
            "paired benchmark reuses or interpolates stale launch calibration {forbidden:?}"
        );
    }
}
