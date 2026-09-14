use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-gpu-cuda"))
}

fn read(relative: &str) -> String {
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

fn braced_item<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing source item {signature:?}"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing opening brace for {signature:?}"));
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("missing closing brace for {signature:?}");
}

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(
            source.contains(token),
            "metrics-only resident population boundary is missing {token:?}"
        );
    }
}

#[test]
fn additive_abi_returns_a_fixed_width_resident_metric_event_receipt() {
    let header = read("native/neoethos_gpu_cuda.h");
    let handle = section(
        &header,
        "struct NeoPopulationResidentMetricsHandleV1 {",
        "};",
    );
    require_all(
        handle,
        &[
            "std::uint32_t abi_version;",
            "std::uint32_t reserved;",
            "std::uint64_t event_id;",
            "std::uint64_t scenario_count;",
            "std::uint64_t month_capacity;",
            "std::uint64_t metric_rows_bytes;",
            "std::uint64_t monthly_pnls_bytes;",
            "std::uint64_t month_start_equities_bytes;",
            "std::uint64_t scenario_descriptor_bytes;",
            "std::uint64_t total_device_bytes;",
            "std::uint64_t outcome_bytes;",
            "std::uint64_t accepted_trade_total_bytes;",
        ],
    );
    require_all(
        &header,
        &[
            "neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(",
            "NeoPopulationResidentMetricsHandleV1* resident_metrics",
            "Compatibility/DeviceParityOnly",
        ],
    );

    let layout = read("native/layout_asserts.cpp");
    require_all(
        &layout,
        &[
            "static_assert(sizeof(NeoPopulationResidentMetricsHandleV1) == 88);",
            "static_assert(alignof(NeoPopulationResidentMetricsHandleV1) == 8);",
            "offsetof(NeoPopulationResidentMetricsHandleV1, event_id) == 8",
            "offsetof(NeoPopulationResidentMetricsHandleV1, total_device_bytes) == 64",
            "NeoPopulationEnqueueMetricsOnlyV1Fn",
            "decltype(&neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1)",
        ],
    );

    let stub = read("native/stub.cpp");
    let stub_enqueue = section(
        &stub,
        "neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(",
        "}",
    );
    require_all(
        stub_enqueue,
        &[
            "NeoPopulationResidentMetricsHandleV1*",
            "NEO_POPULATION_STATUS_UNSUPPORTED",
        ],
    );
}

#[test]
fn checked_plan_is_derived_from_actual_session_extents_and_exact_layout_bytes() {
    let rust = read("src/population.rs");
    let plan = section(&rust, "pub struct PopulationMetricsOnlyPlanV1 {", "}");
    require_all(
        plan,
        &[
            "scenario_count: u64",
            "month_capacity: u64",
            "metric_rows_bytes: u64",
            "monthly_pnls_bytes: u64",
            "month_start_equities_bytes: u64",
            "scenario_descriptor_bytes: u64",
            "total_device_bytes: u64",
            "outcome_bytes: u64",
            "accepted_trade_total_bytes: u64",
        ],
    );
    assert!(
        !plan.contains("pub "),
        "metrics-only plan fields must not be caller-mintable"
    );
    require_all(
        &rust,
        &[
            "const POPULATION_METRIC_ROW_BYTES_V1: u64 = 104;",
            "const POPULATION_SCENARIO_DEVICE_BYTES_V1: u64 = 56;",
            "const POPULATION_F64_BYTES_V1: u64 = 8;",
            "fn checked_from_session_extents_v1(",
            "self.scenario_count",
            "settings.month_capacity",
            ".checked_mul(",
            ".checked_add(",
            "metrics_only_default_month_plan_is_exactly_4000_bytes_per_scenario",
            "assert_eq!(plan.total_device_bytes(), 4_000);",
            "assert_eq!(plan.outcome_bytes(), 0);",
            "assert_eq!(plan.accepted_trade_total_bytes(), 0);",
        ],
    );
}

#[test]
fn rust_handle_is_must_use_opaque_lifetime_bound_and_has_no_host_boundary() {
    let rust = read("src/population.rs");
    require_all(
        &rust,
        &[
            "#[must_use = \"resident GPU metrics must be consumed by the next device stage\"]",
            "pub struct ResidentPopulationMetricsV1<'session>",
            "session: &'session mut PopulationSession",
            "receipt: Box<RawResidentPopulationMetricsHandleV1>",
            "pub fn enqueue_metrics_only_v1(",
            "Result<ResidentPopulationMetricsV1<'_>, CudaPopulationError>",
            "PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(",
            "neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(",
        ],
    );
    let handle = section(
        &rust,
        "pub struct ResidentPopulationMetricsV1<'session> {",
        "}",
    );
    assert!(
        !handle.contains("pub "),
        "resident metric/event handle exposes caller-constructible fields"
    );
    let handle_impl = braced_item(
        &rust,
        "impl<'session> ResidentPopulationMetricsV1<'session> {",
    );
    for forbidden in [
        "event_id(",
        "raw_pointer",
        "as_device_ptr",
        "wait(",
        "read_metrics(",
        "read_diagnostics(",
        "synchronize(",
    ] {
        assert!(
            !handle_impl.contains(forbidden),
            "strict resident handle exposes host/raw boundary through {forbidden:?}"
        );
    }
    for forbidden in [
        "Clone for ResidentPopulationMetricsV1",
        "Copy for ResidentPopulationMetricsV1",
        "Serialize for ResidentPopulationMetricsV1",
        "Deserialize for ResidentPopulationMetricsV1",
        "Default for ResidentPopulationMetricsV1",
    ] {
        assert!(
            !rust.contains(forbidden),
            "opaque resident handle gains detachable authority through {forbidden:?}"
        );
    }
}

#[test]
fn strict_workspace_allocates_only_two_month_arrays_and_metric_rows() {
    let cuda = read("native/prototype_b_population.cu");
    let workspace = section(
        &cuda,
        "std::int32_t ensure_metrics_only_workspace_v1(",
        "std::int32_t enqueue_population_evaluation_v1(",
    );
    require_all(
        workspace,
        &[
            "device_alloc(&session->monthly_pnls",
            "device_alloc(&session->month_start_equities",
            "device_alloc(&session->metric_rows",
            "PopulationWorkspaceModeV1::StrictMetricsOnly",
            "workspace_scenarios",
            "month_capacity",
        ],
    );
    for forbidden in [
        "MAX_TRADES_PER_CANDIDATE",
        "kMaxTradesPerCandidate",
        "device_alloc(&session->outcomes",
        "device_alloc(&session->accepted_trade_total",
        "population_seed_outcomes_kernel",
    ] {
        assert!(
            !workspace.contains(forbidden),
            "strict metrics workspace retains diagnostic allocation through {forbidden:?}"
        );
    }
}

#[test]
fn strict_workspace_mode_stays_immutable_while_exact_extent_may_be_rebuilt() {
    let header = read("native/neoethos_gpu_cuda.h");
    require_all(
        &header,
        &[
            "#define NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH (-43)",
            "#define NEO_POPULATION_STATUS_WORKSPACE_PLAN_MISMATCH (-44)",
        ],
    );
    let cuda = read("native/prototype_b_population.cu");
    let session = section(&cuda, "struct NeoCudaPopulationSession {", "namespace {");
    require_all(
        session,
        &[
            "PopulationWorkspaceModeV1 workspace_mode = PopulationWorkspaceModeV1::Uninitialized;",
            "workspace_scenarios = 0;",
            "month_capacity = 0;",
        ],
    );

    let strict_workspace = section(
        &cuda,
        "std::int32_t ensure_metrics_only_workspace_v1(",
        "std::int32_t enqueue_population_evaluation_v1(",
    );
    require_all(
        strict_workspace,
        &[
            "session->workspace_mode == PopulationWorkspaceModeV1::CompatibilityDeviceParityOnly",
            "return NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH;",
            "session->workspace_mode = PopulationWorkspaceModeV1::StrictMetricsOnly;",
            "session->workspace_scenarios == scenario_count",
            "session->month_capacity == month_capacity",
            "session->release_workspace();",
            "session->outcomes != nullptr",
            "session->accepted_trade_total != nullptr",
        ],
    );

    let implementation = section(
        &cuda,
        "std::int32_t enqueue_population_evaluation_v1(",
        "neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(",
    );
    require_all(
        implementation,
        &[
            "session->workspace_mode == PopulationWorkspaceModeV1::StrictMetricsOnly",
            "NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH",
            "metrics_only_byte_plan_v1(session->workspace_scenarios, session->month_capacity",
            "resident_plan.scenario_descriptor_bytes != session->scenario_upload_bytes",
            "resident_metrics->metric_rows_bytes = resident_plan.metric_rows_bytes",
            "resident_metrics->total_device_bytes = resident_plan.total_device_bytes",
        ],
    );
    assert!(
        !implementation.contains("release_workspace();\n    session->workspace_mode ="),
        "one session can free and relabel an already-selected workspace authority"
    );

    let rust = read("src/population.rs");
    require_all(
        &rust,
        &[
            "validate_exact_resident_receipt_v1(",
            "STATUS_WORKSPACE_MODE_MISMATCH",
            "receipt.metric_rows_bytes == plan.metric_rows_bytes()",
            "receipt.total_device_bytes == plan.total_device_bytes()",
        ],
    );
}

#[test]
fn strict_enqueue_records_same_stream_event_with_null_diagnostics_and_zero_d2h() {
    let cuda = read("native/prototype_b_population.cu");
    let enqueue = braced_item(
        &cuda,
        "neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(",
    );
    require_all(
        enqueue,
        &[
            "PopulationEvaluationModeV1::StrictMetricsOnly",
            "enqueue_population_evaluation_v1(",
        ],
    );
    for forbidden in [
        "kMaxTradesPerCandidate",
        "MAX_TRADES_PER_CANDIDATE",
        "population_seed_outcomes_kernel",
        "session->outcomes",
        "session->accepted_trade_total",
        "cudaMemcpyDeviceToHost",
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
        "neoethos_gpu_cuda_population_wait",
        "neoethos_gpu_cuda_population_read_metrics",
        "neoethos_gpu_cuda_population_read_diagnostics",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "strict enqueue crosses into diagnostics/host state via {forbidden:?}"
        );
    }

    let implementation = section(
        &cuda,
        "std::int32_t enqueue_population_evaluation_v1(",
        "neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(",
    );
    require_all(
        implementation,
        &[
            "ensure_metrics_only_workspace_v1(",
            "session->stream",
            "cudaEventRecord(session->event, session->stream)",
            "resident_metrics->event_id",
            "resident_metrics->scenario_count",
            "resident_metrics->month_capacity",
            "resident_metrics->metric_rows_bytes",
            "resident_metrics->monthly_pnls_bytes",
            "resident_metrics->month_start_equities_bytes",
            "resident_metrics->scenario_descriptor_bytes",
            "resident_metrics->total_device_bytes",
            "resident_metrics->outcome_bytes = 0ull",
            "resident_metrics->accepted_trade_total_bytes = 0ull",
        ],
    );
    for forbidden in [
        "cudaMemcpyDeviceToHost",
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
        "neoethos_gpu_cuda_population_wait",
        "neoethos_gpu_cuda_population_read_metrics",
        "neoethos_gpu_cuda_population_read_diagnostics",
    ] {
        assert!(
            !implementation.contains(forbidden),
            "shared async enqueue crosses into host state via {forbidden:?}"
        );
    }
}

#[test]
fn shared_kernel_guards_every_optional_diagnostic_access() {
    let cuda = read("native/prototype_b_population.cu");
    let reduce = section(
        &cuda,
        "__global__ void population_reduce_kernel(",
        "// Session",
    );
    require_all(
        reduce,
        &[
            "const bool diagnostics_enabled = outcomes != nullptr;",
            "diagnostic_outcome_slot_v1(",
            "if (diagnostic_outcome != nullptr)",
            "if (accepted_trade_total != nullptr)",
        ],
    );
    assert!(
        !reduce.contains("outcomes[position_index]"),
        "reduce kernel writes outcome memory without the nullable slot guard"
    );
    let strict_launch = section(
        &cuda,
        "if (mode == PopulationEvaluationModeV1::StrictMetricsOnly) {",
        "} else {",
    );
    require_all(
        strict_launch,
        &[
            "ensure_metrics_only_workspace_v1(",
            "nullptr",
            "population_reduce_kernel",
        ],
    );
    for forbidden in [
        "population_seed_outcomes_kernel",
        "atomicAdd(",
        "cudaMemsetAsync(session->accepted_trade_total",
    ] {
        assert!(
            !strict_launch.contains(forbidden),
            "strict launch touches diagnostic state through {forbidden:?}"
        );
    }
}

#[test]
fn legacy_evaluate_wait_and_readback_remain_explicit_test_compatibility_only() {
    let header = read("native/neoethos_gpu_cuda.h");
    let rust = read("src/population.rs");
    require_all(
        &header,
        &[
            "Compatibility/DeviceParityOnly",
            "neoethos_gpu_cuda_population_b_evaluate(",
            "neoethos_gpu_cuda_population_wait(",
            "neoethos_gpu_cuda_population_read_metrics(",
            "neoethos_gpu_cuda_population_read_diagnostics(",
        ],
    );
    require_all(
        &rust,
        &[
            "Compatibility/DeviceParityOnly",
            "pub fn evaluate(",
            "pub fn wait(",
            "pub fn read_metrics(",
            "pub fn read_diagnostics(",
        ],
    );
}

#[test]
fn dropped_unconsumed_handle_poison_blocks_reuse_and_leaks_native_owner_fail_closed() {
    let rust = read("src/population.rs");
    require_all(
        &rust,
        &[
            "enum StrictResidentSessionStateV1",
            "StrictIdle",
            "InFlight",
            "Poisoned",
            "strict_resident_state: StrictResidentSessionStateV1",
            "strict_resident_state: StrictResidentSessionStateV1::StrictIdle",
            "consumed: bool",
            "impl Drop for ResidentPopulationMetricsV1<'_>",
            "if !self.consumed",
            "self.session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;",
            "fn require_strict_idle_v1(",
            "STATUS_STRICT_RESIDENT_IN_FLIGHT",
            "STATUS_STRICT_RESIDENT_POISONED",
        ],
    );

    for method in [
        "pub fn upload_dataset(",
        "pub fn upload_parent_dataset_v1(",
        "pub fn bind_evaluation_view_v1(",
        "pub fn read_residency_counters_v1(",
        "pub fn read_device_identity_v1(",
        "pub fn upload_genes(",
        "pub fn upload_scenarios(",
        "pub fn enqueue_metrics_only_v1(",
        "pub fn evaluate(",
        "pub fn wait(",
        "pub fn read_metrics(",
        "pub fn read_diagnostics_for(",
        "pub fn read_diagnostics(",
    ] {
        let body = braced_item(&rust, method);
        assert!(
            body.contains("self.require_strict_idle_v1("),
            "session path {method:?} can be reused after strict resident work"
        );
    }

    let session_drop = section(&rust, "impl Drop for PopulationSession {", "\n}");
    require_all(
        session_drop,
        &[
            "StrictResidentSessionStateV1::InFlight",
            "StrictResidentSessionStateV1::Poisoned",
            "self.handle = std::ptr::null_mut();",
            "return;",
            "neoethos_gpu_cuda_population_destroy(self.handle)",
        ],
    );
    assert!(
        session_drop
            .find("self.handle = std::ptr::null_mut();")
            .unwrap()
            < session_drop
                .find("neoethos_gpu_cuda_population_destroy(self.handle)")
                .unwrap(),
        "unconsumed strict work reaches native destroy before the leak-only guard"
    );

    let cuda = read("native/prototype_b_population.cu");
    require_all(
        &cuda,
        &[
            "enum class PopulationStrictExecutionStateV1",
            "PopulationStrictExecutionStateV1 strict_execution_state",
            "PopulationStrictExecutionStateV1::StrictIdle;",
            "strict_population_work_blocks_host_boundary_v1(",
            "session->strict_execution_state = PopulationStrictExecutionStateV1::InFlight;",
            "session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;",
            "NEO_POPULATION_STATUS_STRICT_RESIDENT_IN_FLIGHT",
        ],
    );
    let native_drop = braced_item(
        &cuda,
        "neoethos_gpu_cuda_population_destroy_terminal_checked_v2(",
    );
    let compatibility_drop = braced_item(&cuda, "neoethos_gpu_cuda_population_destroy(");
    require_all(
        native_drop,
        &[
            "strict_population_work_blocks_host_boundary_v1(session)",
            "session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;",
            "session->release_terminal_checked_v2()",
            "delete session;",
        ],
    );
    require_all(
        compatibility_drop,
        &["neoethos_gpu_cuda_population_destroy_terminal_checked_v2(session)"],
    );
    assert!(
        native_drop
            .find("strict_population_work_blocks_host_boundary_v1(session)")
            .unwrap()
            < native_drop.find("session->release_terminal_checked_v2()").unwrap(),
        "native destroy releases strict resident storage before the leak-only guard"
    );
}

#[test]
fn enqueue_state_is_recorded_before_receipt_validation_and_ambiguous_failures_poison() {
    let rust = read("src/population.rs");
    let enqueue = section(
        &rust,
        "pub fn enqueue_metrics_only_v1(",
        "/// Compatibility/DeviceParityOnly",
    );
    require_all(
        enqueue,
        &[
            "strict_enqueue_failure_is_known_prelaunch_v1(status)",
            "self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;",
            "self.strict_resident_state = StrictResidentSessionStateV1::InFlight;",
            "validate_exact_resident_receipt_v1(receipt.as_ref(), plan)",
            "consumed: false",
        ],
    );
    let call = enqueue
        .find("neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(")
        .unwrap();
    let in_flight = enqueue
        .find("self.strict_resident_state = StrictResidentSessionStateV1::InFlight;")
        .unwrap();
    let validate = enqueue
        .find("validate_exact_resident_receipt_v1(receipt.as_ref(), plan)")
        .unwrap();
    assert!(
        call < in_flight && in_flight < validate,
        "session state must become InFlight immediately after native success and before receipt validation"
    );
    let validation_tail = &enqueue[validate..];
    assert!(
        validation_tail
            .contains("self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;"),
        "receipt mismatch does not poison the already-launched native session"
    );

    let cuda = read("native/prototype_b_population.cu");
    let strict_launch = section(
        &cuda,
        "if (mode == PopulationEvaluationModeV1::StrictMetricsOnly) {",
        "} else {",
    );
    let mark = strict_launch
        .find("session->strict_execution_state = PopulationStrictExecutionStateV1::InFlight;")
        .unwrap();
    let launch = strict_launch
        .find("population_gap_flags_kernel<<<")
        .unwrap();
    assert!(
        mark < launch,
        "native strict state must become InFlight before the first kernel launch"
    );
}

#[test]
fn resident_search_uses_one_exact_full_population_extent_and_has_no_orphan_chunk_abi() {
    let rust = read("src/population.rs");
    let search = read("src/resident_search_v2.rs");
    let scoring = read("src/resident_scoring_v2.rs");
    let header = read("native/neoethos_gpu_cuda.h");
    let cuda = read("native/prototype_b_population.cu");

    let owned_enqueue = braced_item(&rust, "pub(crate) fn enqueue_resident_gene_metrics_owned_v2(");
    require_all(
        owned_enqueue,
        &[
            "retained_evaluation_capacity != logical_population_count",
            "self.scenario_count as u64 != logical_population_count",
            "self.population as u64 != logical_population_count",
            "one immutable full-population chunk",
            "PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(",
        ],
    );
    let scoring_seal = braced_item(&scoring, "pub(crate) fn seal_resident_scoring_plan_v2(");
    require_all(
        scoring_seal,
        &[
            "generation.retained_evaluation_capacity_v1()",
            "generation.logical_population_count_v1()",
            "generation/scoring semantics or full-population capacity differ",
        ],
    );
    require_all(
        &search,
        &[
            "enqueue_full_population_scored_generation_advance_v2(",
            "Native consumes one full-population device chunk",
        ],
    );

    let combined = format!("{rust}\n{header}\n{cuda}");
    for orphan in [
        "PopulationMetricsOnlyPlanV2",
        "RawResidentPopulationMetricsHandleV2",
        "NeoPopulationResidentMetricsHandleV2",
        "ensure_metrics_only_workspace_v2",
        "enqueue_population_evaluation_v2",
        "neoethos_gpu_cuda_population_b_enqueue_metrics_only_v2",
        "ResidentScenarioCapacityV1",
        "bind_resident_scenario_window_v1",
    ] {
        assert!(
            !combined.contains(orphan),
            "superseded, unimplemented chunk ABI `{orphan}` survived beside the full-population path"
        );
    }
}

#[test]
fn exact_metrics_workspace_reuses_equal_extent_and_rebuilds_changed_extent_without_padding() {
    let cuda = read("native/prototype_b_population.cu");
    let workspace = braced_item(&cuda, "std::int32_t ensure_metrics_only_workspace_v1(");
    require_all(
        workspace,
        &[
            "session->workspace_scenarios == scenario_count",
            "session->month_capacity == month_capacity",
            "session->release_workspace();",
            "device_alloc(&session->monthly_pnls, scenarios * months)",
            "device_alloc(&session->month_start_equities, scenarios * months)",
            "device_alloc(&session->metric_rows, scenarios)",
            "session->workspace_scenarios = scenario_count;",
        ],
    );
    let equal_extent = workspace
        .find("session->workspace_scenarios == scenario_count")
        .expect("exact retained extent check");
    let rebuild = workspace
        .find("session->release_workspace();")
        .expect("changed-extent rebuild");
    let allocation = workspace
        .find("device_alloc(&session->monthly_pnls, scenarios * months)")
        .expect("exact metrics allocation");
    assert!(equal_extent < rebuild && rebuild < allocation);
    for forbidden in [
        "pad_scenarios",
        "padded_scenario_count",
        "repeat_last_scenario",
        "clone_final_scenario",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
    ] {
        assert!(
            !workspace.contains(forbidden),
            "exact V1 workspace introduced `{forbidden}`"
        );
    }
}

#[test]
fn resident_scenarios_are_uploaded_once_before_each_full_population_generation() {
    let rust = read("src/population.rs");
    let search = read("src/resident_search_v2.rs");
    let upload = braced_item(&rust, "pub(crate) fn upload_resident_scenarios_v2(");
    require_all(
        upload,
        &[
            "self.genes_uploaded || self.scenarios_uploaded",
            "one fresh generation-owned session",
            "scenario.base_candidate_id",
            "neoethos_gpu_cuda_population_upload_resident_scenarios_v2(",
            "self.population = population;",
            "self.scenario_count = scenarios.len();",
        ],
    );
    let search_upload = braced_item(&search, "pub(crate) fn upload_resident_scenarios_v2(");
    require_all(
        search_upload,
        &[
            ".upload_resident_scenarios_v2(",
            "self.view.logical_population_count",
            "self.view.expected_generation_index",
            "self.view.plan_identity_sha256",
        ],
    );
    let advance = braced_item(
        &search,
        "pub(crate) fn advance_one_full_population_generation_v2(",
    );
    assert!(
        !advance.contains("upload_resident_scenarios_v2"),
        "generation advance reuploads the immutable full-population scenario set"
    );
}
