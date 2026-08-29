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
            "receipt: RawResidentPopulationMetricsHandleV1",
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
    let handle_impl = section(&rust, "impl ResidentPopulationMetricsV1<'_> {", "}");
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
fn strict_workspace_mode_is_immutable_and_receipt_charges_actual_residency() {
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
            "session->workspace_scenarios != scenario_count",
            "return NEO_POPULATION_STATUS_WORKSPACE_PLAN_MISMATCH;",
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
    let enqueue = section(
        &cuda,
        "neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(",
        "neoethos_gpu_cuda_population_b_evaluate(",
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
        let body = section(&rust, method, "\n    }");
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
    let (_, native_drop) = cuda
        .split_once("neoethos_gpu_cuda_population_destroy(")
        .expect("native destroy boundary");
    require_all(
        native_drop,
        &[
            "strict_population_work_blocks_host_boundary_v1(session)",
            "return;",
            "session->release();",
        ],
    );
    assert!(
        native_drop
            .find("strict_population_work_blocks_host_boundary_v1(session)")
            .unwrap()
            < native_drop.find("session->release();").unwrap(),
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
            "validate_exact_resident_receipt_v1(&receipt, plan)",
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
        .find("validate_exact_resident_receipt_v1(&receipt, plan)")
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
fn retained_capacity_and_active_extent_are_distinct_checked_authorities() {
    let rust = read("src/population.rs");
    require_all(
        &rust,
        &[
            "pub struct PopulationMetricsOnlyPlanV2 {",
            "retained_scenario_capacity: u64",
            "active_scenario_count: u64",
            "checked_from_full_workspace_plan_v2(",
            "full_plan.retained_scenario_capacity_v2()",
            "active_scenario_count == 0",
            "active_scenario_count > retained_scenario_capacity",
            "retained_scenario_capacity.checked_mul(POPULATION_METRIC_ROW_BYTES_V1)",
            "retained_scenario_capacity.checked_mul(month_capacity)",
            "pub const fn retained_scenario_capacity(self) -> u64",
            "pub const fn active_scenario_count(self) -> u64",
        ],
    );
    let plan = section(&rust, "pub struct PopulationMetricsOnlyPlanV2 {", "}");
    assert!(
        !plan.contains("pub "),
        "active/capacity extents must come from the opaque full workspace plan, not caller fields"
    );

    require_all(
        &rust,
        &[
            "struct RawResidentPopulationMetricsHandleV2 {",
            "active_scenario_count: u64",
            "retained_scenario_capacity: u64",
            "receipt.active_scenario_count == plan.active_scenario_count()",
            "receipt.retained_scenario_capacity == plan.retained_scenario_capacity()",
            "receipt.metric_rows_bytes == plan.retained_metric_rows_bytes()",
            "receipt.total_device_bytes == plan.retained_total_device_bytes()",
        ],
    );
    assert!(
        !rust.contains(
            "receipt.scenario_count == plan.scenario_count()\n        && receipt.scenario_count"
        ),
        "one receipt count must never stand for both logical work and retained allocation"
    );

    let header = read("native/neoethos_gpu_cuda.h");
    let receipt = section(
        &header,
        "struct NeoPopulationResidentMetricsHandleV2 {",
        "};",
    );
    require_all(
        receipt,
        &[
            "std::uint64_t active_scenario_count;",
            "std::uint64_t retained_scenario_capacity;",
            "std::uint64_t metric_rows_bytes;",
            "std::uint64_t total_device_bytes;",
        ],
    );
}

#[test]
fn smaller_chunks_reuse_retained_workspace_and_launch_only_the_active_extent() {
    let cuda = read("native/prototype_b_population.cu");
    let workspace = section(
        &cuda,
        "std::int32_t ensure_metrics_only_workspace_v2(",
        "std::int32_t enqueue_population_evaluation_v2(",
    );
    require_all(
        workspace,
        &[
            "retained_scenario_capacity",
            "active_scenario_count",
            "active_scenario_count <= 0",
            "active_scenario_count > retained_scenario_capacity",
            "session->retained_scenario_capacity != retained_scenario_capacity",
            "NEO_POPULATION_STATUS_WORKSPACE_PLAN_MISMATCH",
            "session->active_scenario_count = active_scenario_count;",
            "device_alloc(&session->monthly_pnls, retained_scenarios * months)",
            "device_alloc(&session->month_start_equities, retained_scenarios * months)",
            "device_alloc(&session->metric_rows, retained_scenarios)",
        ],
    );
    let reuse = section(
        workspace,
        "if (session->metric_rows != nullptr) {",
        "return NEO_POPULATION_STATUS_OK;",
    );
    for forbidden in [
        "release_workspace",
        "device_alloc",
        "cudaFree",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
    ] {
        assert!(
            !reuse.contains(forbidden),
            "a smaller active chunk mutates retained storage through {forbidden:?}"
        );
    }

    let enqueue = section(
        &cuda,
        "std::int32_t enqueue_population_evaluation_v2(",
        "neoethos_gpu_cuda_population_b_enqueue_metrics_only_v2(",
    );
    require_all(
        enqueue,
        &[
            "scenario_view.count = active_scenario_count;",
            "choose_reduce_block(active_scenario_count, session->sm_count)",
            "active_scenario_count + reduce_block - 1",
            "metrics_only_byte_plan_v2(session->retained_scenario_capacity",
            "resident_metrics->active_scenario_count =",
            "resident_metrics->retained_scenario_capacity =",
        ],
    );
    for forbidden in [
        "pad_scenarios",
        "padded_scenario_count",
        "repeat_last_scenario",
        "clone_final_scenario",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "final chunks must not fake retained capacity through {forbidden:?}"
        );
    }
}

#[test]
fn strict_chunk_rebinding_requires_resident_scenario_capacity_not_host_reupload() {
    let rust = read("src/population.rs");
    let upload = section(
        &rust,
        "pub fn upload_scenarios(",
        "pub fn enqueue_metrics_only_v1(",
    );
    require_all(upload, &["Compatibility/DeviceParityOnly"]);

    require_all(
        &rust,
        &[
            "pub struct ResidentScenarioCapacityV1<'run>",
            "pub fn bind_resident_scenario_window_v1(",
            "scenario_capacity: ResidentScenarioCapacityV1<'run>",
            "active_scenario_count: NonZeroUsize",
            "active_scenario_count.get() <= scenario_capacity.retained_capacity()",
            "pub fn enqueue_metrics_only_active_v2(",
            "self.resident_scenario_capacity.as_ref()",
        ],
    );
    let bind = section(
        &rust,
        "pub fn bind_resident_scenario_window_v1(",
        "pub fn enqueue_metrics_only_active_v2(",
    );
    for forbidden in [
        "neoethos_gpu_cuda_population_upload_scenarios",
        "cudaMemcpy",
        "to_vec(",
        "Vec<ScenarioDescriptor>",
        "synchronize",
        "wait(",
    ] {
        assert!(
            !bind.contains(forbidden),
            "strict chunk rebinding crosses the resident boundary through {forbidden:?}"
        );
    }

    let cuda = read("native/prototype_b_population.cu");
    let strict_bind = section(
        &cuda,
        "neoethos_gpu_cuda_population_bind_resident_scenario_window_v1(",
        "neoethos_gpu_cuda_population_b_enqueue_metrics_only_v2(",
    );
    require_all(
        strict_bind,
        &[
            "active_scenario_count > session->retained_scenario_capacity",
            "session->active_scenario_count = active_scenario_count;",
            "cudaStreamWaitEvent(session->stream",
        ],
    );
    for forbidden in [
        "release_scenarios",
        "device_alloc",
        "cudaFree",
        "cudaMemcpy",
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
    ] {
        assert!(
            !strict_bind.contains(forbidden),
            "resident scenario window is rebound with allocation/host transfer via {forbidden:?}"
        );
    }
}
