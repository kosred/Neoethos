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
            "missing terminal-result token {token:?}"
        );
    }
}

fn require_none(source: &str, forbidden: &[&str]) {
    for token in forbidden {
        assert!(
            !source.contains(token),
            "terminal-result boundary contains forbidden token {token:?}"
        );
    }
}

#[test]
fn terminal_result_abi_is_fixed_width_and_copies_exactly_one_metric_row() {
    let header = read("native/neoethos_gpu_cuda.h");
    let layout = read("native/layout_asserts.cpp");
    let result = section(
        &header,
        "struct NeoPopulationTerminalCompactResultV1 {",
        "};",
    );
    require_all(
        result,
        &[
            "std::uint32_t abi_version;",
            "std::uint32_t reserved;",
            "std::uint64_t event_id;",
            "std::uint64_t scenario_count;",
            "NeoPopulationMetricRow metric_row;",
            "std::uint64_t terminal_synchronization_count;",
            "std::uint64_t terminal_readback_count;",
            "std::uint64_t terminal_readback_rows;",
            "std::uint64_t terminal_readback_bytes;",
        ],
    );
    require_all(
        &header,
        &[
            "neoethos_gpu_cuda_population_consume_terminal_compact_result_v1(",
            "const NeoPopulationResidentMetricsHandleV1* resident_metrics",
            "NeoPopulationTerminalCompactResultV1* compact_result",
        ],
    );
    require_all(
        &layout,
        &[
            "sizeof(NeoPopulationTerminalCompactResultV1) == 160",
            "alignof(NeoPopulationTerminalCompactResultV1) == 8",
            "offsetof(NeoPopulationTerminalCompactResultV1, metric_row) == 24",
            "NeoPopulationConsumeTerminalCompactResultV1Fn",
        ],
    );
}

#[test]
fn native_terminal_consumer_is_the_only_strict_host_boundary_and_is_one_row_only() {
    let native = read("native/prototype_b_population.cu");
    let consume = section(
        &native,
        "neoethos_gpu_cuda_population_consume_terminal_compact_result_v1(",
        "\n}",
    );
    require_all(
        consume,
        &[
            "PopulationWorkspaceModeV1::StrictMetricsOnly",
            "PopulationStrictExecutionStateV1::InFlight",
            "resident_metrics->event_id != session->pending_event_id",
            "resident_metrics->scenario_count != 1ull",
            "resident_metrics->metric_rows_bytes != sizeof(NeoPopulationMetricRow)",
            "session->outcomes != nullptr",
            "session->accepted_trade_total != nullptr",
            "cudaEventSynchronize(session->event)",
            "cudaMemcpy(&compact_result->metric_row, session->metric_rows,",
            "sizeof(NeoPopulationMetricRow), cudaMemcpyDeviceToHost)",
            "terminal_synchronization_count = 1ull",
            "terminal_readback_count = 1ull",
            "terminal_readback_rows = 1ull",
            "terminal_readback_bytes = sizeof(NeoPopulationMetricRow)",
        ],
    );
    require_none(
        consume,
        &[
            "read_diagnostics",
            "neoethos_gpu_cuda_population_wait",
            "neoethos_gpu_cuda_population_read_metrics",
            "cudaStreamSynchronize",
        ],
    );
    assert_eq!(consume.matches("session->outcomes").count(), 1);
    assert_eq!(consume.matches("session->accepted_trade_total").count(), 1);
    assert_eq!(consume.matches("cudaEventSynchronize(").count(), 1);
    assert_eq!(consume.matches("cudaMemcpy(").count(), 1);
}

#[test]
fn terminal_transition_is_fail_closed_and_clears_inflight_only_after_success() {
    let native = read("native/prototype_b_population.cu");
    let consume = section(
        &native,
        "neoethos_gpu_cuda_population_consume_terminal_compact_result_v1(",
        "\n}",
    );
    require_all(
        consume,
        &[
            "PopulationStrictExecutionStateV1::Poisoned",
            "PopulationStrictExecutionStateV1::StrictIdle",
            "session->pending_event_id = 0ull",
            "session->metrics_ready = false",
        ],
    );
    let idle_at = consume
        .rfind("PopulationStrictExecutionStateV1::StrictIdle")
        .expect("successful strict-idle transition");
    let copy_at = consume
        .find("cudaMemcpy(&compact_result->metric_row")
        .expect("bounded terminal readback");
    assert!(
        copy_at < idle_at,
        "strict state may become reusable only after the exact row copy succeeds"
    );
}

#[test]
fn rust_consumer_is_move_only_and_returns_opaque_identity_bound_evidence() {
    let rust = read("src/population.rs");
    let receipt = section(
        &rust,
        "pub struct TerminalCompactPopulationResultReceiptV1 {",
        "\n}",
    );
    require_all(
        receipt,
        &[
            "metric_row: NeoPopulationMetricRow",
            "resident_session_identity_sha256: [u8; 32]",
            "view_identity_sha256: [u8; 32]",
            "gene_batch_identity_sha256: [u8; 32]",
            "scenario_batch_identity_sha256: [u8; 32]",
            "settings_identity_sha256: [u8; 32]",
            "native_build_identity_sha256: [u8; 32]",
            "event_id: u64",
            "receipt_identity_sha256: [u8; 32]",
        ],
    );
    assert!(
        !receipt.contains("pub "),
        "terminal receipt fields must not be caller-mintable"
    );
    let consume = section(
        &rust,
        "pub fn consume_terminal_compact_result_v1(",
        "\n    }",
    );
    require_all(
        consume,
        &[
            "self",
            "neoethos_gpu_cuda_population_consume_terminal_compact_result_v1",
            "validate_terminal_compact_result_v1",
            "StrictResidentSessionStateV1::StrictIdle",
            "self.consumed = true",
            "TerminalCompactPopulationResultReceiptV1",
        ],
    );
    require_none(
        &rust,
        &[
            "Clone for TerminalCompactPopulationResultReceiptV1",
            "Copy for TerminalCompactPopulationResultReceiptV1",
            "Default for TerminalCompactPopulationResultReceiptV1",
            "Deserialize for TerminalCompactPopulationResultReceiptV1",
        ],
    );
}

#[test]
fn exact_session_view_gene_scenario_and_settings_inputs_are_hashed_without_raw_bytes() {
    let rust = read("src/population.rs");
    for function in [
        "fn hash_resident_population_session_identity_v3(",
        "fn hash_population_view_identity_v1(",
        "fn hash_population_gene_batch_identity_v1(",
        "fn hash_population_scenario_batch_identity_v1(",
        "fn hash_population_settings_identity_v1(",
        "fn hash_terminal_compact_result_receipt_v1(",
    ] {
        assert!(
            rust.contains(function),
            "missing exact identity function {function}"
        );
    }
    require_all(
        &rust,
        &[
            "b\"neoethos.population.resident-session.v3\"",
            "b\"neoethos.population.view.v1\"",
            "b\"neoethos.population.gene-batch.v1\"",
            "b\"neoethos.population.scenario-batch.v1\"",
            "b\"neoethos.population.settings.v1\"",
            "b\"neoethos.population.terminal-compact-result.v1\"",
            "value.to_bits().to_le_bytes()",
        ],
    );
    require_none(
        &rust,
        &["sample_hash", "dataset_key", "from_raw_parts", "as_bytes(&"],
    );
}

#[test]
fn terminal_evidence_is_separate_from_all_intermediate_readback_counters() {
    let rust = read("src/population.rs");
    let receipt_impl = section(
        &rust,
        "impl TerminalCompactPopulationResultReceiptV1 {",
        "\n}",
    );
    require_all(
        receipt_impl,
        &[
            "terminal_synchronization_count",
            "terminal_readback_count",
            "terminal_readback_rows",
            "terminal_readback_bytes",
            "receipt_identity_sha256",
        ],
    );
    let native = read("native/prototype_b_population.cu");
    let consume = section(
        &native,
        "neoethos_gpu_cuda_population_consume_terminal_compact_result_v1(",
        "\n}",
    );
    require_none(
        consume,
        &[
            "metric_rows_readback_count",
            "diagnostic_readback_count",
            "accepted_trade_total_readback_count",
            "explicit_synchronization_count",
        ],
    );
}

#[test]
fn stub_refuses_success_and_layout_pins_the_exact_terminal_signature() {
    let stub = read("native/stub.cpp");
    let layout = read("native/layout_asserts.cpp");
    let stub_consume = section(
        &stub,
        "neoethos_gpu_cuda_population_consume_terminal_compact_result_v1(",
        "\n}",
    );
    require_all(
        stub_consume,
        &[
            "NeoPopulationResidentMetricsHandleV1",
            "NeoPopulationTerminalCompactResultV1",
            "NEO_POPULATION_STATUS_UNSUPPORTED",
        ],
    );
    require_all(
        &layout,
        &[
            "NeoPopulationConsumeTerminalCompactResultV1Fn",
            "decltype(&neoethos_gpu_cuda_population_consume_terminal_compact_result_v1)",
        ],
    );
}

#[test]
fn real_device_fixture_uses_one_admission_one_stream_and_releases_after_terminal_result() {
    let resident = read("src/resident_feature_store_v3.rs");
    let device = read("src/resident_population_session_v3_device_tests.rs");
    require_all(
        &resident,
        &[
            "#[cfg(all(test, feature = \"cuda\"))]",
            "mod resident_population_session_v3_device_tests;",
        ],
    );
    require_all(
        &device,
        &[
            "NEOETHOS_REQUIRE_GPU",
            "acquire_discovery_run_device_admission_v1",
            "prepare_resident_smc_parent_v3",
            "RESIDENT_SMC_COLUMN_NAMES_V3",
            "consume_into_population_session_v3",
            "PopulationEvaluationViewV1::full",
            "PopulationEvaluationViewV1::contiguous_range",
            "PopulationEvaluationViewV1::ordered_indices",
            "enqueue_metrics_only_v1",
            "consume_terminal_compact_result_v1",
            "record_consumer_completion",
            "completion_is_ready",
            "parent_upload_count(), 0",
            "stream_creation_count(), 0",
            "metric_rows_readback_count(), 0",
            "diagnostic_readback_count(), 0",
            "accepted_trade_total_readback_count(), 0",
        ],
    );
    require_none(
        &device,
        &[
            "PopulationSession::create",
            "upload_dataset",
            "upload_parent_dataset_v1",
            "to_dense_samples_major",
            "Context::new",
            "Stream::new",
            "fallback",
            "Cpu",
            "f32",
        ],
    );
}
