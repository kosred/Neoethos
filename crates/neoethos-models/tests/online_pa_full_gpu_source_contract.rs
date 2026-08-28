use std::fs;
use std::path::Path;

const ADAPTIVE_IMPL: &str = include_str!("../src/streaming/adaptive_impl.rs");
const REGISTRY: &str = include_str!("../src/registry.rs");
const LIFECYCLE: &str = include_str!("online_pa_full_gpu_lifecycle.rs");

fn repo_file(path: impl AsRef<Path>) -> String {
    let path = std::env::current_dir()
        .expect("current repository directory")
        .join(path);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn adaptive_gpu_facade() -> String {
    repo_file("crates/neoethos-models/src/streaming/adaptive_gpu.rs")
}

fn adaptive_gpu_module(name: &str) -> String {
    repo_file(format!(
        "crates/neoethos-models/src/streaming/adaptive_gpu/{name}.rs"
    ))
}

fn marked_body<'a>(source: &'a str, begin: &str, end: &str) -> &'a str {
    let start = source
        .find(begin)
        .unwrap_or_else(|| panic!("missing `{begin}`"));
    let remainder = &source[start + begin.len()..];
    let finish = remainder
        .find(end)
        .unwrap_or_else(|| panic!("missing `{end}`"));
    &remainder[..finish]
}

#[test]
fn adaptive_gpu_is_a_bounded_facade_over_owned_pipeline_modules() {
    let facade = adaptive_gpu_facade();
    assert!(
        facade.lines().count() <= 120,
        "adaptive_gpu.rs must be a bounded facade, got {} LOC",
        facade.lines().count()
    );
    for module in [
        "update",
        "preprocess",
        "inference",
        "device_utils",
        "fit",
        "predict",
    ] {
        assert!(
            facade.contains(&format!("mod {module};")),
            "facade does not own `{module}`"
        );
        let source = adaptive_gpu_module(module);
        assert!(
            source.lines().count() <= 520,
            "{module}.rs exceeds its 520 LOC ownership bound"
        );
    }
}

#[test]
fn tall_preprocessing_is_chunk_parallel_and_deterministically_reduced() {
    let preprocess = adaptive_gpu_module("preprocess");
    for required in [
        "LABEL_ROWS_PER_PARTIAL",
        "SCALER_ROWS_PER_PARTIAL",
        "TRANSFORM_ELEMENTS_PER_WORK_ITEM",
        "TRANSFORM_FAULTS_PER_PARTIAL",
        "online_pa_original_label_map_v2_kernel",
        "online_pa_label_count_partial_v2_kernel",
        "online_pa_label_count_weight_finalize_v2_kernel",
        "online_pa_ddof0_scaler_partial_v2_kernel",
        "online_pa_ddof0_scaler_finalize_v2_kernel",
        "online_pa_scaler_transform_chunked_v2_kernel",
        "online_pa_transform_fault_partial_v2_kernel",
        "online_pa_preprocess_fault_finalize_v2_kernel",
    ] {
        assert!(
            preprocess.contains(required),
            "missing parallel stage `{required}`"
        );
    }

    for forbidden in [
        "if ABSOLUTE_POS == 0 {\n        for row in 0..rows",
        "for pos in 0..feature_count",
        "online_pa_original_label_map_count_and_slack_weight_v1_kernel",
        "online_pa_ddof0_scaler_fit_v1_kernel",
        "online_pa_scaler_fault_reduce_v1_kernel",
    ] {
        assert!(
            !preprocess.contains(forbidden),
            "tall preprocessing retained single-lane work `{forbidden}`"
        );
    }
    assert!(
        preprocess.contains("partial_index * LABEL_ROWS_PER_PARTIAL")
            && preprocess.contains("partial_index * SCALER_ROWS_PER_PARTIAL"),
        "row chunks must have an explicit deterministic index order"
    );
}

#[test]
fn online_pa_cuda_fit_and_predict_route_the_complete_numerical_pipeline_to_gpu() {
    let update = adaptive_gpu_module("update");
    let inference = adaptive_gpu_module("inference");
    let fit_gpu = adaptive_gpu_module("fit");
    let predict_gpu = adaptive_gpu_module("predict");
    assert!(
        update.contains(
            "passive_aggressive_prediction_based_weighted_slack_v2_epoch_chunk_v3_kernel"
        )
    );
    assert!(inference.contains("online_pa_fused_raw_scale_logits_softmax_v1_kernel"));
    assert!(fit_gpu.contains("try_fit_passive_aggressive_cuda_full_pipeline"));
    assert!(predict_gpu.contains("try_predict_passive_aggressive_cuda_full_pipeline"));

    let fit = marked_body(
        ADAPTIVE_IMPL,
        "ONLINE_PA_FULL_GPU_FIT_BEGIN",
        "ONLINE_PA_FULL_GPU_FIT_END",
    );
    for forbidden in [
        "lease.scope",
        "FeatureScaler::fit",
        ".transform(",
        "remap_three_class_labels",
        "clamped_balanced_class_slack_weights_v1",
        "try_fit_passive_aggressive_cuda(",
    ] {
        assert!(
            !fit.contains(forbidden),
            "GPU fit still performs host numerical work `{forbidden}`"
        );
    }
    assert!(fit.contains("try_fit_passive_aggressive_cuda_full_pipeline"));

    let predict = marked_body(
        ADAPTIVE_IMPL,
        "ONLINE_PA_FULL_GPU_PREDICT_BEGIN",
        "ONLINE_PA_FULL_GPU_PREDICT_END",
    );
    for forbidden in ["lease.scope", ".transform(", "softmax_rows", ".dot("] {
        assert!(
            !predict.contains(forbidden),
            "GPU inference still performs host numerical work `{forbidden}`"
        );
    }
    assert!(predict.contains("try_predict_passive_aggressive_cuda_full_pipeline"));
}

#[test]
fn proven_cuda_runtime_metadata_never_reports_cpu_fallback_degradation() {
    let runtime = marked_body(
        ADAPTIVE_IMPL,
        "fn runtime_details(&self)",
        "pub fn predict_runtime(",
    );
    let cuda_receipt = runtime
        .find("if let Some(full_pipeline)")
        .expect("full CUDA receipt runtime branch");
    let cpu_branch = runtime
        .find("if self.effective_device_policy != \"cpu\"")
        .expect("explicit CPU/non-CPU split");
    assert!(
        cuda_receipt < cpu_branch,
        "the proven CUDA branch must be resolved before CPU fallback metadata"
    );
    assert!(
        runtime[cuda_receipt..cpu_branch]
            .contains("Some(full_pipeline.bound_inference_backend.clone())")
            && runtime[cuda_receipt..cpu_branch].contains(", None)"),
        "a proven full-CUDA artifact must report degraded_reason=None"
    );
    assert!(
        !runtime[..cpu_branch].contains("gpu_policy_cpu_fallback_reason"),
        "CPU fallback metadata must not be computed for a CUDA artifact"
    );
    assert!(
        runtime[cpu_branch..].contains("gpu_policy_cpu_fallback_reason(\"online_pa\")"),
        "the explicit CPU artifact branch must retain fallback self-reporting"
    );
}

#[test]
fn adaptive_gpu_facade_has_no_non_test_legacy_or_inferred_type_reexports() {
    let facade = adaptive_gpu_facade();
    for stale in [
        "PassiveAggressiveCudaFullFitV1,",
        "PassiveAggressiveCudaPredictionV1,",
        "PassiveAggressiveCudaFitV2,",
    ] {
        assert!(
            !facade.contains(stale),
            "adaptive GPU facade retained unused re-export `{stale}`"
        );
    }
    assert!(
        facade.contains("#[cfg(test)]\npub(crate) use update::try_fit_passive_aggressive_cuda;"),
        "legacy PA entry point must compile only for its unit-test consumers"
    );
}

#[test]
fn scaler_std_floor_uses_the_cubecl_native_expand_conversion() {
    let preprocess = adaptive_gpu_module("preprocess");
    assert!(
        preprocess.contains("1.0.into()"),
        "CubeCL if/else branches must return the same NativeExpand<f64> type"
    );
}

#[test]
fn superseded_host_update_wrapper_and_cpu_weight_helper_are_test_only() {
    let update = adaptive_gpu_module("update");
    for required in [
        "#[cfg(test)]\nuse anyhow::{Context, Result, bail};",
        "#[cfg(test)]\n#[derive(Debug)]\npub(crate) struct PassiveAggressiveCudaFitV2",
        "#[cfg(test)]\nfn validate_inputs(",
        "#[cfg(test)]\n#[allow(clippy::too_many_arguments)]\npub(crate) fn try_fit_passive_aggressive_cuda(",
    ] {
        assert!(
            update.contains(required),
            "legacy CUDA host wrapper is not isolated behind tests: `{required}`"
        );
    }
    assert!(
        ADAPTIVE_IMPL
            .contains("#[cfg(test)]\npub(super) fn clamped_balanced_class_slack_weights_v1"),
        "CPU class-weight helper must not compile into the production GPU lane"
    );
}

#[test]
fn superseded_host_update_class_count_import_is_test_only() {
    let update = adaptive_gpu_module("update");
    assert!(
        update.contains("#[cfg(test)]\nuse super::CLASS_COUNT;"),
        "the legacy host wrapper's CLASS_COUNT import must be isolated behind tests"
    );
    assert!(
        update.contains(
            "use super::{DEVICE_ARITHMETIC_REDUCTION_FAULT, DEVICE_ARITHMETIC_UPDATE_FAULT, PA_CUBE_UNITS};"
        ),
        "the production kernel import group must exclude the host-only CLASS_COUNT"
    );
}

#[test]
fn lifecycle_uses_the_public_runtime_prediction_module_and_stable_str_matching() {
    assert!(
        LIFECYCLE.contains("use neoethos_models::runtime::prediction::RuntimePrediction;"),
        "lifecycle must import RuntimePrediction from its public prediction module"
    );
    assert!(
        LIFECYCLE.contains(
            "fn assert_runtime_is_proven_cuda_without_degradation(runtime: &[RuntimePrediction])"
        ),
        "lifecycle must use the imported RuntimePrediction type"
    );
    assert!(
        LIFECYCLE.contains("for _ in 0..5 {\n            match role {"),
        "the borrowed benchmark role must use stable direct str matching"
    );
}

#[test]
fn fused_inference_checks_shifted_logits_before_any_exponentiation() {
    let inference = adaptive_gpu_module("inference");
    for required in [
        "let shifted_0 = logit_0 - maximum.read();",
        "let shifted_1 = logit_1 - maximum.read();",
        "let shifted_2 = logit_2 - maximum.read();",
        "let shifted_logits_finite",
        "if shifted_logits_finite && fault.read() == 0",
        "DEVICE_INFERENCE_ARITHMETIC_FAULT",
    ] {
        assert!(
            inference.contains(required),
            "fused inference is missing `{required}`"
        );
    }
    let shifted_check = inference
        .find("let shifted_logits_finite")
        .expect("shifted-logit finite check");
    let guarded_exp = inference
        .find("if shifted_logits_finite && fault.read() == 0")
        .expect("guard around exponentiation");
    let first_exp = inference.find(".exp()").expect("softmax exponentiation");
    assert!(
        shifted_check < guarded_exp && guarded_exp < first_exp,
        "shifted subtraction must be proven finite before exp executes"
    );
}

#[test]
fn pb_v2_training_is_epoch_major_bounded_and_device_resident_between_chunks() {
    let update = adaptive_gpu_module("update");
    let fit = adaptive_gpu_module("fit");
    for required in [
        "PA_TRAINING_ROWS_PER_LAUNCH",
        "passive_aggressive_prediction_based_weighted_slack_v2_epoch_chunk_v3_kernel",
        "row_start: u32",
        "row_count: u32",
        "for row_offset in 0..row_count as usize",
    ] {
        assert!(
            update.contains(required) || fit.contains(required),
            "bounded update schedule is missing `{required}`"
        );
    }
    assert!(
        !update.contains("for _ in 0..epochs as usize"),
        "the CUDA kernel must not monopolize one block for every epoch"
    );
    for required in [
        "for _epoch in 0..epochs",
        "for chunk_start in (0..rows).step_by(PA_TRAINING_ROWS_PER_LAUNCH)",
        "training_row_chunk_count_per_epoch",
        "training_launch_count",
        "training_interchunk_device_to_host_bytes: 0",
        "neoethos.online_pa.cuda_evidence.whole_fit_call.v3",
        "neoethos.online_pa.cuda_pipeline_stages.v3",
    ] {
        assert!(
            fit.contains(required),
            "full fit is missing dynamic launch evidence `{required}`"
        );
    }
    let launch_loop = fit
        .find("for _epoch in 0..epochs")
        .expect("epoch-major launch loop");
    let first_readback = fit
        .find("let arithmetic_status = read_arithmetic_status")
        .expect("post-training status readback");
    assert!(
        launch_loop < first_readback,
        "all update launches must complete before the first training D2H"
    );
    assert!(
        !fit[launch_loop..first_readback].contains("read_"),
        "interchunk D2H/readback is forbidden"
    );
}

#[test]
fn whole_call_receipts_bind_bytes_device_cost_policy_and_residency_scope() {
    let fit = adaptive_gpu_module("fit");
    let predict = adaptive_gpu_module("predict");
    let device_utils = adaptive_gpu_module("device_utils");
    for required in [
        "neoethos.online_pa.cuda_evidence.whole_fit_call.v3",
        "host_to_device_bytes: raw_feature_h2d_bytes + original_label_h2d_bytes",
        "device_to_host_bytes: artifact_d2h_bytes",
        "requested_device_policy",
        "effective_device_policy",
        "device_identity",
        "kernel_launch_count: whole_fit_kernel_launch_count",
        "training_rows_per_launch",
        "training_row_chunk_count_per_epoch",
        "training_epoch_count",
        "training_interchunk_device_to_host_bytes: 0",
        "rho(y,r)=1",
        "C*w_y",
        "residency_scope: \"call_scoped\".to_string()",
        "persistent_model_buffers: false",
        "scaled_feature_h2d_bytes: 0",
        "remapped_label_h2d_bytes: 0",
        "class_slack_weight_h2d_bytes: 0",
        "parameter_initialization_h2d_bytes: 0",
    ] {
        assert!(
            fit.contains(required),
            "fit receipt is missing `{required}`"
        );
    }
    for required in [
        "neoethos.online_pa.cuda_evidence.whole_predict_call.v2",
        "host_to_device_bytes:",
        "device_to_host_bytes:",
        "residency_scope: \"call_scoped\".to_string()",
        "persistent_model_buffers: false",
    ] {
        assert!(
            predict.contains(required),
            "predict receipt is missing `{required}`"
        );
    }
    for required in [
        "uuid: [u8; 16]",
        "pci_domain",
        "pci_bus",
        "pci_device",
        "compute_capability_major",
        "multiprocessor_count",
    ] {
        assert!(
            ADAPTIVE_IMPL.contains(required),
            "persisted exact device identity is missing `{required}`"
        );
    }
    assert!(device_utils.contains("validate_passive_aggressive_cuda_device_identity"));
}

#[test]
fn hardware_gates_cover_chunk_boundaries_extreme_logits_and_seven_cpu_workers() {
    let gpu_tests = repo_file("crates/neoethos-models/src/streaming/adaptive_pa_full_gpu_tests.rs");
    let lifecycle = repo_file("crates/neoethos-models/tests/online_pa_full_gpu_lifecycle.rs");
    for required in [
        "for rows in [1_023_usize, 1_024, 1_025]",
        "training_row_chunk_count_per_epoch",
        "training_interchunk_device_to_host_bytes",
        "DEVICE_INFERENCE_ARITHMETIC_FAULT",
        "f64::MAX, -f64::MAX, 0.0",
    ] {
        assert!(
            gpu_tests.contains(required),
            "GPU RED coverage is missing `{required}`"
        );
    }
    for required in [
        "CPU_WORKER_WIDTH: usize = 7",
        "WorkerLimit::new(CPU_WORKER_WIDTH)",
        "num_threads(CPU_WORKER_WIDTH)",
        "pb_v2_cpu_oracle",
        "legacy_cpu",
        "worker_width",
        "ONLINE_PA_SEARCH_MAX_EPOCHS: usize = 12",
        "MAX_EPOCH_WALL_TIMEOUT",
        ".kill()",
        "online_pa_rtx_child",
        "online_pa_rtx_validation_serial_orchestrator",
        "lifecycle-auto",
        "lifecycle-gpu-0",
        "parity-64",
        "parity-128",
    ] {
        assert!(
            lifecycle.contains(required),
            "public lifecycle/benchmark RED coverage is missing `{required}`"
        );
    }
}

#[test]
fn benchmark_timers_end_after_equal_probability_materialization_and_before_validation() {
    let lifecycle = repo_file("crates/neoethos-models/tests/online_pa_full_gpu_lifecycle.rs");
    for required in [
        "struct TimedProbabilitySample",
        "probabilities: Array2<f64>",
        "elapsed: Duration",
        "ONLINE_PA_CPU7_TIMED_WORKLOAD_BEGIN",
        "ONLINE_PA_CPU7_TIMED_WORKLOAD_END",
        "ONLINE_PA_EXPERT_TIMED_WORKLOAD_BEGIN",
        "ONLINE_PA_EXPERT_TIMED_WORKLOAD_END",
        "weighted_probability_checksum(&sample.probabilities)",
    ] {
        assert!(
            lifecycle.contains(required),
            "symmetric benchmark timing is missing `{required}`"
        );
    }

    for (begin, end) in [
        (
            "ONLINE_PA_CPU7_TIMED_WORKLOAD_BEGIN",
            "ONLINE_PA_CPU7_TIMED_WORKLOAD_END",
        ),
        (
            "ONLINE_PA_EXPERT_TIMED_WORKLOAD_BEGIN",
            "ONLINE_PA_EXPERT_TIMED_WORKLOAD_END",
        ),
    ] {
        let timed = marked_body(&lifecycle, begin, end);
        assert!(
            timed.contains("probabilities"),
            "{begin} must materialize probabilities before the timer stops"
        );
        for forbidden in [
            "weighted_probability_checksum",
            "serde_json",
            "std::fs::write",
            "receipt",
        ] {
            assert!(
                !timed.contains(forbidden),
                "{begin} includes post-workload validation `{forbidden}`"
            );
        }
    }
}

#[test]
fn rtx_gates_have_one_serial_parent_and_one_fresh_process_child_entrypoint() {
    let lifecycle = repo_file("crates/neoethos-models/tests/online_pa_full_gpu_lifecycle.rs");
    let gpu_tests = repo_file("crates/neoethos-models/src/streaming/adaptive_pa_full_gpu_tests.rs");
    assert_eq!(
        lifecycle.matches("#[ignore =").count(),
        1,
        "the integration binary must expose exactly one ignored RTX parent"
    );
    assert!(
        !gpu_tests.contains("#[ignore ="),
        "million-row parity must move behind the serial subprocess parent"
    );
    for required in [
        "RTX_CHILD_ROLE_ENV",
        "RTX_CHILD_RECEIPT_ENV",
        "RTX_CHILD_SEQUENCE_ENV",
        "struct SerialRtxRunner",
        "active_child",
        "no_concurrent_gpu_children",
        "fn online_pa_rtx_child()",
        "fn online_pa_rtx_validation_serial_orchestrator()",
        ".arg(\"--exact\")",
        ".arg(\"online_pa_rtx_child\")",
        ".arg(\"--test-threads=1\")",
        "\"status\": \"ok\"",
        "\"sequence\"",
        "\"device_uuid\"",
        "\"effective_device_policy\"",
        "lifecycle-auto",
        "lifecycle-gpu-0",
        "parity-64",
        "parity-128",
        "benchmark-pb-v2-cpu7",
        "benchmark-gpu",
        "benchmark-legacy-cpu",
        "max-epoch-64",
        "max-epoch-128",
    ] {
        assert!(
            lifecycle.contains(required),
            "serial RTX subprocess coverage is missing `{required}`"
        );
    }
    for obsolete_parent in [
        "fn expert_model_auto_full_cuda_lifecycle_is_exact_and_fail_closed()",
        "fn expert_model_gpu_0_full_cuda_lifecycle_is_exact_and_fail_closed()",
        "fn full_gpu_pipeline_is_1_25x_faster_than_identical_pb_v2_seven_cpu_oracle()",
        "fn million_row_64_and_128_width_search_max_epochs_finish_before_timeout()",
    ] {
        assert!(
            !lifecycle.contains(obsolete_parent),
            "independently parallelizable RTX parent remains `{obsolete_parent}`"
        );
    }
}

#[test]
fn production_registry_remains_deferred_until_the_full_hardware_gate_is_green() {
    let marker = "pub const CUDA_CAPABLE_MODEL_NAMES: &[&str] = &[";
    let start = REGISTRY.find(marker).expect("CUDA capability census");
    let remainder = &REGISTRY[start + marker.len()..];
    let end = remainder.find("];").expect("closed CUDA capability census");
    assert!(
        !remainder[..end].contains("\"online_pa\""),
        "online_pa registry routing must be a separate post-GREEN patch"
    );
}

#[test]
fn unavoidable_host_work_is_limited_to_io_packing_control_and_receipt_readback() {
    let fit = adaptive_gpu_module("fit");
    let predict = adaptive_gpu_module("predict");
    assert!(ADAPTIVE_IMPL.contains("feature_matrix_from_frame(x)?"));
    assert!(fit.contains("raw_features.iter().copied().collect::<Vec<_>>()"));
    assert!(fit.contains("client.create_from_slice(f64::as_bytes(&raw_features_flat))"));
    assert!(fit.contains("client.create_from_slice(i32::as_bytes(original_labels))"));
    assert!(fit.contains("client.empty("));
    assert!(predict.contains("client.create_from_slice"));
}
