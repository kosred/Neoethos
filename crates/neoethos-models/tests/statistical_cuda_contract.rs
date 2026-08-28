const CUDA_SOURCE: &str = include_str!("../src/statistical/linear_gpu.rs");
const LINEAR_SOURCE: &str = include_str!("../src/statistical/linear_impl.rs");
const MODELS_CARGO: &str = include_str!("../Cargo.toml");

#[cfg(feature = "statistical-gpu")]
use std::path::{Path, PathBuf};
#[cfg(feature = "statistical-gpu")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "statistical-gpu")]
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
#[cfg(feature = "statistical-gpu")]
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
#[cfg(feature = "statistical-gpu")]
use neoethos_models::base::ExpertModel;
#[cfg(feature = "statistical-gpu")]
use neoethos_models::statistical::common::install_statistical_runtime_from_settings;
#[cfg(feature = "statistical-gpu")]
use neoethos_models::{ElasticNetExpert, LogisticExpert};

#[test]
fn statistical_cuda_is_f64_and_connected_to_the_canonical_fail_loud_route() {
    for stale in [
        "Array<f32>",
        "Array1<f32>",
        "Array2<f32>",
        "Vec<f32>",
        "RuntimeCell::<f32>",
        "f32::as_bytes",
        "f32::from_bytes",
        "as f32",
    ] {
        assert!(
            !CUDA_SOURCE.contains(stale),
            "statistical CUDA still exposes the stale f32 surface `{stale}`"
        );
    }

    for required in [
        "Array<f64>",
        "Array1<f64>",
        "Array2<f64>",
        "RuntimeCell::<f64>",
        "f64::as_bytes",
        "f64::from_bytes",
        "soft_threshold_f64",
    ] {
        assert!(
            CUDA_SOURCE.contains(required),
            "statistical CUDA is missing required f64/proximal surface `{required}`"
        );
    }

    for required in [
        "use super::linear_gpu::{",
        "crate::common::cuda_kernel_enabled",
        "try_fit_linear_softmax_cuda",
        "try_predict_linear_softmax_cuda",
    ] {
        assert!(
            LINEAR_SOURCE.contains(required),
            "canonical statistical path is not connected to `{required}`"
        );
    }
    assert!(
        !LINEAR_SOURCE.contains("statistical_cuda_kernel_enabled"),
        "canonical statistical path restored the retired per-kernel CUDA selector"
    );

    for forbidden in [
        "statistical_cuda_fit_fallback_to_cpu",
        "statistical_cuda_predict_fallback_to_cpu",
        "falling back to cpu",
    ] {
        assert!(
            !LINEAR_SOURCE.contains(forbidden),
            "GpuOnly statistical execution retains forbidden fallback `{forbidden}`"
        );
    }
}

#[test]
fn statistical_cuda_keeps_best_parameters_resident_until_the_artifact_boundary() {
    for required in [
        "fn copy_best_parameters_kernel(",
        "let best_weights_handle = client.empty(",
        "let best_bias_handle = client.empty(",
        "copy_best_parameters_kernel::launch::<CudaRuntime>(",
        "read_f64_buffer(&client, best_weights_handle)?",
        "read_f64_buffer(&client, best_bias_handle)?",
    ] {
        assert!(
            CUDA_SOURCE.contains(required),
            "statistical CUDA best-parameter residency is missing `{required}`"
        );
    }

    let epoch_loop = CUDA_SOURCE
        .split("for _ in 0..epochs.max(1) {")
        .nth(1)
        .expect("statistical CUDA epoch loop")
        .split("let weights = if best_val_loss.is_finite()")
        .next()
        .expect("statistical CUDA final artifact boundary");
    for forbidden in [
        "read_f64_buffer(&client, weights_handle.clone())",
        "read_f64_buffer(&client, bias_handle.clone())",
    ] {
        assert!(
            !epoch_loop.contains(forbidden),
            "statistical CUDA synchronizes full parameters inside the epoch loop via `{forbidden}`"
        );
    }
}

#[test]
fn statistical_cuda_computes_row_errors_once_before_gradient_reduction() {
    for required in [
        "fn softmax_error_kernel(",
        "errors: &Array<f64>",
        "let errors_handle = client.empty(",
        "softmax_error_kernel::launch::<CudaRuntime>(",
    ] {
        assert!(
            CUDA_SOURCE.contains(required),
            "statistical CUDA row-error reuse is missing `{required}`"
        );
    }

    let gradient_body = CUDA_SOURCE
        .split("fn softmax_gradient_partials_kernel(")
        .nth(1)
        .expect("statistical CUDA gradient partial kernel")
        .split("#[cube(launch)]\nfn softmax_gradient_reduce_kernel(")
        .next()
        .expect("statistical CUDA gradient reduction boundary");
    assert!(
        !gradient_body.contains("class_probability("),
        "statistical CUDA gradient still recomputes every row softmax once per parameter"
    );

    let epoch_loop = CUDA_SOURCE
        .split("for _ in 0..epochs.max(1) {")
        .nth(1)
        .expect("statistical CUDA epoch loop");
    let error_launch = epoch_loop
        .find("softmax_error_kernel::launch::<CudaRuntime>(")
        .expect("row-error launch inside epoch loop");
    let gradient_launch = epoch_loop
        .find("softmax_gradient_partials_kernel::launch::<CudaRuntime>(")
        .expect("gradient partial launch inside epoch loop");
    assert!(
        error_launch < gradient_launch,
        "row errors must be materialized on-device before gradient reduction"
    );
}

#[test]
fn statistical_cuda_parallelizes_the_full_history_gradient_deterministically() {
    for required in [
        "const GRADIENT_ROWS_PER_PARTIAL:",
        "fn softmax_gradient_partials_kernel(",
        "fn softmax_gradient_reduce_kernel(",
        "let gradient_partial_count = rows.div_ceil(GRADIENT_ROWS_PER_PARTIAL)",
        "let gradient_partials_len = checked_element_count(",
        "let gradient_partials_handle = client.empty(",
        "softmax_gradient_partials_kernel::launch::<CudaRuntime>(",
        "softmax_gradient_reduce_kernel::launch::<CudaRuntime>(",
    ] {
        assert!(
            CUDA_SOURCE.contains(required),
            "statistical CUDA gradient parallelism is missing `{required}`"
        );
    }

    assert!(
        !CUDA_SOURCE.contains("fn softmax_gradient_kernel("),
        "the retired one-worker-per-parameter full-history gradient is still active"
    );

    let partial_body = CUDA_SOURCE
        .split("fn softmax_gradient_partials_kernel(")
        .nth(1)
        .expect("statistical CUDA gradient partial kernel")
        .split("#[cube(launch)]\nfn softmax_gradient_reduce_kernel(")
        .next()
        .expect("statistical CUDA gradient reduction boundary");
    assert!(partial_body.contains("let partial = ABSOLUTE_POS % partial_count_us;"));
    assert!(partial_body.contains("for row in start..end.read()"));

    let reduce_body = CUDA_SOURCE
        .split("fn softmax_gradient_reduce_kernel(")
        .nth(1)
        .expect("statistical CUDA gradient reduction kernel")
        .split("#[cube(launch)]\nfn softmax_apply_kernel(")
        .next()
        .expect("statistical CUDA apply boundary");
    assert!(reduce_body.contains("for partial in 0..partial_count as usize"));
}

#[test]
fn statistical_cuda_parallelizes_validation_and_reuses_prediction_softmax() {
    for required in [
        "fn softmax_loss_rows_kernel(",
        "fn partial_loss_sums_kernel(",
        "fn mean_loss_kernel(",
        "let validation_losses_handle = client.empty(",
        "let validation_partial_losses_handle = client.empty(",
        "softmax_loss_rows_kernel::launch::<CudaRuntime>(",
        "partial_loss_sums_kernel::launch::<CudaRuntime>(",
        "mean_loss_kernel::launch::<CudaRuntime>(",
    ] {
        assert!(
            CUDA_SOURCE.contains(required),
            "statistical CUDA validation parallelism is missing `{required}`"
        );
    }
    assert!(
        !CUDA_SOURCE.contains("fn softmax_loss_kernel("),
        "statistical CUDA retained the single-thread validation dot-product kernel"
    );

    let mean_loss_body = CUDA_SOURCE
        .split("fn mean_loss_kernel(")
        .nth(1)
        .expect("statistical CUDA final loss reduction")
        .split("#[cube(launch)]\nfn softmax_predict_kernel(")
        .next()
        .expect("statistical CUDA prediction boundary");
    assert!(
        !mean_loss_body.contains("for row in 0..rows"),
        "statistical CUDA final loss reduction still serially loops over every validation row"
    );
    assert!(mean_loss_body.contains("for partial in 0..partial_count"));

    let prediction_body = CUDA_SOURCE
        .split("fn softmax_predict_kernel(")
        .nth(1)
        .expect("statistical CUDA prediction kernel")
        .split("fn cuda_device_id(")
        .next()
        .expect("statistical CUDA prediction kernel boundary");
    assert!(
        !prediction_body.contains("class_probability("),
        "statistical CUDA prediction recomputes the complete row softmax once per class"
    );
    for required in ["let logit0", "let e0", "probabilities_out[base]"] {
        assert!(
            prediction_body.contains(required),
            "statistical CUDA prediction is missing shared row work `{required}`"
        );
    }
}

#[test]
fn statistical_cuda_preserves_the_cpu_early_stopping_threshold() {
    assert!(
        CUDA_SOURCE.contains("if loss + 1e-6 < best_val_loss"),
        "statistical CUDA must preserve the canonical CPU early-stopping improvement threshold"
    );
    assert!(
        !CUDA_SOURCE.contains("if loss + 1e-12 < best_val_loss"),
        "statistical CUDA introduced a device-only early-stopping semantic"
    );
}

#[test]
fn statistical_cuda_fails_closed_on_invalid_math_and_dimension_overflow() {
    for required in [
        "fn checked_u32_dimension(",
        "fn checked_buffer_bytes(",
        "u32::try_from(value)",
        ".checked_mul(std::mem::size_of::<f64>())",
        "statistical CUDA validation loss is non-finite",
        "statistical CUDA produced non-finite parameters",
        "statistical CUDA prediction produced invalid probabilities",
        "drop(features_flat);",
        "drop(labels_flat);",
    ] {
        assert!(
            CUDA_SOURCE.contains(required),
            "statistical CUDA fail-closed/memory boundary is missing `{required}`"
        );
    }
    assert!(
        !CUDA_SOURCE.contains(".saturating_mul("),
        "statistical CUDA must refuse overflow instead of saturating allocation or launch sizes"
    );
}

#[test]
fn statistical_cuda_preflights_exact_selected_device_memory_before_uploads() {
    for required in [
        "struct LinearCudaDeviceMemoryPlanV1",
        "fn planned_fit_device_bytes(",
        "fn allocator_reservation_upper_bound(",
        "fn preflight_device_memory(",
        "CudaContext::new(cuda_ordinal)",
        ".mem_get_info()",
        ".memory_usage()",
        "client.properties().memory.max_page_size",
        "max single buffer bytes",
        "allocator reservation upper bound",
        "planned device bytes",
        "available device bytes",
        "statistical cuda validation frame cannot be empty",
    ] {
        assert!(
            CUDA_SOURCE.contains(required),
            "statistical CUDA device-memory preflight is missing `{required}`"
        );
    }
    assert!(
        MODELS_CARGO.contains("cudarc = {")
            && MODELS_CARGO.contains("statistical-gpu = [")
            && MODELS_CARGO.contains("\"dep:cudarc\""),
        "statistical-gpu must directly declare the CUDA memory authority it calls"
    );
    assert!(
        MODELS_CARGO.contains("cubecl = { version = \"=0.10.0\"")
            && MODELS_CARGO.contains("cubecl-common = { version = \"=0.10.0\""),
        "the mirrored SubSlices allocator contract must fail closed on a CubeCL version change"
    );

    let fit_body = CUDA_SOURCE
        .split("pub(crate) fn try_fit_linear_softmax_cuda(")
        .nth(1)
        .expect("statistical CUDA fit body");
    let preflight = fit_body
        .find("preflight_device_memory(")
        .expect("statistical CUDA selected-device preflight");
    let first_upload = fit_body
        .find("client.create_from_slice(")
        .expect("statistical CUDA first device upload");
    assert!(
        preflight < first_upload,
        "statistical CUDA must refuse an oversized fit before its first device upload"
    );
    assert!(
        !CUDA_SOURCE.contains(".unwrap_or(total_device_bytes)"),
        "overflow while deriving available device memory must fail closed"
    );
    assert!(
        !CUDA_SOURCE.contains("free_device_bytes\n        .checked_add(reusable_reserved_bytes)"),
        "fragmented CubeCL reserve must not be treated as fungible free VRAM"
    );
    for stale in ["LinearCudaFitMemoryPlanV1", "preflight_fit_device_memory"] {
        assert!(
            !CUDA_SOURCE.contains(stale),
            "generic statistical CUDA memory admission still exposes fit-only `{stale}`"
        );
    }
}

#[test]
fn statistical_cuda_prediction_preflights_all_transient_buffers_before_uploads() {
    for required in [
        "fn planned_prediction_device_bytes(",
        "prediction features",
        "prediction weights",
        "prediction bias",
        "prediction output",
    ] {
        assert!(
            CUDA_SOURCE.contains(required),
            "statistical CUDA prediction memory plan is missing `{required}`"
        );
    }

    let prediction_body = CUDA_SOURCE
        .split("pub(crate) fn try_predict_linear_softmax_cuda(")
        .nth(1)
        .expect("statistical CUDA prediction body");
    for required in [
        "let cuda_ordinal = cuda_device_id(resolved_device_policy)?;",
        "let client = cubecl_cuda_client(cuda_ordinal);",
        "let memory_plan = planned_prediction_device_bytes(rows, cols)?;",
        "preflight_device_memory(&client, cuda_ordinal, &memory_plan)?;",
    ] {
        assert!(
            prediction_body.contains(required),
            "statistical CUDA prediction does not bind selected-device admission through `{required}`"
        );
    }

    let preflight = prediction_body
        .find("preflight_device_memory(&client, cuda_ordinal, &memory_plan)?;")
        .expect("statistical CUDA prediction selected-device preflight");
    let first_upload = prediction_body
        .find("client.create_from_slice(")
        .expect("statistical CUDA prediction first device upload");
    let output_allocation = prediction_body
        .find("client.empty(")
        .expect("statistical CUDA prediction output allocation");
    assert!(
        preflight < first_upload && preflight < output_allocation,
        "statistical CUDA must refuse an oversized prediction before its first device allocation"
    );
}

#[cfg(feature = "statistical-gpu")]
struct TempArtifactDir(PathBuf);

#[cfg(feature = "statistical-gpu")]
impl TempArtifactDir {
    fn create(name: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "neoethos-{name}-cuda-lifecycle-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create statistical CUDA artifact directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

#[cfg(feature = "statistical-gpu")]
impl Drop for TempArtifactDir {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "failed to remove statistical CUDA artifact directory {}: {error}",
                self.0.display()
            );
        }
    }
}

#[cfg(feature = "statistical-gpu")]
fn statistical_fixture(rows: usize, features: usize) -> (FeatureFrame, Vec<i32>) {
    let columns = (0..features)
        .map(|feature_index| {
            FeatureColumnF64::new(
                format!("feature_{feature_index}"),
                (0..rows)
                    .map(|row_index| {
                        let phase = row_index as f64 * (feature_index + 1) as f64 * 0.023;
                        phase.sin() + 0.15 * phase.cos()
                    })
                    .collect(),
                vec![FeatureCellValidity::Valid; rows],
            )
            .expect("build deterministic statistical CUDA feature")
        })
        .collect::<Vec<_>>();
    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        columns,
    )
    .expect("build statistical CUDA frame");
    let labels = (0..rows)
        .map(|row_index| match row_index % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();
    (frame, labels)
}

#[cfg(feature = "statistical-gpu")]
fn one_worker_lease() -> CpuLease {
    let width = WorkerLimit::new(1).expect("one statistical worker is valid");
    CpuPermitBroker::new(width)
        .acquire(CpuPermitRequest::local(width))
        .expect("statistical CUDA test can acquire one worker")
}

#[cfg(feature = "statistical-gpu")]
fn assert_probability_matrix(probabilities: &ndarray::Array2<f64>, rows: usize) {
    assert_eq!(probabilities.dim(), (rows, 3));
    for row in probabilities.outer_iter() {
        let sum = row.iter().sum::<f64>();
        assert!((sum - 1.0).abs() <= 1e-8, "probability sum is {sum}");
        assert!(row.iter().all(|value| value.is_finite()));
    }
}

#[cfg(feature = "statistical-gpu")]
fn exercise_statistical_cuda_lifecycle(
    name: &str,
    mut model: Box<dyn ExpertModel>,
    mut restored: Box<dyn ExpertModel>,
    frame: &FeatureFrame,
    labels: &[i32],
) {
    let lease = one_worker_lease();
    model
        .fit(frame, labels, &lease)
        .unwrap_or_else(|error| panic!("mandatory {name} CUDA training failed: {error:#}"));
    let trained = model
        .predict_proba(frame, &lease)
        .unwrap_or_else(|error| panic!("mandatory {name} CUDA inference failed: {error:#}"));
    assert_probability_matrix(&trained, frame.n_samples());

    let artifact_dir = TempArtifactDir::create(name);
    model
        .save(artifact_dir.path())
        .unwrap_or_else(|error| panic!("persist {name} CUDA artifact: {error:#}"));
    let runtime: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifact_dir.path().join("model.json"))
            .expect("read statistical CUDA artifact"),
    )
    .expect("parse statistical CUDA artifact");
    assert_eq!(runtime["model_name"], name);
    assert_eq!(runtime["requested_device_policy"], "gpu:0");
    assert_eq!(runtime["effective_device_policy"], "gpu:0");
    assert!(
        runtime["runtime_backend"]
            .as_str()
            .is_some_and(|backend| backend.contains("cuda[gpu:0]")),
        "{name} artifact did not record exact CUDA ordinal zero: {runtime}"
    );
    assert_eq!(runtime["runtime_backend_kind"], "native_cuda");
    assert_eq!(runtime["runtime_degraded_reason"], serde_json::Value::Null);

    restored
        .load(artifact_dir.path())
        .unwrap_or_else(|error| panic!("restore {name} CUDA artifact: {error:#}"));
    let after_load = restored
        .predict_proba(frame, &lease)
        .unwrap_or_else(|error| panic!("restored {name} CUDA inference failed: {error:#}"));
    assert_probability_matrix(&after_load, frame.n_samples());
    for (before, after) in trained.iter().zip(after_load.iter()) {
        assert!(
            (before - after).abs() <= 1e-8,
            "{name} save/load probability drift: trained={before}, restored={after}"
        );
    }
}

#[cfg(feature = "statistical-gpu")]
#[test]
fn logistic_and_elasticnet_cuda_surfaces_train_infer_save_load() {
    assert!(
        neoethos_models::tree_models::config::nvidia_gpu_count() > 0,
        "mandatory statistical CUDA lifecycle gate requires a visible NVIDIA device"
    );
    let mut settings = neoethos_core::Settings::default();
    settings.models.statistical_device = "gpu:0".to_string();
    install_statistical_runtime_from_settings(&settings);

    let (frame, labels) = statistical_fixture(256, 5);
    let mut logistic = LogisticExpert::new();
    logistic.epochs = 12;
    logistic.learning_rate = 0.05;
    exercise_statistical_cuda_lifecycle(
        "logistic",
        Box::new(logistic),
        Box::new(LogisticExpert::new()),
        &frame,
        &labels,
    );

    let mut elasticnet = ElasticNetExpert::new(0.05, 0.5);
    elasticnet.epochs = 12;
    elasticnet.learning_rate = 0.05;
    exercise_statistical_cuda_lifecycle(
        "elasticnet",
        Box::new(elasticnet),
        Box::new(ElasticNetExpert::new(0.05, 0.5)),
        &frame,
        &labels,
    );
}
