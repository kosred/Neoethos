use anyhow::{Context, Result, bail};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use ndarray::{Array1, Array2};

use super::super::adaptive_impl::{
    ONLINE_PA_CUDA_FULL_PIPELINE_SCHEMA_V3, ONLINE_PA_CUDA_PREPROCESSING_BACKEND_V2,
    ONLINE_PA_CUDA_TRAINING_SEMANTICS_V2, PassiveAggressiveCudaDeviceIdentityV1,
    PassiveAggressiveCudaEvidenceV1, PassiveAggressiveCudaPipelineEvidenceV1,
};
use super::device_utils::{
    bytes_for, checked_add, checked_ceil_div, checked_mul, checked_u32, checked_u64, cube_count_1d,
    exact_cuda_ordinal, fail_for_full_pipeline_status, preflight_device_memory,
    query_cuda_device_identity, read_arithmetic_status, read_f64_buffer, read_u32_buffer,
};
use super::preprocess::{
    LABEL_CHANNEL_COUNT, LABEL_ROWS_PER_PARTIAL, SCALER_ROWS_PER_PARTIAL,
    TRANSFORM_ELEMENTS_PER_WORK_ITEM, TRANSFORM_FAULTS_PER_PARTIAL,
    online_pa_ddof0_scaler_finalize_v2_kernel, online_pa_ddof0_scaler_partial_v2_kernel,
    online_pa_full_pipeline_initialize_v2_kernel, online_pa_label_count_partial_v2_kernel,
    online_pa_label_count_weight_finalize_v2_kernel, online_pa_original_label_map_v2_kernel,
    online_pa_preprocess_fault_finalize_v2_kernel, online_pa_scaler_transform_chunked_v2_kernel,
    online_pa_transform_fault_partial_v2_kernel,
};
use super::update::{
    PA_TRAINING_ROWS_PER_LAUNCH,
    passive_aggressive_prediction_based_weighted_slack_v2_epoch_chunk_v3_kernel,
};
use super::{CLASS_COUNT, PA_CUBE_UNITS};
use crate::cubecl_lifecycle::{cubecl_cuda_client, cubecl_residency_scope};

#[derive(Debug)]
pub(crate) struct PassiveAggressiveCudaFullFitV1 {
    pub scaler_means: Vec<f64>,
    pub scaler_stds: Vec<f64>,
    pub class_counts: [u32; CLASS_COUNT],
    pub class_slack_weights: [f64; CLASS_COUNT],
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub runtime_backend: String,
    pub effective_device_policy: String,
    pub training_semantics_schema: String,
    pub device_identity: PassiveAggressiveCudaDeviceIdentityV1,
    pub evidence: PassiveAggressiveCudaEvidenceV1,
}

/// Full call-scoped fit; host work is packing, transfers, control and receipts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fit_passive_aggressive_cuda_full_pipeline(
    requested_device_policy: &str,
    effective_device_policy: &str,
    raw_features: &Array2<f64>,
    original_labels: &[i32],
    aggressiveness: f64,
    epochs: usize,
) -> Result<PassiveAggressiveCudaFullFitV1> {
    if raw_features.nrows() == 0 || raw_features.ncols() == 0 {
        bail!("online_pa full CUDA pipeline requires a non-empty raw feature matrix");
    }
    if original_labels.len() != raw_features.nrows() {
        bail!(
            "online_pa full CUDA row/label mismatch: {} rows vs {} original labels",
            raw_features.nrows(),
            original_labels.len()
        );
    }
    if !aggressiveness.is_finite() || aggressiveness <= 0.0 {
        bail!("online_pa full CUDA aggressiveness must be finite and positive");
    }
    if epochs == 0 {
        bail!("online_pa full CUDA epochs must be positive");
    }
    let cuda_ordinal = exact_cuda_ordinal(requested_device_policy, effective_device_policy)?;
    let device_identity = query_cuda_device_identity(cuda_ordinal)?;
    let rows = raw_features.nrows();
    let cols = raw_features.ncols();
    let rows_u32 = checked_u32(rows, "row count")?;
    let cols_u32 = checked_u32(cols, "feature count")?;
    let feature_count = checked_mul(rows, cols, "raw feature element count")?;
    let feature_count_u32 = checked_u32(feature_count, "raw feature element count")?;
    let weight_count = checked_mul(CLASS_COUNT, cols, "weight element count")?;
    let weight_count_u32 = checked_u32(weight_count, "weight element count")?;
    let label_indicator_count = checked_mul(rows, LABEL_CHANNEL_COUNT, "label indicator count")?;
    let label_partial_count =
        checked_ceil_div(rows, LABEL_ROWS_PER_PARTIAL, "label partial count")?;
    let label_partial_channel_count = checked_mul(
        label_partial_count,
        LABEL_CHANNEL_COUNT,
        "label partial channel count",
    )?;
    let scaler_partial_count =
        checked_ceil_div(rows, SCALER_ROWS_PER_PARTIAL, "scaler partial count")?;
    let scaler_partial_value_count =
        checked_mul(scaler_partial_count, cols, "scaler partial value count")?;
    let transform_work_item_count = checked_ceil_div(
        feature_count,
        TRANSFORM_ELEMENTS_PER_WORK_ITEM,
        "transform work item count",
    )?;
    let transform_fault_partial_count = checked_ceil_div(
        transform_work_item_count,
        TRANSFORM_FAULTS_PER_PARTIAL,
        "transform fault partial count",
    )?;
    let training_row_chunk_count_per_epoch = checked_ceil_div(
        rows,
        PA_TRAINING_ROWS_PER_LAUNCH,
        "training row chunk count",
    )?;
    let training_launch_count = checked_mul(
        epochs,
        training_row_chunk_count_per_epoch,
        "training launch count",
    )?;
    let whole_fit_kernel_launch_count =
        checked_add(9, training_launch_count, "whole-fit kernel launch count")?;
    let label_partial_count_u32 = checked_u32(label_partial_count, "label partial count")?;
    let label_partial_channel_count_u32 =
        checked_u32(label_partial_channel_count, "label partial channel count")?;
    let scaler_partial_count_u32 = checked_u32(scaler_partial_count, "scaler partial count")?;
    let scaler_partial_value_count_u32 =
        checked_u32(scaler_partial_value_count, "scaler partial value count")?;
    let transform_work_item_count_u32 =
        checked_u32(transform_work_item_count, "transform work item count")?;
    let transform_fault_partial_count_u32 = checked_u32(
        transform_fault_partial_count,
        "transform fault partial count",
    )?;

    let raw_feature_bytes = bytes_for::<f64>(feature_count, "raw feature bytes")?;
    let original_label_bytes = bytes_for::<i32>(rows, "original label bytes")?;
    let remapped_label_bytes = bytes_for::<i32>(rows, "remapped label bytes")?;
    let label_indicator_bytes = bytes_for::<u32>(label_indicator_count, "label indicator bytes")?;
    let label_partial_bytes = bytes_for::<u32>(label_partial_channel_count, "label partial bytes")?;
    let label_fault_bytes = bytes_for::<u32>(LABEL_CHANNEL_COUNT, "label fault bytes")?;
    let class_count_bytes = bytes_for::<u32>(CLASS_COUNT, "class-count bytes")?;
    let class_weight_bytes = bytes_for::<f64>(CLASS_COUNT, "class-weight bytes")?;
    let scaler_partial_f64_bytes =
        bytes_for::<f64>(scaler_partial_value_count, "scaler partial f64 bytes")?;
    let scaler_partial_fault_bytes =
        bytes_for::<u32>(scaler_partial_value_count, "scaler partial fault bytes")?;
    let scaler_parameter_bytes = bytes_for::<f64>(cols, "scaler-parameter bytes")?;
    let scaler_fault_bytes = bytes_for::<u32>(cols, "scaler-fault bytes")?;
    let scaled_feature_bytes = bytes_for::<f64>(feature_count, "scaled feature bytes")?;
    let transform_work_fault_bytes =
        bytes_for::<u32>(transform_work_item_count, "transform work fault bytes")?;
    let transform_fault_partial_bytes = bytes_for::<u32>(
        transform_fault_partial_count,
        "transform fault partial bytes",
    )?;
    let weight_bytes = bytes_for::<f64>(weight_count, "weight bytes")?;
    let bias_bytes = bytes_for::<f64>(CLASS_COUNT, "bias bytes")?;
    let arithmetic_status_bytes = bytes_for::<u32>(1, "arithmetic-status bytes")?;

    let raw_features_flat = raw_features.iter().copied().collect::<Vec<_>>();
    let _residency = cubecl_residency_scope();
    let client = cubecl_cuda_client(cuda_ordinal);
    preflight_device_memory(
        &client,
        cuda_ordinal,
        &[
            raw_feature_bytes,
            original_label_bytes,
            remapped_label_bytes,
            label_indicator_bytes,
            label_partial_bytes,
            label_fault_bytes,
            class_count_bytes,
            class_weight_bytes,
            scaler_partial_f64_bytes,
            scaler_partial_f64_bytes,
            scaler_partial_fault_bytes,
            scaler_parameter_bytes,
            scaler_parameter_bytes,
            scaler_fault_bytes,
            scaled_feature_bytes,
            transform_work_fault_bytes,
            transform_fault_partial_bytes,
            weight_bytes,
            bias_bytes,
            arithmetic_status_bytes,
        ],
    )?;

    // Only raw f64 features and original i32 labels cross H2D.
    let raw_features_handle = client.create_from_slice(f64::as_bytes(&raw_features_flat));
    let original_labels_handle = client.create_from_slice(i32::as_bytes(original_labels));
    let remapped_labels_handle = client.empty(remapped_label_bytes);
    let label_indicators_handle = client.empty(label_indicator_bytes);
    let label_partials_handle = client.empty(label_partial_bytes);
    let label_faults_handle = client.empty(label_fault_bytes);
    let class_counts_handle = client.empty(class_count_bytes);
    let class_weights_handle = client.empty(class_weight_bytes);
    let scaler_partial_means_handle = client.empty(scaler_partial_f64_bytes);
    let scaler_partial_m2s_handle = client.empty(scaler_partial_f64_bytes);
    let scaler_partial_faults_handle = client.empty(scaler_partial_fault_bytes);
    let scaler_means_handle = client.empty(scaler_parameter_bytes);
    let scaler_stds_handle = client.empty(scaler_parameter_bytes);
    let scaler_faults_handle = client.empty(scaler_fault_bytes);
    let scaled_features_handle = client.empty(scaled_feature_bytes);
    let transform_work_faults_handle = client.empty(transform_work_fault_bytes);
    let transform_fault_partials_handle = client.empty(transform_fault_partial_bytes);
    let weights_handle = client.empty(weight_bytes);
    let bias_handle = client.empty(bias_bytes);
    let arithmetic_status_handle = client.empty(arithmetic_status_bytes);

    let initialization_items = weight_count.max(CLASS_COUNT).max(1);
    online_pa_full_pipeline_initialize_v2_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(
            cube_count_1d(initialization_items, "initialization cube count")?,
            1,
            1,
        ),
        CubeDim::new_1d(PA_CUBE_UNITS as u32),
        unsafe { ArrayArg::from_raw_parts(weights_handle.clone(), weight_count) },
        unsafe { ArrayArg::from_raw_parts(bias_handle.clone(), CLASS_COUNT) },
        unsafe { ArrayArg::from_raw_parts(arithmetic_status_handle.clone(), 1) },
        weight_count_u32,
    );
    online_pa_original_label_map_v2_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(cube_count_1d(rows, "label map cube count")?, 1, 1),
        CubeDim::new_1d(PA_CUBE_UNITS as u32),
        unsafe { ArrayArg::from_raw_parts(original_labels_handle, rows) },
        unsafe { ArrayArg::from_raw_parts(remapped_labels_handle.clone(), rows) },
        unsafe { ArrayArg::from_raw_parts(label_indicators_handle.clone(), label_indicator_count) },
        rows_u32,
    );
    online_pa_label_count_partial_v2_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(
            cube_count_1d(label_partial_channel_count, "label partial cube count")?,
            1,
            1,
        ),
        CubeDim::new_1d(PA_CUBE_UNITS as u32),
        unsafe { ArrayArg::from_raw_parts(label_indicators_handle, label_indicator_count) },
        unsafe {
            ArrayArg::from_raw_parts(label_partials_handle.clone(), label_partial_channel_count)
        },
        rows_u32,
        label_partial_channel_count_u32,
    );
    online_pa_label_count_weight_finalize_v2_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(LABEL_CHANNEL_COUNT as u32),
        unsafe { ArrayArg::from_raw_parts(label_partials_handle, label_partial_channel_count) },
        unsafe { ArrayArg::from_raw_parts(class_counts_handle.clone(), CLASS_COUNT) },
        unsafe { ArrayArg::from_raw_parts(class_weights_handle.clone(), CLASS_COUNT) },
        unsafe { ArrayArg::from_raw_parts(label_faults_handle.clone(), LABEL_CHANNEL_COUNT) },
        label_partial_count_u32,
        rows_u32,
        f64::MAX,
    );
    online_pa_ddof0_scaler_partial_v2_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(
            cube_count_1d(scaler_partial_value_count, "scaler partial cube count")?,
            1,
            1,
        ),
        CubeDim::new_1d(PA_CUBE_UNITS as u32),
        unsafe { ArrayArg::from_raw_parts(raw_features_handle.clone(), feature_count) },
        unsafe {
            ArrayArg::from_raw_parts(
                scaler_partial_means_handle.clone(),
                scaler_partial_value_count,
            )
        },
        unsafe {
            ArrayArg::from_raw_parts(
                scaler_partial_m2s_handle.clone(),
                scaler_partial_value_count,
            )
        },
        unsafe {
            ArrayArg::from_raw_parts(
                scaler_partial_faults_handle.clone(),
                scaler_partial_value_count,
            )
        },
        rows_u32,
        cols_u32,
        scaler_partial_value_count_u32,
        f64::MAX,
    );
    online_pa_ddof0_scaler_finalize_v2_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(cube_count_1d(cols, "scaler finalize cube count")?, 1, 1),
        CubeDim::new_1d(PA_CUBE_UNITS as u32),
        unsafe {
            ArrayArg::from_raw_parts(scaler_partial_means_handle, scaler_partial_value_count)
        },
        unsafe { ArrayArg::from_raw_parts(scaler_partial_m2s_handle, scaler_partial_value_count) },
        unsafe {
            ArrayArg::from_raw_parts(scaler_partial_faults_handle, scaler_partial_value_count)
        },
        unsafe { ArrayArg::from_raw_parts(scaler_means_handle.clone(), cols) },
        unsafe { ArrayArg::from_raw_parts(scaler_stds_handle.clone(), cols) },
        unsafe { ArrayArg::from_raw_parts(scaler_faults_handle.clone(), cols) },
        rows_u32,
        cols_u32,
        scaler_partial_count_u32,
        f64::MAX,
        1.0e-12,
    );
    online_pa_scaler_transform_chunked_v2_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(
            cube_count_1d(transform_work_item_count, "transform cube count")?,
            1,
            1,
        ),
        CubeDim::new_1d(PA_CUBE_UNITS as u32),
        unsafe { ArrayArg::from_raw_parts(raw_features_handle, feature_count) },
        unsafe { ArrayArg::from_raw_parts(scaler_means_handle.clone(), cols) },
        unsafe { ArrayArg::from_raw_parts(scaler_stds_handle.clone(), cols) },
        unsafe { ArrayArg::from_raw_parts(scaled_features_handle.clone(), feature_count) },
        unsafe {
            ArrayArg::from_raw_parts(
                transform_work_faults_handle.clone(),
                transform_work_item_count,
            )
        },
        feature_count_u32,
        transform_work_item_count_u32,
        cols_u32,
        f64::MAX,
    );
    online_pa_transform_fault_partial_v2_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(
            cube_count_1d(
                transform_fault_partial_count,
                "transform fault partial cube count",
            )?,
            1,
            1,
        ),
        CubeDim::new_1d(PA_CUBE_UNITS as u32),
        unsafe {
            ArrayArg::from_raw_parts(transform_work_faults_handle, transform_work_item_count)
        },
        unsafe {
            ArrayArg::from_raw_parts(
                transform_fault_partials_handle.clone(),
                transform_fault_partial_count,
            )
        },
        transform_work_item_count_u32,
        transform_fault_partial_count_u32,
    );
    online_pa_preprocess_fault_finalize_v2_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(label_faults_handle, LABEL_CHANNEL_COUNT) },
        unsafe { ArrayArg::from_raw_parts(scaler_faults_handle, cols) },
        unsafe {
            ArrayArg::from_raw_parts(
                transform_fault_partials_handle,
                transform_fault_partial_count,
            )
        },
        unsafe { ArrayArg::from_raw_parts(arithmetic_status_handle.clone(), 1) },
        cols_u32,
        transform_fault_partial_count_u32,
    );
    // Exact epoch-major chunks; the first D2H follows the complete sequence.
    for _epoch in 0..epochs {
        for chunk_start in (0..rows).step_by(PA_TRAINING_ROWS_PER_LAUNCH) {
            let chunk_rows = (rows - chunk_start).min(PA_TRAINING_ROWS_PER_LAUNCH);
            passive_aggressive_prediction_based_weighted_slack_v2_epoch_chunk_v3_kernel::launch::<
                CudaRuntime,
            >(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(PA_CUBE_UNITS as u32),
                unsafe { ArrayArg::from_raw_parts(scaled_features_handle.clone(), feature_count) },
                unsafe { ArrayArg::from_raw_parts(remapped_labels_handle.clone(), rows) },
                unsafe { ArrayArg::from_raw_parts(class_weights_handle.clone(), CLASS_COUNT) },
                unsafe { ArrayArg::from_raw_parts(weights_handle.clone(), weight_count) },
                unsafe { ArrayArg::from_raw_parts(bias_handle.clone(), CLASS_COUNT) },
                unsafe { ArrayArg::from_raw_parts(arithmetic_status_handle.clone(), 1) },
                checked_u32(chunk_start, "training row start")?,
                checked_u32(chunk_rows, "training row count")?,
                cols_u32,
                aggressiveness,
                f64::MAX,
            );
        }
    }

    let arithmetic_status = read_arithmetic_status(&client, arithmetic_status_handle)?;
    fail_for_full_pipeline_status(arithmetic_status)?;
    let class_counts = read_u32_buffer(&client, class_counts_handle, "class counts")?;
    let class_slack_weights =
        read_f64_buffer(&client, class_weights_handle, "class-slack weights")?;
    let scaler_means = read_f64_buffer(&client, scaler_means_handle, "scaler means")?;
    let scaler_stds = read_f64_buffer(&client, scaler_stds_handle, "scaler stds")?;
    let weights = read_f64_buffer(&client, weights_handle, "weights")?;
    let bias = read_f64_buffer(&client, bias_handle, "bias")?;
    if class_counts.len() != CLASS_COUNT
        || class_slack_weights.len() != CLASS_COUNT
        || scaler_means.len() != cols
        || scaler_stds.len() != cols
        || weights.len() != weight_count
        || bias.len() != CLASS_COUNT
    {
        bail!(
            "online_pa full CUDA artifact readback dimensions do not match rows={rows}, cols={cols}"
        );
    }

    let class_counts = [class_counts[0], class_counts[1], class_counts[2]];
    let class_slack_weights = [
        class_slack_weights[0],
        class_slack_weights[1],
        class_slack_weights[2],
    ];
    let artifact_d2h_bytes_usize = [
        arithmetic_status_bytes,
        class_count_bytes,
        class_weight_bytes,
        scaler_parameter_bytes,
        scaler_parameter_bytes,
        weight_bytes,
        bias_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_add(total, bytes, "full-pipeline artifact D2H evidence")
    })?;
    let _whole_fit_h2d_bytes = checked_add(
        raw_feature_bytes,
        original_label_bytes,
        "whole fit H2D evidence",
    )?;
    let raw_feature_h2d_bytes = checked_u64(raw_feature_bytes, "raw feature H2D evidence")?;
    let original_label_h2d_bytes =
        checked_u64(original_label_bytes, "original label H2D evidence")?;
    let artifact_d2h_bytes = checked_u64(artifact_d2h_bytes_usize, "artifact D2H evidence")?;
    let ordered_sample_visits = checked_mul(rows, epochs, "ordered sample visits")?;
    let training_rows_per_launch = checked_u64(
        PA_TRAINING_ROWS_PER_LAUNCH,
        "training rows-per-launch evidence",
    )?;
    let training_row_chunk_count_per_epoch = checked_u64(
        training_row_chunk_count_per_epoch,
        "training row-chunk evidence",
    )?;
    let training_epoch_count = checked_u64(epochs, "training epoch evidence")?;
    let training_launch_count = checked_u64(training_launch_count, "training launch evidence")?;
    let whole_fit_kernel_launch_count = checked_u64(
        whole_fit_kernel_launch_count,
        "whole-fit kernel launch evidence",
    )?;
    let training_backend = format!("online_pa_cuda_full_pipeline[cuda:{cuda_ordinal}]");
    let inference_backend = format!("online_pa_cuda_fused_inference[cuda:{cuda_ordinal}]");
    let full_pipeline = PassiveAggressiveCudaPipelineEvidenceV1 {
        execution_pipeline_schema: ONLINE_PA_CUDA_FULL_PIPELINE_SCHEMA_V3.to_string(),
        evidence_scope_schema: "neoethos.online_pa.cuda_pipeline_stages.v3".to_string(),
        requested_device_policy: requested_device_policy.to_string(),
        effective_device_policy: effective_device_policy.to_string(),
        preprocessing_backend: ONLINE_PA_CUDA_PREPROCESSING_BACKEND_V2.to_string(),
        training_backend: training_backend.clone(),
        bound_inference_backend: inference_backend,
        device_identity: device_identity.clone(),
        loss_cost_policy: "rho(y,r)=1; PA-I cap=C*w_y".to_string(),
        residency_scope: "call_scoped".to_string(),
        persistent_model_buffers: false,
        class_counts,
        kernel_launch_count: whole_fit_kernel_launch_count,
        initialization_launch_count: 1,
        label_map_launch_count: 1,
        label_count_partial_launch_count: 1,
        label_count_weight_finalize_launch_count: 1,
        scaler_partial_launch_count: 1,
        scaler_finalize_launch_count: 1,
        scaler_transform_launch_count: 1,
        transform_fault_partial_launch_count: 1,
        preprocess_fault_finalize_launch_count: 1,
        training_rows_per_launch,
        training_row_chunk_count_per_epoch,
        training_epoch_count,
        training_launch_count,
        training_interchunk_device_to_host_bytes: 0,
        raw_feature_h2d_bytes,
        original_label_h2d_bytes,
        scaled_feature_h2d_bytes: 0,
        remapped_label_h2d_bytes: 0,
        class_slack_weight_h2d_bytes: 0,
        parameter_initialization_h2d_bytes: 0,
        artifact_d2h_bytes,
    };

    Ok(PassiveAggressiveCudaFullFitV1 {
        scaler_means,
        scaler_stds,
        class_counts,
        class_slack_weights,
        weights: Array2::from_shape_vec((CLASS_COUNT, cols), weights)
            .context("shape online_pa full CUDA weights")?,
        bias: Array1::from_vec(bias),
        runtime_backend: training_backend,
        effective_device_policy: effective_device_policy.to_string(),
        training_semantics_schema: ONLINE_PA_CUDA_TRAINING_SEMANTICS_V2.to_string(),
        device_identity,
        evidence: PassiveAggressiveCudaEvidenceV1 {
            evidence_scope_schema: "neoethos.online_pa.cuda_evidence.whole_fit_call.v3".to_string(),
            kernel_launch_count: whole_fit_kernel_launch_count,
            host_to_device_bytes: raw_feature_h2d_bytes + original_label_h2d_bytes,
            device_to_host_bytes: artifact_d2h_bytes,
            ordered_sample_visits: checked_u64(ordered_sample_visits, "visit evidence")?,
            full_pipeline: Some(full_pipeline),
        },
    })
}
