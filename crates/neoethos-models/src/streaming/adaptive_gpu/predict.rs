use anyhow::{Context, Result, bail};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use ndarray::{Array1, Array2};

use super::super::adaptive_impl::{
    PassiveAggressiveCudaDeviceIdentityV1, PassiveAggressiveCudaInferenceEvidenceV1,
};
use super::device_utils::{
    bytes_for, checked_add, checked_mul, checked_u32, checked_u64, exact_cuda_ordinal,
    preflight_device_memory, query_cuda_device_identity, read_f64_buffer, read_u32_buffer,
};
use super::inference::online_pa_fused_raw_scale_logits_softmax_v1_kernel;
use super::{CLASS_COUNT, PA_CUBE_UNITS};
use crate::cubecl_lifecycle::{cubecl_cuda_client, cubecl_residency_scope};

#[derive(Debug)]
pub(crate) struct PassiveAggressiveCudaPredictionV1 {
    pub probabilities: Array2<f64>,
    pub runtime_backend: String,
    pub effective_device_policy: String,
    pub device_identity: PassiveAggressiveCudaDeviceIdentityV1,
    pub evidence: PassiveAggressiveCudaInferenceEvidenceV1,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_predict_passive_aggressive_cuda_full_pipeline(
    requested_device_policy: &str,
    effective_device_policy: &str,
    expected_device_identity: &PassiveAggressiveCudaDeviceIdentityV1,
    raw_features: &Array2<f64>,
    scaler_means: &[f64],
    scaler_stds: &[f64],
    weights: &Array2<f64>,
    bias: &Array1<f64>,
) -> Result<PassiveAggressiveCudaPredictionV1> {
    if raw_features.nrows() == 0 || raw_features.ncols() == 0 {
        bail!("online_pa fused CUDA inference requires a non-empty raw feature matrix");
    }
    let rows = raw_features.nrows();
    let cols = raw_features.ncols();
    if scaler_means.len() != cols || scaler_stds.len() != cols {
        bail!(
            "online_pa fused CUDA scaler dimension mismatch: {cols} cols vs {} means/{} stds",
            scaler_means.len(),
            scaler_stds.len()
        );
    }
    if weights.dim() != (CLASS_COUNT, cols) || bias.len() != CLASS_COUNT {
        bail!(
            "online_pa fused CUDA parameter dimension mismatch: weights {:?}, bias {}",
            weights.dim(),
            bias.len()
        );
    }

    let cuda_ordinal = exact_cuda_ordinal(requested_device_policy, effective_device_policy)?;
    let device_identity = query_cuda_device_identity(cuda_ordinal)?;
    if &device_identity != expected_device_identity {
        bail!(
            "online_pa CUDA physical device identity mismatch for inference: expected {expected_device_identity:?}, found {device_identity:?}"
        );
    }

    let rows_u32 = checked_u32(rows, "inference row count")?;
    let cols_u32 = checked_u32(cols, "inference feature count")?;
    let feature_count = checked_mul(rows, cols, "inference feature element count")?;
    let weight_count = checked_mul(CLASS_COUNT, cols, "inference weight element count")?;
    let probability_count = checked_mul(rows, CLASS_COUNT, "probability element count")?;
    let raw_feature_bytes = bytes_for::<f64>(feature_count, "inference raw feature bytes")?;
    let scaler_parameter_bytes = bytes_for::<f64>(cols, "inference scaler-parameter bytes")?;
    let weight_bytes = bytes_for::<f64>(weight_count, "inference weight bytes")?;
    let bias_bytes = bytes_for::<f64>(CLASS_COUNT, "inference bias bytes")?;
    let probability_bytes = bytes_for::<f64>(probability_count, "probability bytes")?;
    let row_status_bytes = bytes_for::<u32>(rows, "inference row-status bytes")?;

    let raw_features_flat = raw_features.iter().copied().collect::<Vec<_>>();
    let weights_flat = weights.iter().copied().collect::<Vec<_>>();
    let bias_flat = bias.iter().copied().collect::<Vec<_>>();
    let _residency = cubecl_residency_scope();
    let client = cubecl_cuda_client(cuda_ordinal);
    preflight_device_memory(
        &client,
        cuda_ordinal,
        &[
            raw_feature_bytes,
            scaler_parameter_bytes,
            scaler_parameter_bytes,
            weight_bytes,
            bias_bytes,
            probability_bytes,
            row_status_bytes,
        ],
    )?;
    let raw_features_handle = client.create_from_slice(f64::as_bytes(&raw_features_flat));
    let scaler_means_handle = client.create_from_slice(f64::as_bytes(scaler_means));
    let scaler_stds_handle = client.create_from_slice(f64::as_bytes(scaler_stds));
    let weights_handle = client.create_from_slice(f64::as_bytes(&weights_flat));
    let bias_handle = client.create_from_slice(f64::as_bytes(&bias_flat));
    let probabilities_handle = client.empty(probability_bytes);
    let row_status_handle = client.empty(row_status_bytes);

    online_pa_fused_raw_scale_logits_softmax_v1_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(rows_u32, 1, 1),
        CubeDim::new_1d(PA_CUBE_UNITS as u32),
        unsafe { ArrayArg::from_raw_parts(raw_features_handle, feature_count) },
        unsafe { ArrayArg::from_raw_parts(scaler_means_handle, cols) },
        unsafe { ArrayArg::from_raw_parts(scaler_stds_handle, cols) },
        unsafe { ArrayArg::from_raw_parts(weights_handle, weight_count) },
        unsafe { ArrayArg::from_raw_parts(bias_handle, CLASS_COUNT) },
        unsafe { ArrayArg::from_raw_parts(probabilities_handle.clone(), probability_count) },
        unsafe { ArrayArg::from_raw_parts(row_status_handle.clone(), rows) },
        rows_u32,
        cols_u32,
        f64::MAX,
    );

    let row_status = read_u32_buffer(&client, row_status_handle, "inference row status")?;
    if row_status.len() != rows {
        bail!(
            "online_pa fused CUDA row-status readback mismatch: expected {rows}, got {}",
            row_status.len()
        );
    }
    if let Some((row, status)) = row_status
        .iter()
        .copied()
        .enumerate()
        .find(|(_, status)| *status != 0)
    {
        bail!("online_pa fused CUDA inference device fault code {status} at row {row}");
    }
    let probabilities = read_f64_buffer(&client, probabilities_handle, "probabilities")?;
    if probabilities.len() != probability_count {
        bail!(
            "online_pa fused CUDA probability readback mismatch: expected {probability_count}, got {}",
            probabilities.len()
        );
    }
    let scaler_h2d_bytes = checked_mul(2, scaler_parameter_bytes, "scaler H2D evidence")?;
    let model_h2d_bytes = checked_add(weight_bytes, bias_bytes, "model H2D evidence")?;
    let host_to_device_bytes = [raw_feature_bytes, scaler_h2d_bytes, model_h2d_bytes]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            checked_add(total, bytes, "whole predict H2D evidence")
        })?;
    let device_to_host_bytes = checked_add(
        probability_bytes,
        row_status_bytes,
        "whole predict D2H evidence",
    )?;
    Ok(PassiveAggressiveCudaPredictionV1 {
        probabilities: Array2::from_shape_vec((rows, CLASS_COUNT), probabilities)
            .context("shape online_pa fused CUDA probabilities")?,
        runtime_backend: format!("online_pa_cuda_fused_inference[cuda:{cuda_ordinal}]"),
        effective_device_policy: effective_device_policy.to_string(),
        device_identity: device_identity.clone(),
        evidence: PassiveAggressiveCudaInferenceEvidenceV1 {
            evidence_scope_schema: "neoethos.online_pa.cuda_evidence.whole_predict_call.v2"
                .to_string(),
            requested_device_policy: requested_device_policy.to_string(),
            effective_device_policy: effective_device_policy.to_string(),
            device_identity,
            residency_scope: "call_scoped".to_string(),
            persistent_model_buffers: false,
            kernel_launch_count: 1,
            host_to_device_bytes: checked_u64(host_to_device_bytes, "whole predict H2D evidence")?,
            device_to_host_bytes: checked_u64(device_to_host_bytes, "whole predict D2H evidence")?,
            raw_feature_h2d_bytes: checked_u64(raw_feature_bytes, "inference raw H2D evidence")?,
            scaler_parameter_h2d_bytes: checked_u64(
                scaler_h2d_bytes,
                "inference scaler H2D evidence",
            )?,
            model_parameter_h2d_bytes: checked_u64(
                model_h2d_bytes,
                "inference model H2D evidence",
            )?,
            probability_d2h_bytes: checked_u64(probability_bytes, "probability D2H evidence")?,
            status_d2h_bytes: checked_u64(row_status_bytes, "status D2H evidence")?,
        },
    })
}
