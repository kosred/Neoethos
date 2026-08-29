#[cfg(test)]
use anyhow::{Context, Result, bail};
#[cfg(test)]
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
#[cfg(test)]
use ndarray::{Array1, Array2};

#[cfg(test)]
use super::super::adaptive_impl::{
    ONLINE_PA_CUDA_TRAINING_SEMANTICS_V2, PassiveAggressiveCudaEvidenceV1,
};
#[cfg(test)]
use super::CLASS_COUNT;
#[cfg(test)]
use super::device_utils::{
    bytes_for, checked_add, checked_ceil_div, checked_mul, checked_u32, checked_u64,
    preflight_device_memory, read_arithmetic_status, read_f64_buffer,
};
use super::{DEVICE_ARITHMETIC_REDUCTION_FAULT, DEVICE_ARITHMETIC_UPDATE_FAULT, PA_CUBE_UNITS};
#[cfg(test)]
use crate::cubecl_lifecycle::{cubecl_cuda_client, cubecl_residency_scope};

const SHARED_NORM_OFFSET: usize = 0;
const SHARED_SCORE_0_OFFSET: usize = PA_CUBE_UNITS;
const SHARED_SCORE_1_OFFSET: usize = PA_CUBE_UNITS * 2;
const SHARED_SCORE_2_OFFSET: usize = PA_CUBE_UNITS * 3;
const SHARED_TAU_OFFSET: usize = PA_CUBE_UNITS * 4;
const SHARED_PREDICTED_OFFSET: usize = SHARED_TAU_OFFSET + 1;
const SHARED_UPDATE_FAULT_OFFSET: usize = SHARED_PREDICTED_OFFSET + 1;
const SHARED_VALUES: usize = SHARED_UPDATE_FAULT_OFFSET + PA_CUBE_UNITS;
pub(super) const PA_TRAINING_ROWS_PER_LAUNCH: usize = 1_024;

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct PassiveAggressiveCudaFitV2 {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub runtime_backend: String,
    pub effective_device_policy: String,
    pub training_semantics_schema: String,
    pub evidence: PassiveAggressiveCudaEvidenceV1,
}

#[cube]
fn reduce_pa_partials(shared: &mut SharedMemory<f64>, unit: usize, stride: usize) {
    if unit < stride {
        shared[SHARED_NORM_OFFSET + unit] =
            shared[SHARED_NORM_OFFSET + unit] + shared[SHARED_NORM_OFFSET + unit + stride];
        shared[SHARED_SCORE_0_OFFSET + unit] =
            shared[SHARED_SCORE_0_OFFSET + unit] + shared[SHARED_SCORE_0_OFFSET + unit + stride];
        shared[SHARED_SCORE_1_OFFSET + unit] =
            shared[SHARED_SCORE_1_OFFSET + unit] + shared[SHARED_SCORE_1_OFFSET + unit + stride];
        shared[SHARED_SCORE_2_OFFSET + unit] =
            shared[SHARED_SCORE_2_OFFSET + unit] + shared[SHARED_SCORE_2_OFFSET + unit + stride];
    }
    sync_cube();
}

#[cube]
fn reduce_pa_update_faults(shared: &mut SharedMemory<f64>, unit: usize, stride: usize) {
    if unit < stride {
        shared[SHARED_UPDATE_FAULT_OFFSET + unit] = shared[SHARED_UPDATE_FAULT_OFFSET + unit]
            + shared[SHARED_UPDATE_FAULT_OFFSET + unit + stride];
    }
    sync_cube();
}

/// Crammer et al. section 8 prediction-based cost-sensitive update. The base
/// PB-v2 cost is uniform rho(y,r)=1 for r!=y. NeoEthos changes only the PA-I
/// slack cap to C*w_y; it does not multiply the unit-margin numerator.
#[cube(launch)]
pub(super) fn passive_aggressive_prediction_based_weighted_slack_v2_epoch_chunk_v3_kernel(
    features: &Array<f64>,
    labels: &Array<i32>,
    class_weights: &Array<f64>,
    weights: &mut Array<f64>,
    bias: &mut Array<f64>,
    arithmetic_status: &mut Array<u32>,
    row_start: u32,
    row_count: u32,
    cols: u32,
    aggressiveness: f64,
    finite_limit: f64,
) {
    let unit = UNIT_POS as usize;
    let cols_us = cols as usize;
    let mut shared = SharedMemory::<f64>::new(SHARED_VALUES);
    if arithmetic_status[0] != 0 {
        terminate!();
    }

    for row_offset in 0..row_count as usize {
        if arithmetic_status[0] != 0 {
            terminate!();
        }
        let row = row_start as usize + row_offset;
        let norm = RuntimeCell::<f64>::new(0.0);
        let score_0 = RuntimeCell::<f64>::new(0.0);
        let score_1 = RuntimeCell::<f64>::new(0.0);
        let score_2 = RuntimeCell::<f64>::new(0.0);
        let row_base = row * cols_us;
        for col in range_stepped(unit, cols_us, PA_CUBE_UNITS) {
            let feature = features[row_base + col];
            norm.store(norm.read() + feature * feature);
            score_0.store(score_0.read() + feature * weights[col]);
            score_1.store(score_1.read() + feature * weights[cols_us + col]);
            score_2.store(score_2.read() + feature * weights[cols_us * 2 + col]);
        }
        shared[SHARED_NORM_OFFSET + unit] = norm.read();
        shared[SHARED_SCORE_0_OFFSET + unit] = score_0.read();
        shared[SHARED_SCORE_1_OFFSET + unit] = score_1.read();
        shared[SHARED_SCORE_2_OFFSET + unit] = score_2.read();
        sync_cube();

        reduce_pa_partials(&mut shared, unit, 128);
        reduce_pa_partials(&mut shared, unit, 64);
        reduce_pa_partials(&mut shared, unit, 32);
        reduce_pa_partials(&mut shared, unit, 16);
        reduce_pa_partials(&mut shared, unit, 8);
        reduce_pa_partials(&mut shared, unit, 4);
        reduce_pa_partials(&mut shared, unit, 2);
        reduce_pa_partials(&mut shared, unit, 1);

        if unit == 0 {
            let reduced_norm = shared[SHARED_NORM_OFFSET];
            let s0 = shared[SHARED_SCORE_0_OFFSET] + bias[0];
            let s1 = shared[SHARED_SCORE_1_OFFSET] + bias[1];
            let s2 = shared[SHARED_SCORE_2_OFFSET] + bias[2];
            let reduction_finite = reduced_norm <= finite_limit
                && reduced_norm >= 0.0
                && s0 <= finite_limit
                && s0 >= -finite_limit
                && s1 <= finite_limit
                && s1 >= -finite_limit
                && s2 <= finite_limit
                && s2 >= -finite_limit;
            let predicted = RuntimeCell::<u32>::new(0);
            let best_score = RuntimeCell::<f64>::new(s0);
            if s1 >= best_score.read() {
                best_score.store(s1);
                predicted.store(1);
            }
            if s2 >= best_score.read() {
                best_score.store(s2);
                predicted.store(2);
            }

            let target = labels[row] as usize;
            let predicted_us = predicted.read() as usize;
            let tau = RuntimeCell::<f64>::new(0.0);
            if reduction_finite && predicted_us != target {
                let margin = best_score.read()
                    - if target == 0 {
                        s0
                    } else if target == 1 {
                        s1
                    } else {
                        s2
                    }
                    + 1.0;
                if margin > 0.0 {
                    let augmented_norm_sq = reduced_norm + 1.0;
                    let denominator = 2.0 * augmented_norm_sq;
                    let candidate = margin / denominator;
                    let weighted_cap = aggressiveness * class_weights[target];
                    let tau_finite = margin <= finite_limit
                        && denominator <= finite_limit
                        && denominator > 0.0
                        && candidate <= finite_limit
                        && candidate >= 0.0
                        && weighted_cap <= finite_limit
                        && weighted_cap > 0.0;
                    if tau_finite {
                        tau.store(candidate);
                        if tau.read() > weighted_cap {
                            tau.store(weighted_cap);
                        }
                    } else {
                        arithmetic_status[0] = DEVICE_ARITHMETIC_REDUCTION_FAULT;
                    }
                }
            } else if !reduction_finite {
                arithmetic_status[0] = DEVICE_ARITHMETIC_REDUCTION_FAULT;
            }
            shared[SHARED_TAU_OFFSET] = tau.read();
            shared[SHARED_PREDICTED_OFFSET] = predicted.read() as f64;
        }
        sync_cube();

        let tau = shared[SHARED_TAU_OFFSET];
        let update_fault = RuntimeCell::<f64>::new(0.0);
        if tau > 0.0 {
            let target = labels[row] as usize;
            let predicted = shared[SHARED_PREDICTED_OFFSET] as usize;
            for col in range_stepped(unit, cols_us, PA_CUBE_UNITS) {
                let delta = tau * features[row_base + col];
                let target_pos = target * cols_us + col;
                let predicted_pos = predicted * cols_us + col;
                let target_next = weights[target_pos] + delta;
                let predicted_next = weights[predicted_pos] - delta;
                let update_finite = delta <= finite_limit
                    && delta >= -finite_limit
                    && target_next <= finite_limit
                    && target_next >= -finite_limit
                    && predicted_next <= finite_limit
                    && predicted_next >= -finite_limit;
                if !update_finite {
                    update_fault.store(1.0);
                }
            }
            if unit == 0 {
                let target_bias_next = bias[target] + tau;
                let predicted_bias_next = bias[predicted] - tau;
                let bias_finite = target_bias_next <= finite_limit
                    && target_bias_next >= -finite_limit
                    && predicted_bias_next <= finite_limit
                    && predicted_bias_next >= -finite_limit;
                if !bias_finite {
                    update_fault.store(1.0);
                }
            }
        }
        shared[SHARED_UPDATE_FAULT_OFFSET + unit] = update_fault.read();
        sync_cube();
        reduce_pa_update_faults(&mut shared, unit, 128);
        reduce_pa_update_faults(&mut shared, unit, 64);
        reduce_pa_update_faults(&mut shared, unit, 32);
        reduce_pa_update_faults(&mut shared, unit, 16);
        reduce_pa_update_faults(&mut shared, unit, 8);
        reduce_pa_update_faults(&mut shared, unit, 4);
        reduce_pa_update_faults(&mut shared, unit, 2);
        reduce_pa_update_faults(&mut shared, unit, 1);
        if unit == 0 && shared[SHARED_UPDATE_FAULT_OFFSET] > 0.0 {
            arithmetic_status[0] = DEVICE_ARITHMETIC_UPDATE_FAULT;
            shared[SHARED_TAU_OFFSET] = 0.0;
        }
        sync_cube();

        let checked_tau = shared[SHARED_TAU_OFFSET];
        if checked_tau > 0.0 {
            let target = labels[row] as usize;
            let predicted = shared[SHARED_PREDICTED_OFFSET] as usize;
            for col in range_stepped(unit, cols_us, PA_CUBE_UNITS) {
                let delta = checked_tau * features[row_base + col];
                let target_pos = target * cols_us + col;
                let predicted_pos = predicted * cols_us + col;
                weights[target_pos] = weights[target_pos] + delta;
                weights[predicted_pos] = weights[predicted_pos] - delta;
            }
            if unit == 0 {
                bias[target] = bias[target] + checked_tau;
                bias[predicted] = bias[predicted] - checked_tau;
            }
        }
        sync_storage();
    }
}

#[cfg(test)]
fn validate_inputs(
    features: &Array2<f64>,
    labels: &[usize],
    class_weights: &[f64; CLASS_COUNT],
    aggressiveness: f64,
    epochs: usize,
) -> Result<()> {
    if features.nrows() == 0 || features.ncols() == 0 {
        bail!("online_pa CUDA requires a non-empty feature matrix");
    }
    if labels.len() != features.nrows() {
        bail!(
            "online_pa CUDA row/label mismatch: {} rows vs {} labels",
            features.nrows(),
            labels.len()
        );
    }
    if labels.iter().any(|label| *label >= CLASS_COUNT) {
        bail!("online_pa CUDA labels must be in 0..3");
    }
    if features.iter().any(|value| !value.is_finite()) {
        bail!("online_pa CUDA feature matrix contains non-finite values");
    }
    if class_weights
        .iter()
        .any(|weight| !weight.is_finite() || !(0.5..=4.0).contains(weight))
    {
        bail!(
            "online_pa CUDA class-slack weights must be finite within the explicit [0.5, 4.0] policy"
        );
    }
    if !aggressiveness.is_finite() || aggressiveness <= 0.0 {
        bail!("online_pa CUDA aggressiveness must be finite and positive");
    }
    if epochs == 0 {
        bail!("online_pa CUDA epochs must be positive");
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fit_passive_aggressive_cuda(
    device_policy: &str,
    features: &Array2<f64>,
    labels: &[usize],
    class_weights: &[f64; CLASS_COUNT],
    aggressiveness: f64,
    epochs: usize,
) -> Result<PassiveAggressiveCudaFitV2> {
    validate_inputs(features, labels, class_weights, aggressiveness, epochs)?;
    let cuda_ordinal = match crate::common::parse_cuda_device_policy(device_policy)? {
        crate::common::CudaDevicePolicy::Gpu { ordinal } => ordinal,
        crate::common::CudaDevicePolicy::Auto => {
            bail!("online_pa CUDA requires an explicitly resolved CUDA device, not auto")
        }
        crate::common::CudaDevicePolicy::Cpu => {
            bail!("online_pa CUDA cannot execute a CPU device policy")
        }
    };
    let rows = features.nrows();
    let cols = features.ncols();
    let cols_u32 = checked_u32(cols, "feature count")?;
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
    let feature_count = checked_mul(rows, cols, "feature element count")?;
    let weight_count = checked_mul(CLASS_COUNT, cols, "weight element count")?;
    let feature_bytes = bytes_for::<f64>(feature_count, "feature bytes")?;
    let label_bytes = bytes_for::<i32>(rows, "label bytes")?;
    let class_weight_bytes = bytes_for::<f64>(CLASS_COUNT, "class-weight bytes")?;
    let weight_bytes = bytes_for::<f64>(weight_count, "weight bytes")?;
    let bias_bytes = bytes_for::<f64>(CLASS_COUNT, "bias bytes")?;
    let arithmetic_status_bytes = bytes_for::<u32>(1, "arithmetic-status bytes")?;

    let features_flat = features.iter().copied().collect::<Vec<_>>();
    let labels_flat = labels.iter().map(|label| *label as i32).collect::<Vec<_>>();
    let initial_weights = vec![0.0_f64; weight_count];
    let initial_bias = vec![0.0_f64; CLASS_COUNT];
    let initial_arithmetic_status = [0_u32];
    let _residency = cubecl_residency_scope();
    let client = cubecl_cuda_client(cuda_ordinal);
    preflight_device_memory(
        &client,
        cuda_ordinal,
        &[
            feature_bytes,
            label_bytes,
            class_weight_bytes,
            weight_bytes,
            bias_bytes,
            arithmetic_status_bytes,
        ],
    )?;
    let features_handle = client.create_from_slice(f64::as_bytes(&features_flat));
    let labels_handle = client.create_from_slice(i32::as_bytes(&labels_flat));
    let class_weights_handle = client.create_from_slice(f64::as_bytes(class_weights));
    let weights_handle = client.create_from_slice(f64::as_bytes(&initial_weights));
    let bias_handle = client.create_from_slice(f64::as_bytes(&initial_bias));
    let arithmetic_status_handle =
        client.create_from_slice(u32::as_bytes(&initial_arithmetic_status));

    for _epoch in 0..epochs {
        for chunk_start in (0..rows).step_by(PA_TRAINING_ROWS_PER_LAUNCH) {
            let chunk_rows = (rows - chunk_start).min(PA_TRAINING_ROWS_PER_LAUNCH);
            passive_aggressive_prediction_based_weighted_slack_v2_epoch_chunk_v3_kernel::launch::<
                CudaRuntime,
            >(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(PA_CUBE_UNITS as u32),
                unsafe { ArrayArg::from_raw_parts(features_handle.clone(), feature_count) },
                unsafe { ArrayArg::from_raw_parts(labels_handle.clone(), rows) },
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
    if arithmetic_status != 0 {
        bail!("online_pa CUDA device arithmetic fault code {arithmetic_status}");
    }
    let weights = read_f64_buffer(&client, weights_handle, "weights")?;
    let bias = read_f64_buffer(&client, bias_handle, "bias")?;
    if weights.len() != weight_count || bias.len() != CLASS_COUNT {
        bail!(
            "online_pa CUDA readback mismatch: {} weights/{weight_count}, {} bias/{CLASS_COUNT}",
            weights.len(),
            bias.len()
        );
    }
    if weights
        .iter()
        .chain(bias.iter())
        .any(|value| !value.is_finite())
    {
        bail!("online_pa CUDA produced non-finite parameters");
    }

    let host_to_device_bytes = [
        feature_bytes,
        label_bytes,
        class_weight_bytes,
        weight_bytes,
        bias_bytes,
        arithmetic_status_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_add(total, bytes, "host-to-device evidence")
    })?;
    let device_to_host_bytes = checked_add(weight_bytes, bias_bytes, "readback evidence")?;
    let device_to_host_bytes = checked_add(
        device_to_host_bytes,
        arithmetic_status_bytes,
        "arithmetic-status readback evidence",
    )?;
    let ordered_sample_visits = checked_mul(rows, epochs, "ordered sample visits")?;
    Ok(PassiveAggressiveCudaFitV2 {
        weights: Array2::from_shape_vec((CLASS_COUNT, cols), weights)
            .context("shape online_pa CUDA weights")?,
        bias: Array1::from_vec(bias),
        runtime_backend: format!("online_pa_cuda[cuda:{cuda_ordinal}]"),
        effective_device_policy: format!("gpu:{cuda_ordinal}"),
        training_semantics_schema: ONLINE_PA_CUDA_TRAINING_SEMANTICS_V2.to_string(),
        evidence: PassiveAggressiveCudaEvidenceV1 {
            evidence_scope_schema: "neoethos.online_pa.cuda_evidence.training_stage.v2".to_string(),
            kernel_launch_count: checked_u64(training_launch_count, "training launch evidence")?,
            host_to_device_bytes: u64::try_from(host_to_device_bytes)
                .context("online_pa H2D evidence exceeds u64")?,
            device_to_host_bytes: u64::try_from(device_to_host_bytes)
                .context("online_pa D2H evidence exceeds u64")?,
            ordered_sample_visits: u64::try_from(ordered_sample_visits)
                .context("online_pa visit evidence exceeds u64")?,
            full_pipeline: None,
        },
    })
}
