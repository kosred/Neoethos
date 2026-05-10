use anyhow::{Context, Result, bail};
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use ndarray::Array2;

const CLASS_COUNT: usize = 3;
const L2_WEIGHT: f32 = 1.0e-4;

#[cube(launch)]
fn candidate_loss_kernel(
    candidates: &Array<f32>,
    features: &Array<f32>,
    labels: &Array<i32>,
    losses: &mut Array<f32>,
    n_rows: u32,
    input_dim: u32,
    hidden_dim: u32,
    param_dim: u32,
) {
    if ABSOLUTE_POS < losses.len() {
        let candidate = ABSOLUTE_POS;
        let param_base = candidate * param_dim;
        let w1_offset = param_base;
        let b1_offset = w1_offset + input_dim * hidden_dim;
        let w2_offset = b1_offset + hidden_dim;
        let b2_offset = w2_offset + hidden_dim * CLASS_COUNT as u32;

        let mut l2 = 0.0f32;
        let mut p = 0u32;
        while p < param_dim {
            let value = candidates[param_base + p];
            l2 += value * value;
            p += 1;
        }
        l2 = l2 / param_dim as f32;

        if n_rows == 0 {
            losses[candidate] = L2_WEIGHT * l2;
            return;
        }

        let mut total_loss = 0.0f32;
        let mut row = 0u32;
        while row < n_rows {
            let mut logit0 = candidates[b2_offset];
            let mut logit1 = candidates[b2_offset + 1];
            let mut logit2 = candidates[b2_offset + 2];

            let mut hidden = 0u32;
            while hidden < hidden_dim {
                let mut activation = candidates[b1_offset + hidden];
                let mut feature = 0u32;
                while feature < input_dim {
                    activation += features[row * input_dim + feature]
                        * candidates[w1_offset + feature * hidden_dim + hidden];
                    feature += 1;
                }
                activation = activation.tanh();
                logit0 += activation * candidates[w2_offset + hidden * CLASS_COUNT as u32];
                logit1 += activation * candidates[w2_offset + hidden * CLASS_COUNT as u32 + 1];
                logit2 += activation * candidates[w2_offset + hidden * CLASS_COUNT as u32 + 2];
                hidden += 1;
            }

            let mut max_logit = logit0;
            if logit1 > max_logit {
                max_logit = logit1;
            }
            if logit2 > max_logit {
                max_logit = logit2;
            }
            let e0 = (logit0 - max_logit).exp();
            let e1 = (logit1 - max_logit).exp();
            let e2 = (logit2 - max_logit).exp();
            let denom = e0 + e1 + e2;
            let label = labels[row];
            let mut probability = if label == 0 {
                e0 / denom
            } else if label == 1 {
                e1 / denom
            } else {
                e2 / denom
            };
            if probability < 1.0e-6 {
                probability = 1.0e-6;
            }
            if probability > 0.999999 {
                probability = 0.999999;
            }
            total_loss -= probability.ln();
            row += 1;
        }

        losses[candidate] = total_loss / n_rows as f32 + L2_WEIGHT * l2;
    }
}

pub(crate) fn neuro_evo_cuda_kernel_enabled(policy: &str) -> bool {
    let requested_gpu = {
        let normalized = policy.trim().to_ascii_lowercase();
        normalized == "gpu" || normalized.starts_with("gpu:")
    };
    let env_enabled = !matches!(
        std::env::var("FOREX_BOT_NEURO_EVO_CUDA_KERNEL")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "0" | "false" | "off" | "disable" | "disabled")
    );
    requested_gpu && env_enabled
}

fn cuda_device_id(policy: &str) -> usize {
    std::env::var("FOREX_BOT_NEURO_EVO_CUDA_DEVICE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| {
            policy
                .trim()
                .to_ascii_lowercase()
                .strip_prefix("gpu:")
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(0)
}

fn kernel_units(client: &ComputeClient<CudaRuntime>) -> u32 {
    let max_units = client.properties().hardware.max_units_per_cube.max(1);
    std::env::var("FOREX_BOT_NEURO_EVO_KERNEL_UNITS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(max_units)
        .min(max_units)
        .max(1)
}

fn flatten_candidates(candidates: &[Vec<f64>], param_dim: usize) -> Result<Vec<f32>> {
    let mut flat = Vec::with_capacity(candidates.len().saturating_mul(param_dim));
    for candidate in candidates {
        if candidate.len() != param_dim {
            bail!(
                "neuro-evo cuda candidate dimension mismatch: expected {}, received {}",
                param_dim,
                candidate.len()
            );
        }
        flat.extend(candidate.iter().map(|value| *value as f32));
    }
    Ok(flat)
}

fn flatten_features(features: &Array2<f32>, input_dim: usize) -> Result<Vec<f32>> {
    crate::common::cuda_flatten_features(features, input_dim, "neuro-evo")
}

fn launch_loss_kernel(
    client: &ComputeClient<CudaRuntime>,
    candidates_flat: &[f32],
    features: &Array2<f32>,
    labels: &[usize],
    candidate_count: usize,
    input_dim: usize,
    hidden_dim: usize,
    param_dim: usize,
) -> Result<Vec<f32>> {
    if candidate_count == 0 {
        return Ok(Vec::new());
    }
    if features.nrows() != labels.len() {
        bail!(
            "neuro-evo cuda labels mismatch: {} labels for {} feature rows",
            labels.len(),
            features.nrows()
        );
    }
    if labels.iter().any(|label| *label >= CLASS_COUNT) {
        bail!("neuro-evo cuda labels must be in 0..3");
    }

    let features_flat = flatten_features(features, input_dim)?;
    let labels = labels.iter().map(|label| *label as i32).collect::<Vec<_>>();
    let candidates_handle = client.create_from_slice(f32::as_bytes(candidates_flat));
    let features_handle = client.create_from_slice(f32::as_bytes(&features_flat));
    let labels_handle = client.create_from_slice(i32::as_bytes(&labels));
    let losses_handle = client.empty(candidate_count.saturating_mul(std::mem::size_of::<f32>()));

    let units = kernel_units(client);
    let cubes = (candidate_count as u32).div_ceil(units);
    candidate_loss_kernel::launch::<CudaRuntime>(
        client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts::<f32>(&candidates_handle, candidates_flat.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&features_handle, features_flat.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&labels_handle, labels.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&losses_handle, candidate_count, 1) },
        ScalarArg::new(features.nrows() as u32),
        ScalarArg::new(input_dim as u32),
        ScalarArg::new(hidden_dim as u32),
        ScalarArg::new(param_dim as u32),
    )
    .context("launch neuro-evo cuda loss kernel")?;

    let bytes = client.read_one(losses_handle);
    Ok(f32::from_bytes(&bytes).to_vec())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_selection_losses_cuda(
    candidates: &[Vec<f64>],
    train_features: &Array2<f32>,
    train_labels: &[usize],
    val_features: &Array2<f32>,
    val_labels: &[usize],
    input_dim: usize,
    hidden_dim: usize,
    param_dim: usize,
    policy: &str,
) -> Result<Vec<(f64, f64, f64)>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let candidates_flat = flatten_candidates(candidates, param_dim)?;
    let device = CudaDevice::new(cuda_device_id(policy));
    let client = CudaRuntime::client(&device);
    let train_losses = launch_loss_kernel(
        &client,
        &candidates_flat,
        train_features,
        train_labels,
        candidates.len(),
        input_dim,
        hidden_dim,
        param_dim,
    )?;
    let val_losses = if val_labels.is_empty() {
        train_losses.clone()
    } else {
        launch_loss_kernel(
            &client,
            &candidates_flat,
            val_features,
            val_labels,
            candidates.len(),
            input_dim,
            hidden_dim,
            param_dim,
        )?
    };

    Ok(train_losses
        .into_iter()
        .zip(val_losses)
        .map(|(train_loss, val_loss)| {
            let train_loss = train_loss as f64;
            let val_loss = val_loss as f64;
            let selection_loss = if val_labels.is_empty() {
                train_loss
            } else {
                0.65 * train_loss + 0.35 * val_loss
            };
            (selection_loss, train_loss, val_loss)
        })
        .collect())
}
