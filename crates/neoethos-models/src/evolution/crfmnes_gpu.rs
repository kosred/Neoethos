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
        let n_rows_us = n_rows as usize;
        let input_dim_us = input_dim as usize;
        let hidden_dim_us = hidden_dim as usize;
        let param_dim_us = param_dim as usize;
        let param_base = candidate * param_dim_us;
        let w1_offset = param_base;
        let b1_offset = w1_offset + input_dim_us * hidden_dim_us;
        let w2_offset = b1_offset + hidden_dim_us;
        let b2_offset = w2_offset + hidden_dim_us * CLASS_COUNT;

        let l2 = RuntimeCell::<f32>::new(0.0);
        for p in 0..param_dim_us {
            let value = candidates[param_base + p];
            l2.store(l2.read() + value * value);
        }
        let l2_final = l2.read() / param_dim as f32;

        if n_rows == 0 {
            losses[candidate] = L2_WEIGHT * l2_final;
            terminate!();
        }

        let total_loss = RuntimeCell::<f32>::new(0.0);
        for row in 0..n_rows_us {
            let logit0 = RuntimeCell::<f32>::new(candidates[b2_offset]);
            let logit1 = RuntimeCell::<f32>::new(candidates[b2_offset + 1]);
            let logit2 = RuntimeCell::<f32>::new(candidates[b2_offset + 2]);

            for hidden in 0..hidden_dim_us {
                let activation = RuntimeCell::<f32>::new(candidates[b1_offset + hidden]);
                for feature in 0..input_dim_us {
                    activation.store(
                        activation.read()
                            + features[row * input_dim_us + feature]
                                * candidates[w1_offset + feature * hidden_dim_us + hidden],
                    );
                }
                let act = activation.read().tanh();
                logit0.store(logit0.read() + act * candidates[w2_offset + hidden * CLASS_COUNT]);
                logit1
                    .store(logit1.read() + act * candidates[w2_offset + hidden * CLASS_COUNT + 1]);
                logit2
                    .store(logit2.read() + act * candidates[w2_offset + hidden * CLASS_COUNT + 2]);
            }

            let l0 = logit0.read();
            let l1 = logit1.read();
            let l2v = logit2.read();
            let max_logit = RuntimeCell::<f32>::new(l0);
            if l1 > max_logit.read() {
                max_logit.store(l1);
            }
            if l2v > max_logit.read() {
                max_logit.store(l2v);
            }
            let m = max_logit.read();
            let e0 = (l0 - m).exp();
            let e1 = (l1 - m).exp();
            let e2 = (l2v - m).exp();
            let denom = e0 + e1 + e2;
            let label = labels[row];
            let probability = RuntimeCell::<f32>::new(e2 / denom);
            if label == 0 {
                probability.store(e0 / denom);
            } else if label == 1 {
                probability.store(e1 / denom);
            }
            if probability.read() < 1.0e-6 {
                probability.store(1.0e-6);
            }
            if probability.read() > 0.999999 {
                probability.store(0.999999);
            }
            total_loss.store(total_loss.read() - probability.read().ln());
        }

        losses[candidate] = total_loss.read() / n_rows as f32 + L2_WEIGHT * l2_final;
    }
}

pub(crate) fn neuro_evo_cuda_kernel_enabled(policy: &str) -> bool {
    crate::common::cuda_kernel_enabled(policy, "NEOETHOS_BOT_NEURO_EVO_CUDA_KERNEL")
}

fn cuda_device_id(policy: &str) -> usize {
    crate::common::cuda_device_id_from_policy(policy, "NEOETHOS_BOT_NEURO_EVO_CUDA_DEVICE", None)
}

fn kernel_units(client: &ComputeClient<CudaRuntime>) -> u32 {
    crate::common::cuda_kernel_units(
        client.properties().hardware.max_units_per_cube,
        "NEOETHOS_BOT_NEURO_EVO_KERNEL_UNITS",
    )
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
