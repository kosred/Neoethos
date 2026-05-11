use anyhow::{Context, Result, bail};
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use ndarray::{Array1, Array2};

use super::common::normalize_statistical_device_policy;

const CLASS_COUNT: usize = 3;

pub(crate) struct LinearCudaFit {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub runtime_backend: String,
}

#[cube]
fn sign_f32(value: f32) -> f32 {
    let mut out: f32 = 0.0;
    if value > 0.0 {
        out = 1.0;
    } else if value < 0.0 {
        out = -1.0;
    }
    out
}

#[cube]
fn clamp_probability(value: f32) -> f32 {
    let mut out = value;
    if out < 0.000001 {
        out = 0.000001;
    }
    if out > 0.999999 {
        out = 0.999999;
    }
    out
}

#[cube]
fn class_probability(
    features: &Array<f32>,
    weights: &Array<f32>,
    bias: &Array<f32>,
    row: u32,
    cols: u32,
    class_idx: u32,
) -> f32 {
    let mut logit0 = bias[0];
    let mut logit1 = bias[1];
    let mut logit2 = bias[2];
    let row_us = row as usize;
    let cols_us = cols as usize;
    let row_base = row_us * cols_us;
    let mut col = 0usize;
    while col < cols_us {
        let feature = features[row_base + col];
        let weight_base = col * CLASS_COUNT;
        // cubecl 0.9: compound assignment panics on Const-init mut bindings.
        logit0 = logit0 + feature * weights[weight_base];
        logit1 = logit1 + feature * weights[weight_base + 1];
        logit2 = logit2 + feature * weights[weight_base + 2];
        col = col + 1;
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
    let mut out = e2 / denom;
    if class_idx == 0 {
        out = e0 / denom;
    } else if class_idx == 1 {
        out = e1 / denom;
    }
    out
}

#[cube(launch)]
fn softmax_gradient_kernel(
    features: &Array<f32>,
    labels: &Array<i32>,
    weights: &Array<f32>,
    bias: &Array<f32>,
    grad_weights: &mut Array<f32>,
    grad_bias: &mut Array<f32>,
    rows: u32,
    cols: u32,
    alpha: f32,
    l1_ratio: f32,
) {
    let cols_us = cols as usize;
    let rows_us = rows as usize;
    let weight_len = cols_us * CLASS_COUNT;
    let total_len = weight_len + CLASS_COUNT;
    if ABSOLUTE_POS < total_len {
        let pos = ABSOLUTE_POS;
        let is_bias = pos >= weight_len;
        let mut class_idx: usize = pos % CLASS_COUNT;
        if is_bias {
            class_idx = pos - weight_len;
        }
        let mut feature_idx: usize = pos / CLASS_COUNT;
        if is_bias {
            feature_idx = 0;
        }

        let mut grad: f32 = 0.0;
        let mut row = 0usize;
        while row < rows_us {
            let probability =
                class_probability(features, weights, bias, row as u32, cols, class_idx as u32);
            let label = labels[row];
            let mut target: f32 = 0.0;
            if label == class_idx as i32 {
                target = 1.0;
            }
            let error = probability - target;
            if is_bias {
                grad = grad + error;
            } else {
                grad = grad + features[row * cols_us + feature_idx] * error;
            }
            row = row + 1;
        }
        grad = grad / rows as f32;

        if is_bias {
            grad_bias[class_idx] = grad;
        } else {
            let weight = weights[pos];
            let l2 = (1.0 - l1_ratio) * weight;
            let l1 = l1_ratio * sign_f32(weight);
            grad_weights[pos] = grad + alpha * (l2 + l1);
        }
    }
}

#[cube(launch)]
fn softmax_apply_kernel(
    weights: &mut Array<f32>,
    bias: &mut Array<f32>,
    grad_weights: &Array<f32>,
    grad_bias: &Array<f32>,
    learning_rate: f32,
    weight_len: u32,
) {
    let weight_len = weight_len as usize;
    let total_len = weight_len + CLASS_COUNT;
    if ABSOLUTE_POS < total_len {
        let pos = ABSOLUTE_POS;
        if pos < weight_len {
            weights[pos] -= learning_rate * grad_weights[pos];
        } else {
            let class_idx = pos - weight_len;
            bias[class_idx] -= learning_rate * grad_bias[class_idx];
        }
    }
}

#[cube(launch)]
fn softmax_loss_kernel(
    features: &Array<f32>,
    labels: &Array<i32>,
    weights: &Array<f32>,
    bias: &Array<f32>,
    loss_out: &mut Array<f32>,
    rows: u32,
    cols: u32,
) {
    if ABSOLUTE_POS == 0 {
        if rows == 0 {
            loss_out[0] = 0.0;
            terminate!();
        }

        let mut loss: f32 = 0.0;
        let rows_us = rows as usize;
        let mut row = 0usize;
        while row < rows_us {
            let label = labels[row] as u32;
            let probability = class_probability(features, weights, bias, row as u32, cols, label);
            loss = loss - clamp_probability(probability).ln();
            row = row + 1;
        }
        loss_out[0] = loss / rows as f32;
    }
}

#[cube(launch)]
fn softmax_predict_kernel(
    features: &Array<f32>,
    weights: &Array<f32>,
    bias: &Array<f32>,
    probabilities_out: &mut Array<f32>,
    rows: u32,
    cols: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let base = row * CLASS_COUNT;
        probabilities_out[base] = class_probability(features, weights, bias, row as u32, cols, 0);
        probabilities_out[base + 1] =
            class_probability(features, weights, bias, row as u32, cols, 1);
        probabilities_out[base + 2] =
            class_probability(features, weights, bias, row as u32, cols, 2);
    }
}

pub(crate) fn statistical_cuda_kernel_enabled(model_name: &str) -> bool {
    let requested = requested_policy(model_name);
    let normalized = normalize_statistical_device_policy(&requested);
    let requested_gpu = normalized == "gpu" || normalized.starts_with("gpu:");
    let global_enabled = !is_disabled_env("FOREX_BOT_STATISTICAL_CUDA_KERNEL");
    let model_env = format!(
        "FOREX_BOT_{}_CUDA_KERNEL",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    requested_gpu && global_enabled && !is_disabled_env(&model_env)
}

fn is_disabled_env(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "0" | "false" | "off" | "disable" | "disabled")
    )
}

fn requested_policy(model_name: &str) -> String {
    let model_key = format!(
        "FOREX_BOT_{}_DEVICE",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    std::env::var(&model_key)
        .or_else(|_| std::env::var("FOREX_BOT_META_DEVICE"))
        .unwrap_or_else(|_| "auto".to_string())
}

fn cuda_device_id(model_name: &str) -> usize {
    let model_key = format!(
        "FOREX_BOT_{}_CUDA_DEVICE",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    std::env::var(&model_key)
        .or_else(|_| std::env::var("FOREX_BOT_STATISTICAL_CUDA_DEVICE"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| {
            normalize_statistical_device_policy(&requested_policy(model_name))
                .strip_prefix("gpu:")
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(0)
}

fn kernel_units(client: &ComputeClient<CudaRuntime>) -> u32 {
    let max_units = client.properties().hardware.max_units_per_cube.max(1);
    std::env::var("FOREX_BOT_STATISTICAL_KERNEL_UNITS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(max_units)
        .min(max_units)
        .max(1)
}

fn flatten_features(features: &Array2<f32>, cols: usize) -> Result<Vec<f32>> {
    let flat = crate::common::cuda_flatten_features(features, cols, "statistical")?;
    if flat.iter().any(|value| !value.is_finite()) {
        bail!("statistical cuda feature matrix contains non-finite values");
    }
    Ok(flat)
}

fn flatten_labels(labels: &[usize], rows: usize) -> Result<Vec<i32>> {
    if labels.len() != rows {
        bail!(
            "statistical cuda label mismatch: {} labels for {} feature rows",
            labels.len(),
            rows
        );
    }
    if labels.iter().any(|label| *label >= CLASS_COUNT) {
        bail!("statistical cuda labels must be in 0..3");
    }
    Ok(labels.iter().map(|label| *label as i32).collect())
}

fn read_f32_buffer(
    client: &ComputeClient<CudaRuntime>,
    handle: cubecl::server::Handle,
) -> Vec<f32> {
    let bytes = client.read_one(handle);
    f32::from_bytes(&bytes).to_vec()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fit_linear_softmax_cuda(
    model_name: &str,
    train_features: &Array2<f32>,
    train_labels: &[usize],
    val_features: Option<&Array2<f32>>,
    val_labels: &[usize],
    alpha: f32,
    l1_ratio: f32,
    learning_rate: f32,
    epochs: usize,
) -> Result<LinearCudaFit> {
    let rows = train_features.nrows();
    let cols = train_features.ncols();
    if rows == 0 || cols == 0 {
        bail!("statistical cuda training requires a non-empty feature matrix");
    }
    if val_features.is_none() && !val_labels.is_empty() {
        bail!("statistical cuda validation labels were provided without validation features");
    }

    let device = CudaDevice::new(cuda_device_id(model_name));
    let client = CudaRuntime::client(&device);
    let units = kernel_units(&client);

    let features_flat = flatten_features(train_features, cols)?;
    let labels_flat = flatten_labels(train_labels, rows)?;
    let features_handle = client.create_from_slice(f32::as_bytes(&features_flat));
    let labels_handle = client.create_from_slice(i32::as_bytes(&labels_flat));

    let weight_len = cols.saturating_mul(CLASS_COUNT);
    let initial_weights = vec![0.0f32; weight_len];
    let initial_bias = vec![0.0f32; CLASS_COUNT];
    let weights_handle = client.create_from_slice(f32::as_bytes(&initial_weights));
    let bias_handle = client.create_from_slice(f32::as_bytes(&initial_bias));
    let grad_weights_handle = client.empty(weight_len.saturating_mul(std::mem::size_of::<f32>()));
    let grad_bias_handle = client.empty(CLASS_COUNT.saturating_mul(std::mem::size_of::<f32>()));

    let validation = if let Some(val_features) = val_features {
        let val_rows = val_features.nrows();
        let val_features_flat = flatten_features(val_features, cols)?;
        let val_labels_flat = flatten_labels(val_labels, val_rows)?;
        Some((
            val_rows,
            client.create_from_slice(f32::as_bytes(&val_features_flat)),
            client.create_from_slice(i32::as_bytes(&val_labels_flat)),
        ))
    } else {
        None
    };
    let loss_handle = client.empty(std::mem::size_of::<f32>());

    let mut best_weights = Vec::<f32>::new();
    let mut best_bias = Vec::<f32>::new();
    let mut best_val_loss = f32::INFINITY;
    let mut stale_epochs = 0usize;
    let patience = 25usize;
    let total_params = weight_len + CLASS_COUNT;
    let grad_cubes = (total_params as u32).div_ceil(units);

    for _ in 0..epochs.max(1) {
        softmax_gradient_kernel::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(grad_cubes, 1, 1),
            CubeDim::new_1d(units),
            unsafe { ArrayArg::from_raw_parts::<f32>(&features_handle, features_flat.len(), 1) },
            unsafe { ArrayArg::from_raw_parts::<i32>(&labels_handle, labels_flat.len(), 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&weights_handle, weight_len, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&bias_handle, CLASS_COUNT, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&grad_weights_handle, weight_len, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&grad_bias_handle, CLASS_COUNT, 1) },
            ScalarArg::new(rows as u32),
            ScalarArg::new(cols as u32),
            ScalarArg::new(alpha),
            ScalarArg::new(l1_ratio),
        )
        .context("launch statistical cuda softmax gradient kernel")?;

        softmax_apply_kernel::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(grad_cubes, 1, 1),
            CubeDim::new_1d(units),
            unsafe { ArrayArg::from_raw_parts::<f32>(&weights_handle, weight_len, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&bias_handle, CLASS_COUNT, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&grad_weights_handle, weight_len, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&grad_bias_handle, CLASS_COUNT, 1) },
            ScalarArg::new(learning_rate),
            ScalarArg::new(weight_len as u32),
        )
        .context("launch statistical cuda softmax apply kernel")?;

        if let Some((val_rows, val_features_handle, val_labels_handle)) = validation.as_ref() {
            softmax_loss_kernel::launch::<CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe {
                    ArrayArg::from_raw_parts::<f32>(
                        val_features_handle,
                        val_rows.saturating_mul(cols),
                        1,
                    )
                },
                unsafe { ArrayArg::from_raw_parts::<i32>(val_labels_handle, *val_rows, 1) },
                unsafe { ArrayArg::from_raw_parts::<f32>(&weights_handle, weight_len, 1) },
                unsafe { ArrayArg::from_raw_parts::<f32>(&bias_handle, CLASS_COUNT, 1) },
                unsafe { ArrayArg::from_raw_parts::<f32>(&loss_handle, 1, 1) },
                ScalarArg::new(*val_rows as u32),
                ScalarArg::new(cols as u32),
            )
            .context("launch statistical cuda softmax validation loss kernel")?;
            let loss = read_f32_buffer(&client, loss_handle.clone())
                .into_iter()
                .next()
                .context("statistical cuda validation loss missing")?;
            if loss + 1e-6 < best_val_loss {
                best_val_loss = loss;
                best_weights = read_f32_buffer(&client, weights_handle.clone());
                best_bias = read_f32_buffer(&client, bias_handle.clone());
                stale_epochs = 0;
            } else {
                stale_epochs += 1;
                if stale_epochs >= patience {
                    break;
                }
            }
        }
    }

    let weights = if best_val_loss.is_finite() {
        best_weights
    } else {
        read_f32_buffer(&client, weights_handle)
    };
    let bias = if best_val_loss.is_finite() {
        best_bias
    } else {
        read_f32_buffer(&client, bias_handle)
    };
    if weights.len() != weight_len || bias.len() != CLASS_COUNT {
        bail!(
            "statistical cuda parameter length mismatch: weights {} vs {}, bias {} vs {}",
            weights.len(),
            weight_len,
            bias.len(),
            CLASS_COUNT
        );
    }

    Ok(LinearCudaFit {
        weights: Array2::from_shape_vec((cols, CLASS_COUNT), weights)
            .context("shape statistical cuda weights")?,
        bias: Array1::from_vec(bias),
        runtime_backend: format!("{}_softmax_cuda", model_name),
    })
}

pub(crate) fn try_predict_linear_softmax_cuda(
    model_name: &str,
    features: &Array2<f32>,
    weights: &Array2<f32>,
    bias: &Array1<f32>,
) -> Result<Array2<f32>> {
    let rows = features.nrows();
    let cols = features.ncols();
    if rows == 0 {
        return Ok(Array2::<f32>::zeros((0, CLASS_COUNT)));
    }
    if weights.nrows() != cols || weights.ncols() != CLASS_COUNT || bias.len() != CLASS_COUNT {
        bail!("statistical cuda prediction received inconsistent model dimensions");
    }

    let device = CudaDevice::new(cuda_device_id(model_name));
    let client = CudaRuntime::client(&device);
    let units = kernel_units(&client);
    let features_flat = flatten_features(features, cols)?;
    let weights_flat = weights.iter().copied().collect::<Vec<_>>();
    let bias_flat = bias.iter().copied().collect::<Vec<_>>();

    let features_handle = client.create_from_slice(f32::as_bytes(&features_flat));
    let weights_handle = client.create_from_slice(f32::as_bytes(&weights_flat));
    let bias_handle = client.create_from_slice(f32::as_bytes(&bias_flat));
    let output_len = rows.saturating_mul(CLASS_COUNT);
    let output_handle = client.empty(output_len.saturating_mul(std::mem::size_of::<f32>()));
    let cubes = (rows as u32).div_ceil(units);

    softmax_predict_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts::<f32>(&features_handle, features_flat.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&weights_handle, weights_flat.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&bias_handle, bias_flat.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&output_handle, output_len, 1) },
        ScalarArg::new(rows as u32),
        ScalarArg::new(cols as u32),
    )
    .context("launch statistical cuda softmax prediction kernel")?;

    let probabilities = read_f32_buffer(&client, output_handle);
    if probabilities.len() != output_len {
        bail!(
            "statistical cuda prediction length mismatch: expected {}, received {}",
            output_len,
            probabilities.len()
        );
    }
    Array2::from_shape_vec((rows, CLASS_COUNT), probabilities)
        .context("shape statistical cuda predictions")
}
