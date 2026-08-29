use anyhow::{Context, Result, bail};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cudarc::driver::CudaContext;
use ndarray::{Array1, Array2};

use neoethos_core::BackendKind;

use crate::cubecl_lifecycle::{cubecl_cuda_client, cubecl_residency_scope};

const CLASS_COUNT: usize = 3;
const GRADIENT_ROWS_PER_PARTIAL: usize = 4096;
const LOSS_ROWS_PER_PARTIAL: usize = 1024;
const DEVICE_MEMORY_HEADROOM_PERCENT: usize = 10;
const DEVICE_MEMORY_MIN_HEADROOM_BYTES: usize = 256 * 1024 * 1024;

pub(crate) struct LinearCudaFit {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub runtime_backend: String,
    pub runtime_backend_kind: BackendKind,
}

#[cube]
fn soft_threshold_f64(value: f64, threshold: f64) -> f64 {
    // cubecl 0.10: literal-init `let mut` panics on later assignment.
    // RuntimeCell wraps the binding so writes go through `expand_no_check`.
    let out = RuntimeCell::<f64>::new(0.0);
    if value > threshold {
        out.store(value - threshold);
    } else if value < -threshold {
        out.store(value + threshold);
    }
    out.read()
}

#[cube]
fn clamp_probability(value: f64) -> f64 {
    // cubecl 0.10: even `let mut x = param;` produces an immutable
    // binding; reassignment goes through `assign_expand` which panics.
    // RuntimeCell is the only path to a runtime-mutable scalar.
    let out = RuntimeCell::<f64>::new(value);
    if out.read() < 0.000001 {
        out.store(0.000001);
    }
    if out.read() > 0.999999 {
        out.store(0.999999);
    }
    out.read()
}

#[cube]
fn class_probability(
    features: &Array<f64>,
    weights: &Array<f64>,
    bias: &Array<f64>,
    row: u32,
    cols: u32,
    class_idx: u32,
) -> f64 {
    let logit0 = RuntimeCell::<f64>::new(bias[0]);
    let logit1 = RuntimeCell::<f64>::new(bias[1]);
    let logit2 = RuntimeCell::<f64>::new(bias[2]);
    let row_us = row as usize;
    let cols_us = cols as usize;
    let row_base = row_us * cols_us;
    for col in 0..cols_us {
        let feature = features[row_base + col];
        let weight_base = col * CLASS_COUNT;
        logit0.store(logit0.read() + feature * weights[weight_base]);
        logit1.store(logit1.read() + feature * weights[weight_base + 1]);
        logit2.store(logit2.read() + feature * weights[weight_base + 2]);
    }

    let l0 = logit0.read();
    let l1 = logit1.read();
    let l2 = logit2.read();
    let max_logit = RuntimeCell::<f64>::new(l0);
    if l1 > max_logit.read() {
        max_logit.store(l1);
    }
    if l2 > max_logit.read() {
        max_logit.store(l2);
    }
    let m = max_logit.read();
    let e0 = (l0 - m).exp();
    let e1 = (l1 - m).exp();
    let e2 = (l2 - m).exp();
    let denom = e0 + e1 + e2;
    let out = RuntimeCell::<f64>::new(e2 / denom);
    if class_idx == 0 {
        out.store(e0 / denom);
    } else if class_idx == 1 {
        out.store(e1 / denom);
    }
    out.read()
}

#[cube(launch)]
fn softmax_error_kernel(
    features: &Array<f64>,
    labels: &Array<i32>,
    weights: &Array<f64>,
    bias: &Array<f64>,
    errors: &mut Array<f64>,
    rows: u32,
    cols: u32,
) {
    // Compute the three residuals once per row. The two-stage gradient below
    // reuses this matrix for every parameter/row-partial worker.
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let cols_us = cols as usize;
        let row_base = row * cols_us;
        let logit0 = RuntimeCell::<f64>::new(bias[0]);
        let logit1 = RuntimeCell::<f64>::new(bias[1]);
        let logit2 = RuntimeCell::<f64>::new(bias[2]);
        for col in 0..cols_us {
            let feature = features[row_base + col];
            let weight_base = col * CLASS_COUNT;
            logit0.store(logit0.read() + feature * weights[weight_base]);
            logit1.store(logit1.read() + feature * weights[weight_base + 1]);
            logit2.store(logit2.read() + feature * weights[weight_base + 2]);
        }

        let l0 = logit0.read();
        let l1 = logit1.read();
        let l2 = logit2.read();
        let max_logit = RuntimeCell::<f64>::new(l0);
        if l1 > max_logit.read() {
            max_logit.store(l1);
        }
        if l2 > max_logit.read() {
            max_logit.store(l2);
        }
        let m = max_logit.read();
        let e0 = (l0 - m).exp();
        let e1 = (l1 - m).exp();
        let e2 = (l2 - m).exp();
        let denom = e0 + e1 + e2;
        let error_base = row * CLASS_COUNT;
        errors[error_base] = e0 / denom;
        errors[error_base + 1] = e1 / denom;
        errors[error_base + 2] = e2 / denom;

        let label = labels[row];
        if label == 0 {
            errors[error_base] -= 1.0;
        } else if label == 1 {
            errors[error_base + 1] -= 1.0;
        } else {
            errors[error_base + 2] -= 1.0;
        }
    }
}

#[cube(launch)]
fn softmax_gradient_partials_kernel(
    features: &Array<f64>,
    errors: &Array<f64>,
    gradient_partials: &mut Array<f64>,
    rows: u32,
    cols: u32,
    rows_per_partial: u32,
    partial_count: u32,
) {
    let cols_us = cols as usize;
    let weight_len = cols_us * CLASS_COUNT;
    let total_params = weight_len + CLASS_COUNT;
    let partial_count_us = partial_count as usize;
    let total_workers = total_params * partial_count_us;
    if ABSOLUTE_POS < total_workers {
        let pos = ABSOLUTE_POS / partial_count_us;
        let partial = ABSOLUTE_POS % partial_count_us;
        let is_bias = pos >= weight_len;
        let class_idx_cell = RuntimeCell::<u32>::new((pos % CLASS_COUNT) as u32);
        if is_bias {
            class_idx_cell.store((pos - weight_len) as u32);
        }
        let feature_idx_cell = RuntimeCell::<u32>::new((pos / CLASS_COUNT) as u32);
        if is_bias {
            feature_idx_cell.store(0);
        }

        let class_idx = class_idx_cell.read() as usize;
        let feature_idx = feature_idx_cell.read() as usize;
        let start = partial * rows_per_partial as usize;
        let end = RuntimeCell::<usize>::new(start + rows_per_partial as usize);
        if end.read() > rows as usize {
            end.store(rows as usize);
        }
        let grad = RuntimeCell::<f64>::new(0.0);
        for row in start..end.read() {
            let error = errors[row * CLASS_COUNT + class_idx];
            if is_bias {
                grad.store(grad.read() + error);
            } else {
                grad.store(grad.read() + features[row * cols_us + feature_idx] * error);
            }
        }
        gradient_partials[pos * partial_count_us + partial] = grad.read();
    }
}

#[cube(launch)]
fn softmax_gradient_reduce_kernel(
    gradient_partials: &Array<f64>,
    weights: &Array<f64>,
    grad_weights: &mut Array<f64>,
    grad_bias: &mut Array<f64>,
    rows: u32,
    cols: u32,
    partial_count: u32,
    alpha: f64,
    l1_ratio: f64,
) {
    let weight_len = cols as usize * CLASS_COUNT;
    let total_params = weight_len + CLASS_COUNT;
    if ABSOLUTE_POS < total_params {
        let pos = ABSOLUTE_POS;
        let partial_sum = RuntimeCell::<f64>::new(0.0);
        for partial in 0..partial_count as usize {
            partial_sum.store(
                partial_sum.read() + gradient_partials[pos * partial_count as usize + partial],
            );
        }
        let final_grad = partial_sum.read() / rows as f64;

        if pos >= weight_len {
            grad_bias[pos - weight_len] = final_grad;
        } else {
            let weight = weights[pos];
            let l2 = (1.0 - l1_ratio) * weight;
            grad_weights[pos] = final_grad + alpha * l2;
        }
    }
}

#[cube(launch)]
fn softmax_apply_kernel(
    weights: &mut Array<f64>,
    bias: &mut Array<f64>,
    grad_weights: &Array<f64>,
    grad_bias: &Array<f64>,
    learning_rate: f64,
    l1_threshold: f64,
    weight_len: u32,
) {
    let weight_len = weight_len as usize;
    let total_len = weight_len + CLASS_COUNT;
    if ABSOLUTE_POS < total_len {
        let pos = ABSOLUTE_POS;
        if pos < weight_len {
            let updated = weights[pos] - learning_rate * grad_weights[pos];
            weights[pos] = soft_threshold_f64(updated, l1_threshold);
        } else {
            let class_idx = pos - weight_len;
            bias[class_idx] -= learning_rate * grad_bias[class_idx];
        }
    }
}

#[cube(launch)]
fn copy_best_parameters_kernel(
    weights: &Array<f64>,
    bias: &Array<f64>,
    best_weights: &mut Array<f64>,
    best_bias: &mut Array<f64>,
    weight_len: u32,
) {
    let weight_len = weight_len as usize;
    let total_len = weight_len + CLASS_COUNT;
    if ABSOLUTE_POS < total_len {
        let pos = ABSOLUTE_POS;
        if pos < weight_len {
            best_weights[pos] = weights[pos];
        } else {
            let class_idx = pos - weight_len;
            best_bias[class_idx] = bias[class_idx];
        }
    }
}

#[cube(launch)]
fn softmax_loss_rows_kernel(
    features: &Array<f64>,
    labels: &Array<i32>,
    weights: &Array<f64>,
    bias: &Array<f64>,
    losses: &mut Array<f64>,
    rows: u32,
    cols: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let label = labels[row] as u32;
        let probability = class_probability(features, weights, bias, row as u32, cols, label);
        losses[row] = -clamp_probability(probability).ln();
    }
}

#[cube(launch)]
fn partial_loss_sums_kernel(
    losses: &Array<f64>,
    partial_losses: &mut Array<f64>,
    rows: u32,
    rows_per_partial: u32,
    partial_count: u32,
) {
    if ABSOLUTE_POS < partial_count as usize {
        let partial = ABSOLUTE_POS;
        let start = partial * rows_per_partial as usize;
        let end = RuntimeCell::<usize>::new(start + rows_per_partial as usize);
        if end.read() > rows as usize {
            end.store(rows as usize);
        }
        let loss = RuntimeCell::<f64>::new(0.0);
        for row in start..end.read() {
            loss.store(loss.read() + losses[row]);
        }
        partial_losses[partial] = loss.read();
    }
}

#[cube(launch)]
fn mean_loss_kernel(
    partial_losses: &Array<f64>,
    loss_out: &mut Array<f64>,
    partial_count: u32,
    rows: u32,
) {
    if ABSOLUTE_POS == 0 {
        if rows == 0 {
            loss_out[0] = 0.0;
            terminate!();
        }

        // Both logits and the large row reduction are parallel above. This
        // final deterministic pass touches only the bounded partial buffer and
        // preserves one scalar synchronization for early stopping.
        let loss = RuntimeCell::<f64>::new(0.0);
        for partial in 0..partial_count as usize {
            loss.store(loss.read() + partial_losses[partial]);
        }
        loss_out[0] = loss.read() / rows as f64;
    }
}

#[cube(launch)]
fn softmax_predict_kernel(
    features: &Array<f64>,
    weights: &Array<f64>,
    bias: &Array<f64>,
    probabilities_out: &mut Array<f64>,
    rows: u32,
    cols: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let cols_us = cols as usize;
        let row_base = row * cols_us;
        let logit0 = RuntimeCell::<f64>::new(bias[0]);
        let logit1 = RuntimeCell::<f64>::new(bias[1]);
        let logit2 = RuntimeCell::<f64>::new(bias[2]);
        for col in 0..cols_us {
            let feature = features[row_base + col];
            let weight_base = col * CLASS_COUNT;
            logit0.store(logit0.read() + feature * weights[weight_base]);
            logit1.store(logit1.read() + feature * weights[weight_base + 1]);
            logit2.store(logit2.read() + feature * weights[weight_base + 2]);
        }

        let l0 = logit0.read();
        let l1 = logit1.read();
        let l2 = logit2.read();
        let max_logit = RuntimeCell::<f64>::new(l0);
        if l1 > max_logit.read() {
            max_logit.store(l1);
        }
        if l2 > max_logit.read() {
            max_logit.store(l2);
        }
        let maximum = max_logit.read();
        let e0 = (l0 - maximum).exp();
        let e1 = (l1 - maximum).exp();
        let e2 = (l2 - maximum).exp();
        let mass = e0 + e1 + e2;
        let base = row * CLASS_COUNT;
        probabilities_out[base] = e0 / mass;
        probabilities_out[base + 1] = e1 / mass;
        probabilities_out[base + 2] = e2 / mass;
    }
}

fn cuda_device_id(resolved_device_policy: &str) -> Result<usize> {
    // The ordinal comes from the policy (`gpu:1`), which comes from
    // `models.statistical_device`. The two env names that used to outrank
    // it are deleted — see `common::cuda_device_id_from_policy`.
    crate::common::cuda_device_id_from_policy(resolved_device_policy)
}

fn kernel_units(client: &ComputeClient<CudaRuntime>) -> u32 {
    crate::common::cuda_kernel_units(client.properties().hardware.max_units_per_cube)
}

fn checked_u32_dimension(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .with_context(|| format!("statistical CUDA {label} does not fit the u32 kernel ABI"))
}

fn checked_element_count(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("statistical CUDA {label} element count overflow"))
}

fn checked_buffer_bytes(elements: usize, label: &str) -> Result<usize> {
    elements
        .checked_mul(std::mem::size_of::<f64>())
        .with_context(|| format!("statistical CUDA {label} byte count overflow"))
}

fn checked_i32_buffer_bytes(elements: usize, label: &str) -> Result<usize> {
    elements
        .checked_mul(std::mem::size_of::<i32>())
        .with_context(|| format!("statistical CUDA {label} byte count overflow"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LinearCudaDeviceMemoryPlanV1 {
    logical_peak_bytes: usize,
    max_single_buffer_bytes: usize,
    buffer_bytes: Vec<usize>,
}

fn add_planned_device_bytes(
    plan: &mut LinearCudaDeviceMemoryPlanV1,
    bytes: usize,
    label: &str,
) -> Result<()> {
    plan.logical_peak_bytes = plan
        .logical_peak_bytes
        .checked_add(bytes)
        .with_context(|| format!("statistical CUDA planned device bytes overflow at {label}"))?;
    plan.max_single_buffer_bytes = plan.max_single_buffer_bytes.max(bytes);
    plan.buffer_bytes.push(bytes);
    Ok(())
}

fn planned_fit_device_bytes(
    rows: usize,
    cols: usize,
    validation_rows: usize,
) -> Result<LinearCudaDeviceMemoryPlanV1> {
    let weight_len = checked_element_count(cols, CLASS_COUNT, "weight matrix")?;
    let total_params = weight_len
        .checked_add(CLASS_COUNT)
        .context("statistical CUDA parameter count overflow")?;
    let gradient_partial_count = rows.div_ceil(GRADIENT_ROWS_PER_PARTIAL).max(1);
    let gradient_partials_len = checked_element_count(
        total_params,
        gradient_partial_count,
        "gradient partial matrix",
    )?;
    let train_feature_len = checked_element_count(rows, cols, "training feature matrix")?;
    let errors_len = checked_element_count(rows, CLASS_COUNT, "row-error matrix")?;

    let mut planned = LinearCudaDeviceMemoryPlanV1 {
        logical_peak_bytes: 0,
        max_single_buffer_bytes: 0,
        buffer_bytes: Vec::new(),
    };
    for (bytes, label) in [
        (
            checked_buffer_bytes(train_feature_len, "training feature matrix")?,
            "training features",
        ),
        (
            checked_i32_buffer_bytes(rows, "training labels")?,
            "training labels",
        ),
        (
            checked_buffer_bytes(errors_len, "row-error matrix")?,
            "row errors",
        ),
        (
            checked_buffer_bytes(weight_len, "weight matrix")?,
            "weights",
        ),
        (checked_buffer_bytes(CLASS_COUNT, "bias vector")?, "bias"),
        (
            checked_buffer_bytes(weight_len, "best weight matrix")?,
            "best weights",
        ),
        (
            checked_buffer_bytes(CLASS_COUNT, "best bias vector")?,
            "best bias",
        ),
        (
            checked_buffer_bytes(weight_len, "weight gradient")?,
            "weight gradient",
        ),
        (
            checked_buffer_bytes(CLASS_COUNT, "bias gradient")?,
            "bias gradient",
        ),
        (
            checked_buffer_bytes(gradient_partials_len, "gradient partial matrix")?,
            "gradient partials",
        ),
        (std::mem::size_of::<f64>(), "validation loss scalar"),
    ] {
        add_planned_device_bytes(&mut planned, bytes, label)?;
    }

    if validation_rows > 0 {
        let validation_feature_len =
            checked_element_count(validation_rows, cols, "validation feature matrix")?;
        let validation_partial_count = validation_rows.div_ceil(LOSS_ROWS_PER_PARTIAL).max(1);
        for (bytes, label) in [
            (
                checked_buffer_bytes(validation_feature_len, "validation feature matrix")?,
                "validation features",
            ),
            (
                checked_i32_buffer_bytes(validation_rows, "validation labels")?,
                "validation labels",
            ),
            (
                checked_buffer_bytes(validation_rows, "validation row losses")?,
                "validation row losses",
            ),
            (
                checked_buffer_bytes(validation_partial_count, "validation partial losses")?,
                "validation partial losses",
            ),
        ] {
            add_planned_device_bytes(&mut planned, bytes, label)?;
        }
    }

    Ok(planned)
}

fn planned_prediction_device_bytes(
    rows: usize,
    cols: usize,
) -> Result<LinearCudaDeviceMemoryPlanV1> {
    let feature_len = checked_element_count(rows, cols, "prediction feature matrix")?;
    let weight_len = checked_element_count(cols, CLASS_COUNT, "prediction weight matrix")?;
    let output_len = checked_element_count(rows, CLASS_COUNT, "prediction output matrix")?;
    let mut planned = LinearCudaDeviceMemoryPlanV1 {
        logical_peak_bytes: 0,
        max_single_buffer_bytes: 0,
        buffer_bytes: Vec::new(),
    };

    for (bytes, label) in [
        (
            checked_buffer_bytes(feature_len, "prediction feature matrix")?,
            "prediction features",
        ),
        (
            checked_buffer_bytes(weight_len, "prediction weight matrix")?,
            "prediction weights",
        ),
        (
            checked_buffer_bytes(CLASS_COUNT, "prediction bias vector")?,
            "prediction bias",
        ),
        (
            checked_buffer_bytes(output_len, "prediction output matrix")?,
            "prediction output",
        ),
    ] {
        add_planned_device_bytes(&mut planned, bytes, label)?;
    }

    Ok(planned)
}

fn align_up_checked(value: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 {
        bail!("statistical CUDA allocator reported zero memory alignment");
    }
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - remainder)
        .context("statistical CUDA allocator page alignment overflow")
}

fn allocator_page_size_for_buffer(
    bytes: usize,
    max_page_size: usize,
    alignment: usize,
) -> Result<usize> {
    if bytes == 0 {
        return Ok(0);
    }
    if bytes > max_page_size {
        bail!(
            "statistical CUDA max single buffer bytes {bytes} exceed selected device allocator page limit {max_page_size}"
        );
    }

    const MB: usize = 1024 * 1024;
    let mut current = max_page_size;
    let mut base = 1u32;
    let mut intermediate = Vec::new();
    while current >= 32 * MB {
        current /= 4;
        current = align_up_checked(current, alignment)?;
        let divisor = 1usize
            .checked_shl(base)
            .context("statistical CUDA allocator pool divisor overflow")?;
        intermediate.push((current, current / divisor));
        base = base
            .checked_add(1)
            .context("statistical CUDA allocator pool count overflow")?;
    }

    for (page_size, max_slice_size) in intermediate.into_iter().rev() {
        let close_to_page = page_size
            .checked_sub(bytes)
            .and_then(|difference| difference.checked_mul(5))
            .is_some_and(|scaled| scaled < page_size);
        if max_slice_size >= bytes || close_to_page {
            return Ok(page_size);
        }
    }

    let final_page_size = max_page_size / alignment * alignment;
    if bytes > final_page_size {
        bail!(
            "statistical CUDA max single buffer bytes {bytes} exceed aligned allocator page limit {final_page_size}"
        );
    }
    Ok(final_page_size)
}

fn allocator_reservation_upper_bound(
    plan: &LinearCudaDeviceMemoryPlanV1,
    max_page_size: usize,
    alignment: usize,
) -> Result<usize> {
    plan.buffer_bytes.iter().try_fold(0usize, |total, bytes| {
        let page_size = allocator_page_size_for_buffer(*bytes, max_page_size, alignment)?;
        total
            .checked_add(page_size)
            .context("statistical CUDA allocator reservation upper bound overflow")
    })
}

fn preflight_device_memory(
    client: &ComputeClient<CudaRuntime>,
    cuda_ordinal: usize,
    plan: &LinearCudaDeviceMemoryPlanV1,
) -> Result<()> {
    let max_page_size = usize::try_from(client.properties().memory.max_page_size)
        .context("statistical CUDA max single buffer bytes exceed usize")?;
    if plan.max_single_buffer_bytes > max_page_size {
        bail!(
            "statistical CUDA max single buffer bytes {} exceed selected device allocator page limit {max_page_size}",
            plan.max_single_buffer_bytes
        );
    }
    let memory_alignment = usize::try_from(client.properties().memory.alignment)
        .context("statistical CUDA allocator alignment exceeds usize")?;
    let allocator_reservation_bytes =
        allocator_reservation_upper_bound(plan, max_page_size, memory_alignment)?;

    let usage = client
        .memory_usage()
        .context("inspect CubeCL CUDA memory usage before statistical operation")?;
    let cuda_context = CudaContext::new(cuda_ordinal).with_context(|| {
        format!("retain selected CUDA ordinal {cuda_ordinal} for memory preflight")
    })?;
    let (free_device_bytes, total_device_bytes) = cuda_context
        .mem_get_info()
        .context("inspect selected CUDA device memory before statistical operation")?;
    if total_device_bytes == 0 {
        bail!("statistical CUDA selected device reports zero total memory");
    }

    let available_device_bytes = free_device_bytes.min(total_device_bytes);
    let percentage_headroom = total_device_bytes
        .checked_mul(DEVICE_MEMORY_HEADROOM_PERCENT)
        .context("statistical CUDA device headroom overflow")?
        / 100;
    let required_headroom = percentage_headroom
        .max(DEVICE_MEMORY_MIN_HEADROOM_BYTES)
        .min(total_device_bytes);
    let usable_device_bytes = available_device_bytes.saturating_sub(required_headroom);
    if allocator_reservation_bytes > usable_device_bytes {
        bail!(
            "statistical CUDA planned device bytes {} require allocator reservation upper bound {allocator_reservation_bytes}, exceeding available device bytes {usable_device_bytes} after reserving {required_headroom} bytes of headroom (free {free_device_bytes}, CubeCL reserved {}, CubeCL in-use {}, total {total_device_bytes})",
            plan.logical_peak_bytes,
            usage.bytes_reserved,
            usage.bytes_in_use,
        );
    }
    Ok(())
}

fn flatten_features(features: &Array2<f64>, cols: usize) -> Result<Vec<f64>> {
    if features.ncols() != cols {
        bail!(
            "statistical cuda feature width mismatch: {} columns vs expected {cols}",
            features.ncols()
        );
    }
    let flat = features.iter().copied().collect::<Vec<_>>();
    let expected = checked_element_count(features.nrows(), cols, "feature matrix")?;
    if flat.len() != expected {
        bail!(
            "statistical cuda feature length mismatch: {} values vs expected {expected}",
            flat.len()
        );
    }
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

fn read_f64_buffer(
    client: &ComputeClient<CudaRuntime>,
    handle: cubecl::server::Handle,
) -> Result<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .context("read statistical f64 CUDA device buffer")?;
    Ok(f64::from_bytes(&bytes).to_vec())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fit_linear_softmax_cuda(
    model_name: &str,
    resolved_device_policy: &str,
    train_features: &Array2<f64>,
    train_labels: &[usize],
    val_features: Option<&Array2<f64>>,
    val_labels: &[usize],
    alpha: f64,
    l1_ratio: f64,
    learning_rate: f64,
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
    if val_features.is_some_and(|features| features.nrows() == 0) {
        bail!("statistical cuda validation frame cannot be empty");
    }
    let rows_u32 = checked_u32_dimension(rows, "training row count")?;
    let cols_u32 = checked_u32_dimension(cols, "feature column count")?;
    let gradient_rows_per_partial_u32 =
        checked_u32_dimension(GRADIENT_ROWS_PER_PARTIAL, "gradient rows per partial")?;
    let loss_rows_per_partial_u32 =
        checked_u32_dimension(LOSS_ROWS_PER_PARTIAL, "loss rows per partial")?;

    let _cubecl_call_residency = cubecl_residency_scope();
    let cuda_ordinal = cuda_device_id(resolved_device_policy)?;
    let client = cubecl_cuda_client(cuda_ordinal);
    let units = kernel_units(&client);
    let validation_rows = val_features.map_or(0, |features| features.nrows());
    let memory_plan = planned_fit_device_bytes(rows, cols, validation_rows)?;
    preflight_device_memory(&client, cuda_ordinal, &memory_plan)?;

    let features_flat = flatten_features(train_features, cols)?;
    let labels_flat = flatten_labels(train_labels, rows)?;
    let feature_values_len = features_flat.len();
    let label_count = labels_flat.len();
    let features_handle = client.create_from_slice(f64::as_bytes(&features_flat));
    let labels_handle = client.create_from_slice(i32::as_bytes(&labels_flat));
    drop(features_flat);
    drop(labels_flat);
    let errors_len = checked_element_count(rows, CLASS_COUNT, "row-error matrix")?;
    let errors_handle = client.empty(checked_buffer_bytes(errors_len, "row-error matrix")?);

    let weight_len = checked_element_count(cols, CLASS_COUNT, "weight matrix")?;
    let weight_len_u32 = checked_u32_dimension(weight_len, "weight count")?;
    let total_params = weight_len
        .checked_add(CLASS_COUNT)
        .context("statistical CUDA parameter count overflow")?;
    let total_params_u32 = checked_u32_dimension(total_params, "parameter count")?;
    let gradient_partial_count = rows.div_ceil(GRADIENT_ROWS_PER_PARTIAL).max(1);
    let gradient_partial_count_u32 =
        checked_u32_dimension(gradient_partial_count, "gradient partial count")?;
    let gradient_partials_len = checked_element_count(
        total_params,
        gradient_partial_count,
        "gradient partial matrix",
    )?;
    let gradient_workers_u32 =
        checked_u32_dimension(gradient_partials_len, "gradient partial worker count")?;
    let initial_weights = vec![0.0_f64; weight_len];
    let initial_bias = vec![0.0_f64; CLASS_COUNT];
    let weights_handle = client.create_from_slice(f64::as_bytes(&initial_weights));
    let bias_handle = client.create_from_slice(f64::as_bytes(&initial_bias));
    let best_weights_handle = client.empty(checked_buffer_bytes(weight_len, "best weights")?);
    let best_bias_handle = client.empty(checked_buffer_bytes(CLASS_COUNT, "best bias")?);
    let grad_weights_handle = client.empty(checked_buffer_bytes(weight_len, "weight gradient")?);
    let grad_bias_handle = client.empty(checked_buffer_bytes(CLASS_COUNT, "bias gradient")?);
    let gradient_partials_handle = client.empty(checked_buffer_bytes(
        gradient_partials_len,
        "gradient partial matrix",
    )?);

    let validation = if let Some(val_features) = val_features {
        let val_rows = val_features.nrows();
        let val_rows_u32 = checked_u32_dimension(val_rows, "validation row count")?;
        let val_features_flat = flatten_features(val_features, cols)?;
        let val_labels_flat = flatten_labels(val_labels, val_rows)?;
        let val_feature_values_len = val_features_flat.len();
        let validation_loss_bytes = checked_buffer_bytes(val_rows, "validation row losses")?;
        let validation_losses_handle = client.empty(validation_loss_bytes);
        let validation_partial_count = val_rows.div_ceil(LOSS_ROWS_PER_PARTIAL).max(1);
        let validation_partial_count_u32 =
            checked_u32_dimension(validation_partial_count, "validation partial count")?;
        let validation_partial_loss_bytes =
            checked_buffer_bytes(validation_partial_count, "validation partial losses")?;
        let validation_partial_losses_handle = client.empty(validation_partial_loss_bytes);
        Some((
            val_rows,
            val_rows_u32,
            val_feature_values_len,
            validation_partial_count,
            validation_partial_count_u32,
            client.create_from_slice(f64::as_bytes(&val_features_flat)),
            client.create_from_slice(i32::as_bytes(&val_labels_flat)),
            validation_losses_handle,
            validation_partial_losses_handle,
        ))
    } else {
        None
    };
    let loss_handle = client.empty(std::mem::size_of::<f64>());

    let mut best_val_loss = f64::INFINITY;
    let mut stale_epochs = 0usize;
    let patience = 25usize;
    let grad_cubes = total_params_u32.div_ceil(units);
    let gradient_partial_cubes = gradient_workers_u32.div_ceil(units);
    let error_cubes = rows_u32.div_ceil(units);

    for _ in 0..epochs.max(1) {
        softmax_error_kernel::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(error_cubes, 1, 1),
            CubeDim::new_1d(units),
            unsafe { ArrayArg::from_raw_parts(features_handle.clone(), feature_values_len) },
            unsafe { ArrayArg::from_raw_parts(labels_handle.clone(), label_count) },
            unsafe { ArrayArg::from_raw_parts(weights_handle.clone(), weight_len) },
            unsafe { ArrayArg::from_raw_parts(bias_handle.clone(), CLASS_COUNT) },
            unsafe { ArrayArg::from_raw_parts(errors_handle.clone(), errors_len) },
            rows_u32,
            cols_u32,
        );

        softmax_gradient_partials_kernel::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(gradient_partial_cubes, 1, 1),
            CubeDim::new_1d(units),
            unsafe { ArrayArg::from_raw_parts(features_handle.clone(), feature_values_len) },
            unsafe { ArrayArg::from_raw_parts(errors_handle.clone(), errors_len) },
            unsafe {
                ArrayArg::from_raw_parts(gradient_partials_handle.clone(), gradient_partials_len)
            },
            rows_u32,
            cols_u32,
            gradient_rows_per_partial_u32,
            gradient_partial_count_u32,
        );

        softmax_gradient_reduce_kernel::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(grad_cubes, 1, 1),
            CubeDim::new_1d(units),
            unsafe {
                ArrayArg::from_raw_parts(gradient_partials_handle.clone(), gradient_partials_len)
            },
            unsafe { ArrayArg::from_raw_parts(weights_handle.clone(), weight_len) },
            unsafe { ArrayArg::from_raw_parts(grad_weights_handle.clone(), weight_len) },
            unsafe { ArrayArg::from_raw_parts(grad_bias_handle.clone(), CLASS_COUNT) },
            rows_u32,
            cols_u32,
            gradient_partial_count_u32,
            alpha,
            l1_ratio,
        );

        softmax_apply_kernel::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(grad_cubes, 1, 1),
            CubeDim::new_1d(units),
            unsafe { ArrayArg::from_raw_parts(weights_handle.clone(), weight_len) },
            unsafe { ArrayArg::from_raw_parts(bias_handle.clone(), CLASS_COUNT) },
            unsafe { ArrayArg::from_raw_parts(grad_weights_handle.clone(), weight_len) },
            unsafe { ArrayArg::from_raw_parts(grad_bias_handle.clone(), CLASS_COUNT) },
            learning_rate,
            learning_rate * alpha * l1_ratio,
            weight_len_u32,
        );

        if let Some((
            val_rows,
            val_rows_u32,
            val_feature_values_len,
            validation_partial_count,
            validation_partial_count_u32,
            val_features_handle,
            val_labels_handle,
            validation_losses_handle,
            validation_partial_losses_handle,
        )) = validation.as_ref()
        {
            let loss_row_cubes = (val_rows_u32.div_ceil(units)).max(1);
            softmax_loss_rows_kernel::launch::<CudaRuntime>(
                &client,
                CubeCount::Static(loss_row_cubes, 1, 1),
                CubeDim::new_1d(units),
                unsafe {
                    ArrayArg::from_raw_parts(val_features_handle.clone(), *val_feature_values_len)
                },
                unsafe { ArrayArg::from_raw_parts(val_labels_handle.clone(), *val_rows) },
                unsafe { ArrayArg::from_raw_parts(weights_handle.clone(), weight_len) },
                unsafe { ArrayArg::from_raw_parts(bias_handle.clone(), CLASS_COUNT) },
                unsafe { ArrayArg::from_raw_parts(validation_losses_handle.clone(), *val_rows) },
                *val_rows_u32,
                cols_u32,
            );
            let partial_loss_cubes = (validation_partial_count_u32.div_ceil(units)).max(1);
            partial_loss_sums_kernel::launch::<CudaRuntime>(
                &client,
                CubeCount::Static(partial_loss_cubes, 1, 1),
                CubeDim::new_1d(units),
                unsafe { ArrayArg::from_raw_parts(validation_losses_handle.clone(), *val_rows) },
                unsafe {
                    ArrayArg::from_raw_parts(
                        validation_partial_losses_handle.clone(),
                        *validation_partial_count,
                    )
                },
                *val_rows_u32,
                loss_rows_per_partial_u32,
                *validation_partial_count_u32,
            );
            mean_loss_kernel::launch::<CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe {
                    ArrayArg::from_raw_parts(
                        validation_partial_losses_handle.clone(),
                        *validation_partial_count,
                    )
                },
                unsafe { ArrayArg::from_raw_parts(loss_handle.clone(), 1) },
                *validation_partial_count_u32,
                *val_rows_u32,
            );
            let loss = read_f64_buffer(&client, loss_handle.clone())?
                .into_iter()
                .next()
                .context("statistical cuda validation loss missing")?;
            if !loss.is_finite() {
                bail!("statistical CUDA validation loss is non-finite");
            }
            if loss + 1e-6 < best_val_loss {
                best_val_loss = loss;
                // Keep the full checkpoint resident. Only the scalar loss is
                // synchronized per epoch; weights and bias cross the device
                // boundary once, when the final artifact is constructed.
                copy_best_parameters_kernel::launch::<CudaRuntime>(
                    &client,
                    CubeCount::Static(grad_cubes, 1, 1),
                    CubeDim::new_1d(units),
                    unsafe { ArrayArg::from_raw_parts(weights_handle.clone(), weight_len) },
                    unsafe { ArrayArg::from_raw_parts(bias_handle.clone(), CLASS_COUNT) },
                    unsafe { ArrayArg::from_raw_parts(best_weights_handle.clone(), weight_len) },
                    unsafe { ArrayArg::from_raw_parts(best_bias_handle.clone(), CLASS_COUNT) },
                    weight_len_u32,
                );
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
        read_f64_buffer(&client, best_weights_handle)?
    } else {
        read_f64_buffer(&client, weights_handle)?
    };
    let bias = if best_val_loss.is_finite() {
        read_f64_buffer(&client, best_bias_handle)?
    } else {
        read_f64_buffer(&client, bias_handle)?
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
    if weights
        .iter()
        .chain(bias.iter())
        .any(|value| !value.is_finite())
    {
        bail!("statistical CUDA produced non-finite parameters");
    }

    Ok(LinearCudaFit {
        weights: Array2::from_shape_vec((cols, CLASS_COUNT), weights)
            .context("shape statistical cuda weights")?,
        bias: Array1::from_vec(bias),
        runtime_backend: format!("{}_softmax_cuda[{resolved_device_policy}]", model_name),
        runtime_backend_kind: BackendKind::NativeCuda,
    })
}

pub(crate) fn try_predict_linear_softmax_cuda(
    model_name: &str,
    resolved_device_policy: &str,
    features: &Array2<f64>,
    weights: &Array2<f64>,
    bias: &Array1<f64>,
) -> Result<Array2<f64>> {
    let rows = features.nrows();
    let cols = features.ncols();
    if rows == 0 {
        return Ok(Array2::<f64>::zeros((0, CLASS_COUNT)));
    }
    if weights.nrows() != cols || weights.ncols() != CLASS_COUNT || bias.len() != CLASS_COUNT {
        bail!("{model_name} statistical cuda prediction received inconsistent model dimensions");
    }
    let rows_u32 = checked_u32_dimension(rows, "prediction row count")?;
    let cols_u32 = checked_u32_dimension(cols, "prediction feature count")?;

    let _cubecl_call_residency = cubecl_residency_scope();
    let cuda_ordinal = cuda_device_id(resolved_device_policy)?;
    let client = cubecl_cuda_client(cuda_ordinal);
    let memory_plan = planned_prediction_device_bytes(rows, cols)?;
    preflight_device_memory(&client, cuda_ordinal, &memory_plan)?;
    let units = kernel_units(&client);
    let features_flat = flatten_features(features, cols)?;
    let weights_flat = weights.iter().copied().collect::<Vec<_>>();
    let bias_flat = bias.iter().copied().collect::<Vec<_>>();
    let feature_values_len = features_flat.len();
    let weight_values_len = weights_flat.len();
    let bias_values_len = bias_flat.len();

    let features_handle = client.create_from_slice(f64::as_bytes(&features_flat));
    let weights_handle = client.create_from_slice(f64::as_bytes(&weights_flat));
    let bias_handle = client.create_from_slice(f64::as_bytes(&bias_flat));
    drop(features_flat);
    drop(weights_flat);
    drop(bias_flat);
    let output_len = checked_element_count(rows, CLASS_COUNT, "prediction output")?;
    let output_handle = client.empty(checked_buffer_bytes(output_len, "prediction output")?);
    let cubes = rows_u32.div_ceil(units);

    softmax_predict_kernel::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts(features_handle.clone(), feature_values_len) },
        unsafe { ArrayArg::from_raw_parts(weights_handle.clone(), weight_values_len) },
        unsafe { ArrayArg::from_raw_parts(bias_handle.clone(), bias_values_len) },
        unsafe { ArrayArg::from_raw_parts(output_handle.clone(), output_len) },
        rows_u32,
        cols_u32,
    );

    let probabilities = read_f64_buffer(&client, output_handle)?;
    if probabilities.len() != output_len {
        bail!(
            "statistical cuda prediction length mismatch: expected {}, received {}",
            output_len,
            probabilities.len()
        );
    }
    let invalid_probability = probabilities.chunks_exact(CLASS_COUNT).any(|row| {
        row.iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            || (row.iter().sum::<f64>() - 1.0).abs() > 1e-9
    });
    if invalid_probability {
        bail!("statistical CUDA prediction produced invalid probabilities");
    }
    Array2::from_shape_vec((rows, CLASS_COUNT), probabilities)
        .context("shape statistical cuda predictions")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statistical_cuda_memory_plan_matches_cubecl_subslice_page_rounding() {
        const MIB: usize = 1024 * 1024;
        const GIB: usize = 1024 * MIB;
        let max_page_size = 8 * GIB;
        let alignment = 512;
        let buffers = [64 * 1024, 8 * MIB, 64 * MIB, 256 * MIB, 5 * GIB];
        let expected_pages = [8 * MIB, 8 * MIB, 512 * MIB, 2 * GIB, 8 * GIB];

        for (buffer, expected_page) in buffers.into_iter().zip(expected_pages) {
            assert_eq!(
                allocator_page_size_for_buffer(buffer, max_page_size, alignment)
                    .expect("derive CubeCL page size"),
                expected_page
            );
        }

        let plan = LinearCudaDeviceMemoryPlanV1 {
            logical_peak_bytes: buffers.into_iter().sum(),
            max_single_buffer_bytes: *buffers.iter().max().expect("memory plan buffer"),
            buffer_bytes: buffers.to_vec(),
        };
        assert_eq!(
            allocator_reservation_upper_bound(&plan, max_page_size, alignment)
                .expect("derive CubeCL reservation upper bound"),
            expected_pages.into_iter().sum::<usize>()
        );
    }

    #[test]
    fn statistical_cuda_prediction_memory_plan_accounts_for_all_transient_buffers() {
        let rows = 7usize;
        let cols = 5usize;
        let bytes_per_value = std::mem::size_of::<f64>();
        let expected_buffers = vec![
            rows * cols * bytes_per_value,
            cols * CLASS_COUNT * bytes_per_value,
            CLASS_COUNT * bytes_per_value,
            rows * CLASS_COUNT * bytes_per_value,
        ];

        let plan = planned_prediction_device_bytes(rows, cols)
            .expect("plan statistical CUDA prediction device memory");
        assert_eq!(plan.buffer_bytes, expected_buffers);
        assert_eq!(
            plan.logical_peak_bytes,
            expected_buffers.iter().sum::<usize>()
        );
        assert_eq!(
            plan.max_single_buffer_bytes,
            *expected_buffers
                .iter()
                .max()
                .expect("prediction memory plan buffer")
        );
    }

    fn cpu_soft_threshold(value: f64, threshold: f64) -> f64 {
        if value > threshold {
            value - threshold
        } else if value < -threshold {
            value + threshold
        } else {
            0.0
        }
    }

    fn training_fixture() -> (Array2<f64>, Vec<usize>, Array2<f64>, Vec<usize>) {
        let mut train = Vec::new();
        let mut train_labels = Vec::new();
        let mut validation = Vec::new();
        let mut validation_labels = Vec::new();

        for row in 0..96 {
            let class = row % CLASS_COUNT;
            let base = class as f64 - 1.0;
            let values = [base + row as f64 * 0.001, base * -0.75 + 0.2];
            if row < 72 {
                train.extend(values);
                train_labels.push(class);
            } else {
                validation.extend(values);
                validation_labels.push(class);
            }
        }

        (
            Array2::from_shape_vec((72, 2), train).expect("shape CUDA train fixture"),
            train_labels,
            Array2::from_shape_vec((24, 2), validation).expect("shape CUDA validation fixture"),
            validation_labels,
        )
    }

    fn cpu_probabilities(
        features: &Array2<f64>,
        weights: &Array2<f64>,
        bias: &Array1<f64>,
    ) -> Array2<f64> {
        let mut output = Array2::zeros((features.nrows(), CLASS_COUNT));
        for row in 0..features.nrows() {
            let mut logits = [0.0_f64; CLASS_COUNT];
            for class in 0..CLASS_COUNT {
                logits[class] = bias[class];
                for col in 0..features.ncols() {
                    logits[class] += features[(row, col)] * weights[(col, class)];
                }
            }
            let max = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
            let exps = logits.map(|value| (value - max).exp());
            let mass = exps.iter().sum::<f64>();
            for class in 0..CLASS_COUNT {
                output[(row, class)] = exps[class] / mass;
            }
        }
        output
    }

    fn cpu_one_epoch(
        features: &Array2<f64>,
        labels: &[usize],
        alpha: f64,
        l1_ratio: f64,
        learning_rate: f64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut weights = Array2::<f64>::zeros((features.ncols(), CLASS_COUNT));
        let mut bias = Array1::<f64>::zeros(CLASS_COUNT);
        let probabilities = cpu_probabilities(features, &weights, &bias);
        let rows = features.nrows() as f64;
        for class in 0..CLASS_COUNT {
            let bias_gradient = (0..features.nrows())
                .map(|row| {
                    let target = if labels[row] == class { 1.0 } else { 0.0 };
                    probabilities[(row, class)] - target
                })
                .sum::<f64>()
                / rows;
            for col in 0..features.ncols() {
                let mut weight_gradient = 0.0_f64;
                for row in 0..features.nrows() {
                    let target = if labels[row] == class { 1.0 } else { 0.0 };
                    let error = probabilities[(row, class)] - target;
                    weight_gradient += features[(row, col)] * error;
                }
                weight_gradient /= rows;
                let updated = -learning_rate * weight_gradient;
                weights[(col, class)] =
                    cpu_soft_threshold(updated, learning_rate * alpha * l1_ratio);
            }
            bias[class] = -learning_rate * bias_gradient;
        }
        (weights, bias)
    }

    #[test]
    fn statistical_cuda_f64_fit_and_predict_launch_real_kernels() {
        let (train, train_labels, validation, validation_labels) = training_fixture();
        let fit = try_fit_linear_softmax_cuda(
            "logistic",
            "gpu:0",
            &train,
            &train_labels,
            Some(&validation),
            &validation_labels,
            0.01,
            0.0,
            0.05,
            20,
        )
        .expect("real CUDA f64 fit must launch or fail loudly");

        assert_eq!(fit.runtime_backend_kind, BackendKind::NativeCuda);
        assert!(fit.weights.iter().any(|weight| weight.abs() > 1e-12));

        let gpu = try_predict_linear_softmax_cuda(
            "logistic",
            "gpu:0",
            &validation,
            &fit.weights,
            &fit.bias,
        )
        .expect("real CUDA f64 prediction must launch or fail loudly");
        let cpu = cpu_probabilities(&validation, &fit.weights, &fit.bias);
        assert_eq!(gpu.dim(), cpu.dim());
        for (gpu_value, cpu_value) in gpu.iter().zip(cpu.iter()) {
            assert!(gpu_value.is_finite());
            assert!((gpu_value - cpu_value).abs() <= 1e-10);
        }
        for row in gpu.rows() {
            assert!((row.sum() - 1.0).abs() <= 1e-12);
        }
    }

    #[test]
    fn statistical_cuda_f64_elasticnet_uses_exact_proximal_zeroes() {
        let (train, train_labels, _, _) = training_fixture();
        let fit = try_fit_linear_softmax_cuda(
            "elasticnet",
            "gpu:0",
            &train,
            &train_labels,
            None,
            &[],
            100.0,
            1.0,
            0.1,
            1,
        )
        .expect("real CUDA f64 proximal fit must launch or fail loudly");

        assert!(fit.weights.iter().all(|weight| *weight == 0.0));
    }

    #[test]
    fn statistical_cuda_row_error_reuse_matches_one_cpu_epoch() {
        let (train, train_labels, _, _) = training_fixture();
        let alpha = 0.03;
        let l1_ratio = 0.4;
        let learning_rate = 0.05;
        let expected = cpu_one_epoch(&train, &train_labels, alpha, l1_ratio, learning_rate);
        let actual = try_fit_linear_softmax_cuda(
            "elasticnet",
            "gpu:0",
            &train,
            &train_labels,
            None,
            &[],
            alpha,
            l1_ratio,
            learning_rate,
            1,
        )
        .expect("real CUDA row-error reuse must launch or fail loudly");

        for (gpu, cpu) in actual.weights.iter().zip(expected.0.iter()) {
            assert!(
                (gpu - cpu).abs() <= 1e-10,
                "weight drift: gpu={gpu}, cpu={cpu}"
            );
        }
        for (gpu, cpu) in actual.bias.iter().zip(expected.1.iter()) {
            assert!(
                (gpu - cpu).abs() <= 1e-10,
                "bias drift: gpu={gpu}, cpu={cpu}"
            );
        }
    }
}
