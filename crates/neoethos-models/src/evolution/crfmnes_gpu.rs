use anyhow::{Context, Result, bail};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl_common::stream_id::StreamId;
use cudarc::driver::CudaContext;
use ndarray::Array2;
use std::sync::Arc;

use crate::cubecl_lifecycle::{
    CubeClResidencyScope, cubecl_cuda_client, cubecl_residency_scope, cuda_allocator_page_layouts,
    cuda_allocator_page_size_for_buffer,
};

const CLASS_COUNT: usize = 3;
const L2_WEIGHT: f32 = 1.0e-4;

// Retain the original actual-device oracle; production uses the tiled path.
#[cfg(test)]
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

/// Independent (row, candidate) work. Keep each row's feature/hidden reductions
/// and softmax/clamps in the original order; the scratch stores ln(prob), NOT
/// a negated or partially reduced loss.
#[cube(launch)]
fn candidate_row_log_probability_kernel(
    candidates: &Array<f32>,
    features: &Array<f32>,
    labels: &Array<i32>,
    log_probabilities: &mut Array<f32>,
    candidate_count: u32,
    row_start: u32,
    tile_rows: u32,
    input_dim: u32,
    hidden_dim: u32,
    param_dim: u32,
) {
    let count = candidate_count as usize;
    if ABSOLUTE_POS < count * tile_rows as usize {
        let candidate = ABSOLUTE_POS % count;
        let row = row_start as usize + ABSOLUTE_POS / count;
        let input_dim_us = input_dim as usize;
        let hidden_dim_us = hidden_dim as usize;
        let param_base = candidate * param_dim as usize;
        let w1_offset = param_base;
        let b1_offset = w1_offset + input_dim_us * hidden_dim_us;
        let w2_offset = b1_offset + hidden_dim_us;
        let b2_offset = w2_offset + hidden_dim_us * CLASS_COUNT;
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
            logit1.store(logit1.read() + act * candidates[w2_offset + hidden * CLASS_COUNT + 1]);
            logit2.store(logit2.read() + act * candidates[w2_offset + hidden * CLASS_COUNT + 2]);
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
        log_probabilities[ABSOLUTE_POS] = probability.read().ln();
    }
}

/// Same ordered f32 subtraction across ALL rows, including tile boundaries.
/// The three candidate-sized planes hold train, validation, and cached L2.
#[cube(launch)]
fn candidate_ordered_loss_kernel(
    candidates: &Array<f32>,
    log_probabilities: &Array<f32>,
    accumulators: &mut Array<f32>,
    candidate_count: u32,
    param_dim: u32,
    row_start: u32,
    tile_rows: u32,
    n_rows: u32,
    dataset_slot: u32,
) {
    if ABSOLUTE_POS < candidate_count as usize {
        let candidate = ABSOLUTE_POS;
        let count = candidate_count as usize;
        let output_index = dataset_slot as usize * count + candidate;
        if dataset_slot == 0 && row_start == 0 {
            accumulators[count + candidate] = 0.0;
            let l2 = RuntimeCell::<f32>::new(0.0);
            let param_base = candidate * param_dim as usize;
            for p in 0..param_dim as usize {
                let value = candidates[param_base + p];
                l2.store(l2.read() + value * value);
            }
            accumulators[2 * count + candidate] = l2.read() / param_dim as f32;
        }
        let total_loss = RuntimeCell::<f32>::new(0.0);
        if row_start != 0 {
            total_loss.store(accumulators[output_index]);
        }
        for row in 0..tile_rows as usize {
            total_loss.store(total_loss.read() - log_probabilities[row * count + candidate]);
        }
        if row_start + tile_rows == n_rows {
            let l2_final = accumulators[2 * count + candidate];
            if n_rows == 0 {
                accumulators[output_index] = L2_WEIGHT * l2_final;
            } else {
                accumulators[output_index] =
                    total_loss.read() / n_rows as f32 + L2_WEIGHT * l2_final;
            }
        } else {
            accumulators[output_index] = total_loss.read();
        }
    }
}

fn cuda_device_id(policy: &str) -> Result<usize> {
    crate::common::cuda_device_id_from_policy(policy)
}

fn kernel_units(client: &ComputeClient<CudaRuntime>) -> u32 {
    crate::common::cuda_kernel_units(client.properties().hardware.max_units_per_cube)
}

fn flatten_candidates(candidates: &[Vec<f64>], param_dim: usize) -> Result<Vec<f32>> {
    let elements = candidates
        .len()
        .checked_mul(param_dim)
        .context("neuro-evo CUDA candidate element count overflow")?;
    let mut flat = Vec::new();
    flat.try_reserve_exact(elements)
        .context("reserve neuro-evo CUDA candidate staging")?;
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

fn f32_bytes(elements: usize) -> Result<usize> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .context("neuro-evo CUDA buffer byte count overflow")
}

fn pinned_transfer_page(bytes: usize, max_page_size: usize) -> Result<usize> {
    // Pinned CubeCL CUDA storage uses size_of::<u128>() alignment and the
    // device's SAME SubSlices page configuration (compute/stream.rs).
    cuda_allocator_page_size_for_buffer(bytes, max_page_size, std::mem::size_of::<u128>())
}

fn candidate_host_bytes(
    count: usize,
    bytes_per_candidate: usize,
    max_page_size: usize,
) -> Result<usize> {
    // Ordinary uploads are pageable, but readback and metadata can use pinned
    // pages. Conservatively reserve an upload-sized page as well; do not model
    // all host transfer allocation as only a multiple of logical payload bytes.
    let candidate_bytes = count
        .checked_mul(bytes_per_candidate)
        .context("candidate host bytes overflow")?;
    let loss_bytes = f32_bytes(count)?;
    candidate_bytes
        .checked_add(
            loss_bytes
                .checked_mul(3)
                .context("loss host bytes overflow")?,
        )
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_add(pinned_transfer_page(candidate_bytes, max_page_size).ok()?))
        .and_then(|n| {
            n.checked_add(
                pinned_transfer_page(loss_bytes.checked_mul(3)?, max_page_size)
                    .ok()?
                    .checked_add(pinned_transfer_page(4, max_page_size).ok()?)?,
            )
        })
        .context("neuro-evo CUDA pinned/pageable staging exceeds representable allocator limits")
}

/// Select only a concurrent evaluation width, never a population/model cap.
/// Fresh-page rounding matches the existing statistical CUDA preflight.
fn candidate_batch_width(
    remaining: usize,
    param_dim: usize,
    resident_buffer_bytes: &[usize],
    usable_device_bytes: usize,
    max_page_size: usize,
    alignment: usize,
    host_staging_budget: usize,
) -> Result<usize> {
    if remaining == 0 {
        return Ok(0);
    }
    if param_dim == 0 || alignment == 0 || max_page_size == 0 {
        bail!("neuro-evo CUDA batching requires nonzero dimensions and allocator limits");
    }
    let bytes_per_candidate = f32_bytes(param_dim)?;
    let fixed_pages = resident_buffer_bytes
        .iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(cuda_allocator_page_size_for_buffer(
                    *bytes,
                    max_page_size,
                    alignment,
                )?)
                .context("neuro-evo CUDA resident page budget overflow")
        })?;
    // create_from_slice owns a byte copy plus transfer staging; loss readback
    // and the two scalar loss vectors are bounded by the same selected width.
    let host_bytes_per_candidate = bytes_per_candidate
        .checked_add(3 * std::mem::size_of::<f32>())
        .and_then(|bytes| bytes.checked_mul(3))
        .context("neuro-evo CUDA host staging budget overflow")?;
    let upper = remaining
        // CubeCL tensor indexing is u32 on this kernel route, including the
        // candidate * parameter product, not just each individual dimension.
        .min(u32::MAX as usize / param_dim)
        .min(u32::MAX as usize / 3)
        .min((max_page_size / alignment * alignment) / bytes_per_candidate)
        .min((max_page_size / alignment * alignment) / (3 * std::mem::size_of::<f32>()))
        .min(host_staging_budget / host_bytes_per_candidate);
    let fits = |count: usize| -> Result<bool> {
        let candidate_bytes = count
            .checked_mul(bytes_per_candidate)
            .context("neuro-evo CUDA batch byte count overflow")?;
        let candidate_page =
            cuda_allocator_page_size_for_buffer(candidate_bytes, max_page_size, alignment)?;
        let accumulator_page = cuda_allocator_page_size_for_buffer(
            f32_bytes(
                count
                    .checked_mul(3)
                    .context("CUDA accumulator shape overflow")?,
            )?,
            max_page_size,
            alignment,
        )?;
        // Joint admission reserves at least one full row of scratch BEFORE
        // maximizing candidates; the candidate batch cannot consume that space.
        let minimum_scratch_page =
            cuda_allocator_page_size_for_buffer(f32_bytes(count)?, max_page_size, alignment)?;
        Ok(
            candidate_host_bytes(count, bytes_per_candidate, max_page_size)? <= host_staging_budget
                && fixed_pages
                    .checked_add(candidate_page)
                    .and_then(|bytes| bytes.checked_add(accumulator_page))
                    .and_then(|bytes| bytes.checked_add(minimum_scratch_page))
                    .is_some_and(|bytes| bytes <= usable_device_bytes),
        )
    };
    if upper == 0 || !fits(1)? {
        bail!(
            "neuro-evo CUDA cannot admit one complete candidate with current device/host headroom; population and model dimensions were not reduced"
        );
    }
    // Page choice is NOT monotone: SubSlices puts allocations close to a
    // page's full size back into that smaller page. Include both sides of its
    // strict >80% transition before searching constant-page intervals.
    let mut choices = vec![upper, 0];
    for allocator_alignment in [alignment, std::mem::size_of::<u128>()] {
        for (page, slice) in cuda_allocator_page_layouts(max_page_size, allocator_alignment)? {
            let close_threshold = (page / 5) * 4 + (page % 5) * 4 / 5;
            for endpoint in [page, slice, close_threshold] {
                choices.push((endpoint / bytes_per_candidate).min(upper));
                choices.push((endpoint / std::mem::size_of::<f32>()).min(upper));
                choices.push((endpoint / (3 * std::mem::size_of::<f32>())).min(upper));
            }
        }
    }
    choices.sort_unstable();
    choices.dedup();
    for interval in choices.windows(2).rev() {
        let mut low = interval[0] + 1;
        let mut high = interval[1];
        if !fits(low)? {
            continue;
        }
        // Within one constant-page interval only pageable bytes increase.
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if fits(mid)? {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return Ok(low);
    }
    unreachable!("one candidate was admitted above")
}

/// Largest reusable scratch tile after joint candidate admission. No all-row
/// candidate matrix is assumed: both allocator pages and u32 products bind it.
fn row_tile_capacity(
    candidates: usize,
    param_dim: usize,
    max_rows: usize,
    device_budget: usize,
    max_page_size: usize,
    alignment: usize,
) -> Result<usize> {
    if candidates == 0 {
        bail!("CUDA row tiling requires a nonempty candidate batch");
    }
    let candidate_bytes = f32_bytes(
        candidates
            .checked_mul(param_dim)
            .context("CUDA candidate shape overflow")?,
    )?;
    let accumulator_bytes = f32_bytes(
        candidates
            .checked_mul(3)
            .context("CUDA accumulator shape overflow")?,
    )?;
    let fixed = cuda_allocator_page_size_for_buffer(candidate_bytes, max_page_size, alignment)?
        .checked_add(cuda_allocator_page_size_for_buffer(
            accumulator_bytes,
            max_page_size,
            alignment,
        )?)
        .context("CUDA tile fixed-page sum overflow")?;
    let available = device_budget
        .checked_sub(fixed)
        .context("CUDA row scratch has no admitted headroom")?;
    let bytes_per_row = f32_bytes(candidates)?;
    let upper = max_rows
        .max(1)
        .min(u32::MAX as usize / candidates)
        .min(max_page_size / bytes_per_row);
    let mut choices = vec![upper];
    for (page, slice) in cuda_allocator_page_layouts(max_page_size, alignment)? {
        choices.push((page / bytes_per_row).min(upper));
        choices.push((slice / bytes_per_row).min(upper));
    }
    choices.sort_unstable();
    choices.dedup();
    for rows in choices.into_iter().rev().filter(|rows| *rows > 0) {
        let bytes = rows
            .checked_mul(bytes_per_row)
            .context("CUDA scratch byte count overflow")?;
        if cuda_allocator_page_size_for_buffer(bytes, max_page_size, alignment)? <= available {
            return Ok(rows);
        }
    }
    bail!("neuro-evo CUDA cannot admit one ordered row tile; no rows or candidates were removed")
}

fn usable_device_bytes(context: &CudaContext) -> Result<usize> {
    let (free, total) = context
        .mem_get_info()
        .context("inspect selected neuro-evo CUDA device free memory")?;
    if total == 0 {
        bail!("neuro-evo CUDA device reports zero total memory");
    }
    // Same reserve policy as the existing statistical CUDA preflight. A
    // snapshot is admission evidence, not a reservation against other jobs.
    let headroom = (total / 10).max(256 * 1024 * 1024).min(total);
    Ok(free.min(total).saturating_sub(headroom))
}

fn host_staging_budget() -> usize {
    usize::try_from(neoethos_core::available_memory_bytes() / 2).unwrap_or(usize::MAX)
}

struct ResidentLossInputs {
    features: cubecl::server::Handle,
    labels: cubecl::server::Handle,
    feature_count: usize,
    rows: u32,
}

#[cfg(test)]
thread_local! {
    static DATASET_UPLOAD_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn dataset_upload_calls() -> usize {
    DATASET_UPLOAD_CALLS.get()
}

fn upload_loss_inputs(
    client: &ComputeClient<CudaRuntime>,
    features: &Array2<f32>,
    labels: &[usize],
    input_dim: usize,
) -> Result<ResidentLossInputs> {
    let features_flat = flatten_features(features, input_dim)?;
    let labels_flat = labels.iter().map(|label| *label as i32).collect::<Vec<_>>();
    let features_handle = client.create_from_slice(f32::as_bytes(&features_flat));
    let labels_handle = client.create_from_slice(i32::as_bytes(&labels_flat));
    #[cfg(test)]
    DATASET_UPLOAD_CALLS.set(DATASET_UPLOAD_CALLS.get() + 1);
    Ok(ResidentLossInputs {
        features: features_handle,
        labels: labels_handle,
        feature_count: features_flat.len(),
        rows: u32::try_from(features.nrows()).context("neuro-evo CUDA rows exceed u32")?,
    })
}

#[cfg(test)]
fn launch_loss_kernel(
    client: &ComputeClient<CudaRuntime>,
    candidates_handle: &cubecl::server::Handle,
    candidates_len: usize,
    inputs: &ResidentLossInputs,
    candidate_count: usize,
    input_dim: usize,
    hidden_dim: usize,
    param_dim: usize,
) -> Result<Vec<f32>> {
    if candidate_count == 0 {
        return Ok(Vec::new());
    }
    let losses_handle = client.empty(f32_bytes(candidate_count)?);

    let units = kernel_units(client);
    let cubes = (candidate_count as u32).div_ceil(units);
    candidate_loss_kernel::launch::<CudaRuntime>(
        client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts(candidates_handle.clone(), candidates_len) },
        unsafe { ArrayArg::from_raw_parts(inputs.features.clone(), inputs.feature_count) },
        unsafe { ArrayArg::from_raw_parts(inputs.labels.clone(), inputs.rows as usize) },
        unsafe { ArrayArg::from_raw_parts(losses_handle.clone(), candidate_count) },
        inputs.rows,
        input_dim as u32,
        hidden_dim as u32,
        param_dim as u32,
    );

    // See `neat_gpu.rs`: `kernel::launch::<R>` is `()` in cubecl 0.10, and the
    // `Result` moved to the readback.
    let bytes = client
        .read_one(losses_handle)
        .context("read back neuro-evo cuda candidate losses")?;
    let losses = f32::from_bytes(&bytes);
    if losses.len() != candidate_count {
        bail!(
            "neuro-evo CUDA loss count mismatch: expected {candidate_count}, received {}",
            losses.len()
        );
    }
    Ok(losses.to_vec())
}

/// One optimizer run owns its exact train/validation buffers and CUDA stream.
/// Evolutionary ask/tell and candidate generation remain host-side; only the
/// fitness datasets persist here. Candidate buffers and loss readbacks are
/// still bounded per batch, with current memory admission on every callback.
pub(crate) struct CrfmnesCudaSession {
    train: ResidentLossInputs,
    validation: Option<ResidentLossInputs>,
    client: ComputeClient<CudaRuntime>,
    context: Arc<CudaContext>,
    stream: StreamId,
    input_dim: usize,
    hidden_dim: usize,
    param_dim: usize,
    max_page_size: usize,
    alignment: usize,
    // Struct fields drop in declaration order: release all handles before
    // the scope can synchronize and clean their exact allocator pools.
    _residency: CubeClResidencyScope,
}

impl CrfmnesCudaSession {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        train_features: &Array2<f32>,
        train_labels: &[usize],
        val_features: &Array2<f32>,
        val_labels: &[usize],
        input_dim: usize,
        hidden_dim: usize,
        param_dim: usize,
        policy: &str,
    ) -> Result<Self> {
        for (dimension, label) in [
            (input_dim, "input"),
            (hidden_dim, "hidden"),
            (param_dim, "parameter"),
        ] {
            u32::try_from(dimension)
                .with_context(|| format!("neuro-evo CUDA {label} dimension exceeds u32"))?;
        }
        let expected_params = input_dim
            .checked_mul(hidden_dim)
            .and_then(|count| count.checked_add(hidden_dim))
            .and_then(|count| {
                hidden_dim
                    .checked_mul(CLASS_COUNT)
                    .and_then(|output| count.checked_add(output))
            })
            .and_then(|count| count.checked_add(CLASS_COUNT))
            .context("neuro-evo CUDA model parameter count overflow")?;
        if param_dim != expected_params {
            bail!("neuro-evo CUDA candidate/model dimension mismatch");
        }
        let mut resident_bytes = Vec::new();
        for (index, (features, labels)) in
            [(train_features, train_labels), (val_features, val_labels)]
                .into_iter()
                .enumerate()
        {
            if index == 1 && val_labels.is_empty() {
                continue;
            }
            u32::try_from(features.nrows()).context("neuro-evo CUDA row count exceeds u32")?;
            u32::try_from(features.len())
                .context("neuro-evo CUDA feature tensor exceeds u32 indexing")?;
            if features.ncols() != input_dim || features.nrows() != labels.len() {
                bail!("neuro-evo CUDA feature/label dimensions do not match the model");
            }
            if labels.iter().any(|label| *label >= CLASS_COUNT) {
                bail!("neuro-evo cuda labels must be in 0..3");
            }
            resident_bytes.push(f32_bytes(features.len())?);
            resident_bytes.push(f32_bytes(labels.len())?);
        }
        let _cubecl_call_residency = cubecl_residency_scope();
        let ordinal = cuda_device_id(policy)?;
        let stream = StreamId::current();
        let client = cubecl_cuda_client(ordinal);
        let context =
            CudaContext::new(ordinal).context("retain neuro-evo CUDA memory-query context")?;
        let max_page_size = usize::try_from(client.properties().memory.max_page_size)
            .context("neuro-evo CUDA allocator page limit exceeds usize")?;
        let alignment = usize::try_from(client.properties().memory.alignment)
            .context("neuro-evo CUDA allocator alignment exceeds usize")?;
        let upload_bytes = resident_bytes.iter().try_fold(0usize, |total, bytes| {
            total
                .checked_add(
                    bytes
                        .checked_mul(3)
                        .context("dataset pageable staging overflow")?,
                )
                .and_then(|n| n.checked_add(pinned_transfer_page(*bytes, max_page_size).ok()?))
                .context("neuro-evo CUDA dataset byte count overflow")
        })?;
        let admit_upload = || {
            let initial_host_budget = host_staging_budget()
                .checked_sub(std::mem::size_of::<(f64, f64, f64)>())
                .and_then(|bytes| bytes.checked_sub(upload_bytes))
                .context("neuro-evo CUDA datasets/results exceed current host staging headroom")?;
            candidate_batch_width(
                1,
                param_dim,
                &resident_bytes,
                usable_device_bytes(&context)?,
                max_page_size,
                alignment,
                initial_host_budget,
            )
        };
        if admit_upload().is_err() {
            // A surrounding model scope can retain idle pages from a prior model.
            // Reclaim them before a single fresh retry; do not discard live data.
            client.memory_cleanup();
            cubecl::future::block_on(client.sync())
                .context("release idle neuro-evo CUDA pages before dataset admission")?;
            admit_upload()?;
        }
        let train = upload_loss_inputs(&client, train_features, train_labels, input_dim)?;
        let validation = if val_labels.is_empty() {
            None
        } else {
            Some(upload_loss_inputs(
                &client,
                val_features,
                val_labels,
                input_dim,
            )?)
        };
        cubecl::future::block_on(client.sync())
            .context("finish resident neuro-evo dataset uploads")?;
        let dataset_upload_bytes = resident_bytes.iter().try_fold(0usize, |sum, bytes| {
            sum.checked_add(*bytes)
                .context("CUDA dataset upload counter overflow")
        })?;
        tracing::info!(target: "neoethos_models::crfmnes", train_rows=train.rows,
            validation_rows=validation.as_ref().map_or(0, |data| data.rows),
            dataset_upload_bytes, "CR-FM-NES run-owned datasets uploaded once");
        Ok(Self {
            train,
            validation,
            client,
            context,
            stream,
            input_dim,
            hidden_dim,
            param_dim,
            max_page_size,
            alignment,
            _residency: _cubecl_call_residency,
        })
    }

    pub(crate) fn selection_losses(
        &mut self,
        candidates: &[Vec<f64>],
    ) -> Result<Vec<(f64, f64, f64)>> {
        if StreamId::current() != self.stream {
            bail!("neuro-evo CUDA session cannot change its allocating stream");
        }
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        if candidates
            .iter()
            .any(|candidate| candidate.len() != self.param_dim)
        {
            bail!("neuro-evo CUDA candidate/model dimension mismatch");
        }
        let result_bytes = candidates
            .len()
            .checked_mul(std::mem::size_of::<(f64, f64, f64)>())
            .context("neuro-evo CUDA result byte count overflow")?;
        // Keep result storage and at least one candidate batch inside the current
        // host headroom before reserving. The session already owns dataset pages.
        let minimum_batch_bytes =
            candidate_host_bytes(1, f32_bytes(self.param_dim)?, self.max_page_size)?;
        let client = &self.client;
        let admit_results = || -> Result<()> {
            let initial_host_budget = host_staging_budget()
                .checked_sub(result_bytes)
                .context("neuro-evo CUDA results exceed current host staging headroom")?;
            if minimum_batch_bytes > initial_host_budget {
                bail!(
                    "neuro-evo CUDA results and one candidate exceed current host staging headroom"
                );
            }
            Ok(())
        };
        if admit_results().is_err() {
            client.memory_cleanup();
            cubecl::future::block_on(client.sync())
                .context("release idle neuro-evo host pages before result admission")?;
            admit_results()?;
        }
        let mut results = Vec::new();
        results
            .try_reserve_exact(candidates.len())
            .context("reserve neuro-evo CUDA results")?;
        let mut offset = 0;
        let mut total_tiles = 0_u64;
        let mut total_row_work_items = 0_u64;
        let mut candidate_upload_bytes = 0_u64;
        let mut final_readback_bytes = 0_u64;
        let mut candidate_batches = 0_u64;
        let mut row_kernel_launches = 0_u64;
        let mut max_row_tile_rows = 0_usize;
        let mut max_scratch_bytes = 0_usize;
        while offset < candidates.len() {
            let choose_width = || {
                let device_budget = usable_device_bytes(&self.context)?;
                let width = candidate_batch_width(
                    candidates.len() - offset,
                    self.param_dim,
                    &[],
                    device_budget,
                    self.max_page_size,
                    self.alignment,
                    host_staging_budget()
                        .checked_sub(result_bytes)
                        .context("neuro-evo CUDA results exceed current host staging headroom")?,
                )?;
                // Rounded launch extent must fit u32 as well as logical cells.
                let units = kernel_units(client) as usize;
                let max_grid_items = u32::MAX as usize / units * units;
                let max_rows = (self.train.rows as usize)
                    .max(
                        self.validation
                            .as_ref()
                            .map_or(0, |data| data.rows as usize),
                    )
                    .min(max_grid_items / width);
                let rows = row_tile_capacity(
                    width,
                    self.param_dim,
                    max_rows,
                    device_budget,
                    self.max_page_size,
                    self.alignment,
                )?;
                Ok::<_, anyhow::Error>((width, rows))
            };
            let (width, tile_capacity) = match choose_width() {
                Ok(width) => width,
                Err(_) => {
                    // Finished batch handles are gone. Release reusable idle pages
                    // only if fresh-allocation admission fails; live dataset handles
                    // remain resident. Never count another stream's pool as free.
                    client.memory_cleanup();
                    cubecl::future::block_on(client.sync())
                        .context("release idle neuro-evo CUDA batch pages")?;
                    choose_width()?
                }
            };
            let candidates_flat =
                flatten_candidates(&candidates[offset..offset + width], self.param_dim)?;
            let candidates_handle = client.create_from_slice(f32::as_bytes(&candidates_flat));
            let scratch_elements = width
                .checked_mul(tile_capacity)
                .context("CUDA row tile shape overflow")?;
            let scratch = client.empty(f32_bytes(scratch_elements)?);
            let accumulator_elements = width
                .checked_mul(3)
                .context("CUDA accumulator shape overflow")?;
            let accumulators = client.empty(f32_bytes(accumulator_elements)?);
            let (tiles, work_items) = self.launch_dataset_tiles(
                &candidates_handle,
                candidates_flat.len(),
                &self.train,
                &scratch,
                scratch_elements,
                &accumulators,
                width,
                tile_capacity,
                0,
            )?;
            total_tiles = total_tiles
                .checked_add(tiles)
                .context("CUDA tile counter overflow")?;
            total_row_work_items = total_row_work_items
                .checked_add(work_items)
                .context("CUDA row counter overflow")?;
            if self.train.rows > 0 {
                row_kernel_launches += tiles;
            }
            if let Some(validation) = &self.validation {
                let (tiles, work_items) = self.launch_dataset_tiles(
                    &candidates_handle,
                    candidates_flat.len(),
                    validation,
                    &scratch,
                    scratch_elements,
                    &accumulators,
                    width,
                    tile_capacity,
                    1,
                )?;
                total_tiles = total_tiles
                    .checked_add(tiles)
                    .context("CUDA tile counter overflow")?;
                total_row_work_items = total_row_work_items
                    .checked_add(work_items)
                    .context("CUDA row counter overflow")?;
                if validation.rows > 0 {
                    row_kernel_launches += tiles;
                }
            }
            // No per-row/tile transfer or host reduction. This read synchronizes
            // both ordered dataset evaluations and surfaces asynchronous errors.
            let bytes = client
                .read_one(accumulators)
                .context("read final tiled neuro-evo CUDA losses")?;
            let losses = f32::from_bytes(&bytes);
            if losses.len() != accumulator_elements {
                bail!("CUDA tiled accumulator readback length mismatch");
            }
            for candidate in 0..width {
                let train_loss = losses[candidate] as f64;
                let val_loss = if self.validation.is_some() {
                    losses[width + candidate] as f64
                } else {
                    train_loss
                };
                let selection_loss = if self.validation.is_none() {
                    train_loss
                } else {
                    0.65 * train_loss + 0.35 * val_loss
                };
                results.push((selection_loss, train_loss, val_loss));
            }
            candidate_upload_bytes = candidate_upload_bytes
                .checked_add(f32_bytes(candidates_flat.len())? as u64)
                .context("CUDA upload counter overflow")?;
            final_readback_bytes = final_readback_bytes
                .checked_add(f32_bytes(accumulator_elements)? as u64)
                .context("CUDA readback counter overflow")?;
            max_row_tile_rows = max_row_tile_rows.max(tile_capacity);
            max_scratch_bytes = max_scratch_bytes.max(f32_bytes(scratch_elements)?);
            candidate_batches += 1;
            offset += width;
        }
        tracing::info!(target: "neoethos_models::crfmnes", candidates=candidates.len(),
            candidate_batches, ordered_row_tiles=total_tiles, row_kernel_launches,
            row_work_items=total_row_work_items, max_row_tile_rows, max_scratch_bytes,
            candidate_upload_bytes, final_readback_bytes, dataset_upload_bytes=0_u64,
            "CR-FM-NES resident row-parallel evaluation completed");
        Ok(results)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_dataset_tiles(
        &self,
        candidates: &cubecl::server::Handle,
        candidates_len: usize,
        inputs: &ResidentLossInputs,
        scratch: &cubecl::server::Handle,
        scratch_len: usize,
        accumulators: &cubecl::server::Handle,
        count: usize,
        tile_capacity: usize,
        dataset_slot: u32,
    ) -> Result<(u64, u64)> {
        let units = kernel_units(&self.client);
        let mut row_start = 0_usize;
        let mut tiles = 0_u64;
        loop {
            let tile_rows = (inputs.rows as usize - row_start).min(tile_capacity);
            let work_items = count
                .checked_mul(tile_rows)
                .context("CUDA row work item count overflow")?;
            if work_items > 0 {
                let cubes = u32::try_from(work_items)?.div_ceil(units);
                cubes
                    .checked_mul(units)
                    .context("CUDA rounded row launch exceeds u32")?;
                candidate_row_log_probability_kernel::launch::<CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(units),
                    unsafe { ArrayArg::from_raw_parts(candidates.clone(), candidates_len) },
                    unsafe {
                        ArrayArg::from_raw_parts(inputs.features.clone(), inputs.feature_count)
                    },
                    unsafe {
                        ArrayArg::from_raw_parts(inputs.labels.clone(), inputs.rows as usize)
                    },
                    unsafe { ArrayArg::from_raw_parts(scratch.clone(), scratch_len) },
                    count as u32,
                    row_start as u32,
                    tile_rows as u32,
                    self.input_dim as u32,
                    self.hidden_dim as u32,
                    self.param_dim as u32,
                );
            }
            candidate_ordered_loss_kernel::launch::<CudaRuntime>(
                &self.client,
                CubeCount::Static((count as u32).div_ceil(units), 1, 1),
                CubeDim::new_1d(units),
                unsafe { ArrayArg::from_raw_parts(candidates.clone(), candidates_len) },
                unsafe { ArrayArg::from_raw_parts(scratch.clone(), scratch_len) },
                unsafe { ArrayArg::from_raw_parts(accumulators.clone(), count * 3) },
                count as u32,
                self.param_dim as u32,
                row_start as u32,
                tile_rows as u32,
                inputs.rows,
                dataset_slot,
            );
            tiles += 1;
            row_start += tile_rows;
            if row_start == inputs.rows as usize {
                break;
            }
        }
        Ok((tiles, count as u64 * inputs.rows as u64))
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn try_selection_losses_cuda(
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
    CrfmnesCudaSession::new(
        train_features,
        train_labels,
        val_features,
        val_labels,
        input_dim,
        hidden_dim,
        param_dim,
        policy,
    )?
    .selection_losses(candidates)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{
        CrfmnesCudaSession, candidate_batch_width, candidate_host_bytes, dataset_upload_calls,
        flatten_candidates, row_tile_capacity, try_selection_losses_cuda,
    };

    #[test]
    fn crfmnes_cuda_row_tiles_jointly_admit_scratch_and_preserve_all_rows() {
        const MIB: usize = 1024 * 1024;
        // A formerly sufficient candidate+loss budget does not include scratch.
        assert!(
            candidate_batch_width(1, 1024, &[], 16 * MIB, 8 * 1024 * MIB, 512, 512 * MIB).is_err()
        );
        for budget in [24 * MIB, 64 * MIB, 256 * MIB] {
            let count =
                candidate_batch_width(2048, 1024, &[], budget, 8 * 1024 * MIB, 512, 512 * MIB)
                    .unwrap();
            let rows =
                row_tile_capacity(count, 1024, 100_003, budget, 8 * 1024 * MIB, 512).unwrap();
            assert!(rows > 0 && rows <= 100_003);
            assert!(count * rows <= u32::MAX as usize);
            let mut next = 0;
            let mut observed = 0;
            while next < 100_003 {
                let length = rows.min(100_003 - next);
                assert_eq!(next, observed);
                observed += length;
                next += length;
            }
            assert_eq!(observed, 100_003);
        }
    }

    #[test]
    fn crfmnes_cuda_row_tiles_match_exhaustive_page_admission() {
        const MIB: usize = 1024 * 1024;
        for budget in [24 * MIB, 40 * MIB, 160 * MIB] {
            for host_budget in [48 * MIB, 512 * MIB] {
                let expected = (1..=2048)
                    .filter(|count| {
                        let gpu = [count * 4096, count * 12, count * 4]
                            .into_iter()
                            .map(|bytes| {
                                super::cuda_allocator_page_size_for_buffer(
                                    bytes,
                                    8 * 1024 * MIB,
                                    512,
                                )
                                .unwrap()
                            })
                            .sum::<usize>();
                        gpu <= budget
                            && candidate_host_bytes(*count, 4096, 8 * 1024 * MIB).unwrap()
                                <= host_budget
                    })
                    .max();
                let actual = candidate_batch_width(
                    2048,
                    1024,
                    &[],
                    budget,
                    8 * 1024 * MIB,
                    512,
                    host_budget,
                );
                assert_eq!(actual.ok(), expected);
            }
        }
        assert!(row_tile_capacity(0, 1024, 10, MIB, MIB, 512).is_err());
        assert!(row_tile_capacity(1, usize::MAX, 10, MIB, MIB, 512).is_err());
        let rows = row_tile_capacity(3, 32, usize::MAX, usize::MAX, 32 * 1024 * MIB, 512).unwrap();
        assert!(rows * 3 <= u32::MAX as usize);
    }

    #[test]
    fn crfmnes_cuda_batching_scales_with_free_memory_not_population_caps() {
        const MIB: usize = 1024 * 1024;
        let small = candidate_batch_width(4097, MIB / 4, &[], 32 * MIB, 512 * MIB, 512, usize::MAX)
            .unwrap();
        let large =
            candidate_batch_width(4097, MIB / 4, &[], 1024 * MIB, 512 * MIB, 512, usize::MAX)
                .unwrap();
        assert!(
            large > small,
            "more available VRAM must admit a wider batch"
        );
        assert_eq!(
            large, 512,
            "the allocator's per-buffer limit remains binding"
        );
        assert_eq!(
            candidate_batch_width(17, 1024, &[], 1024 * MIB, 512 * MIB, 512, usize::MAX).unwrap(),
            17
        );
        assert_eq!(
            candidate_batch_width(2048, 1024, &[], 24 * MIB, 8 * 1024 * MIB, 512, 512 * MIB)
                .unwrap(),
            2048,
            "a later close-to-page interval can fit after a wider fresh page fails"
        );
    }

    #[test]
    fn crfmnes_cuda_batching_preserves_all_candidates_across_changing_headroom() {
        const MIB: usize = 1024 * 1024;
        let candidates: Vec<Vec<f64>> = (0..43).map(|index| vec![index as f64; 1024]).collect();
        let mut rebuilt = Vec::new();
        let mut offset = 0;
        while offset < candidates.len() {
            // Exercise host-limited widths 1, 7, 13 and an uneven last batch.
            let requested = [1, 7, 13][rebuilt.len() % 3];
            let host_bytes = candidate_host_bytes(requested, 1024 * 4, 512 * MIB).unwrap();
            let width = candidate_batch_width(
                candidates.len() - offset,
                1024,
                &[],
                1024 * MIB,
                512 * MIB,
                512,
                host_bytes,
            )
            .unwrap();
            let flat = flatten_candidates(&candidates[offset..offset + width], 1024).unwrap();
            rebuilt.extend(flat.chunks_exact(1024).map(|values| values[0] as usize));
            offset += width;
        }
        assert_eq!(rebuilt, (0..43).collect::<Vec<_>>());
    }

    #[test]
    fn crfmnes_cuda_batching_rejects_insufficient_memory_and_overflow() {
        const MIB: usize = 1024 * 1024;
        assert!(candidate_batch_width(1, 1024, &[], 0, 512 * MIB, 512, usize::MAX).is_err());
        assert!(
            candidate_batch_width(
                1,
                1024,
                &[513 * MIB],
                usize::MAX,
                512 * MIB,
                512,
                usize::MAX
            )
            .is_err()
        );
        assert!(
            candidate_batch_width(1, usize::MAX, &[], usize::MAX, usize::MAX, 512, usize::MAX)
                .is_err()
        );
        assert!(candidate_batch_width(1, 1024, &[], usize::MAX, 512 * MIB, 0, usize::MAX).is_err());
        assert!(candidate_batch_width(1, 1024, &[], usize::MAX, 512 * MIB, 512, 0).is_err());
        assert_eq!(candidate_batch_width(0, 0, &[], 0, 0, 0, 0).unwrap(), 0);
        assert!(flatten_candidates(&[vec![0.0; 2]], 3).is_err());
        // A tiny payload still needs full pinned allocator pages.
        assert!(candidate_host_bytes(1, 4096, 8 * 1024 * MIB).unwrap() >= 24 * MIB);
        assert!(
            candidate_batch_width(1, 1024, &[], usize::MAX, 8 * 1024 * MIB, 512, 16 * MIB).is_err()
        );
        let width = candidate_batch_width(
            usize::MAX,
            1024,
            &[],
            usize::MAX,
            32 * 1024 * MIB,
            512,
            usize::MAX,
        )
        .unwrap();
        assert_eq!(width, u32::MAX as usize / 1024);
        assert!(width * 1024 <= u32::MAX as usize);
    }

    #[test]
    fn crfmnes_cuda_selection_losses_launch_real_kernel() {
        let input_dim = 2;
        let hidden_dim = 3;
        let param_dim = input_dim * hidden_dim + hidden_dim + hidden_dim * 3 + 3;
        let candidates = vec![
            vec![0.0_f64; param_dim],
            (0..param_dim)
                .map(|index| (index as f64 - 5.0) * 0.01)
                .collect(),
        ];
        let train_features = array![
            [0.25_f32, -0.5_f32],
            [0.75_f32, 0.125_f32],
            [-0.4_f32, 0.9_f32],
            [0.6_f32, -0.2_f32],
        ];
        let train_labels = [0_usize, 1, 2, 1];
        let val_features = array![[0.1_f32, 0.2_f32], [-0.3_f32, 0.8_f32]];
        let val_labels = [0_usize, 2];

        let losses = try_selection_losses_cuda(
            &candidates,
            &train_features,
            &train_labels,
            &val_features,
            &val_labels,
            input_dim,
            hidden_dim,
            param_dim,
            "gpu:0",
        )
        .expect("mandatory CR-FM-NES CUDA kernel launch");

        assert_eq!(losses.len(), candidates.len());
        assert!(losses.iter().all(|losses| {
            losses.0.is_finite() && losses.1.is_finite() && losses.2.is_finite()
        }));
        // Each candidate's reductions are unchanged by the number of peers in
        // its launch. Exercise the actual producer with single-candidate calls,
        // not a CPU approximation of the kernel.
        for (candidate, expected) in candidates.iter().zip(losses) {
            let one = try_selection_losses_cuda(
                std::slice::from_ref(candidate),
                &train_features,
                &train_labels,
                &val_features,
                &val_labels,
                input_dim,
                hidden_dim,
                param_dim,
                "gpu:0",
            )
            .expect("mandatory single-candidate CR-FM-NES CUDA launch");
            assert_eq!(one.len(), 1);
            assert_eq!(
                [one[0].0.to_bits(), one[0].1.to_bits(), one[0].2.to_bits()],
                [
                    expected.0.to_bits(),
                    expected.1.to_bits(),
                    expected.2.to_bits()
                ]
            );
        }
    }

    #[test]
    fn crfmnes_cuda_session_uploads_datasets_once_across_populations() {
        let (input_dim, hidden_dim) = (2, 3);
        let param_dim = input_dim * hidden_dim + hidden_dim + hidden_dim * 3 + 3;
        let train_features = array![[0.25_f32, -0.5], [0.75, 0.125], [-0.4, 0.9], [0.6, -0.2]];
        let train_labels = [0_usize, 1, 2, 1];
        let val_features = array![[0.1_f32, 0.2], [-0.3, 0.8]];
        let val_labels = [0_usize, 2];
        let first = vec![vec![0.0; param_dim], vec![0.01; param_dim]];
        let second = vec![
            (0..param_dim).map(|i| (i as f64 - 5.0) * 0.02).collect(),
            first[1].clone(),
            first[0].clone(),
        ];
        let before_uploads = dataset_upload_calls();
        let mut session = CrfmnesCudaSession::new(
            &train_features,
            &train_labels,
            &val_features,
            &val_labels,
            input_dim,
            hidden_dim,
            param_dim,
            "gpu:0",
        )
        .expect("mandatory resident CR-FM-NES CUDA dataset uploads");
        // Counter increments at the actual upload producer, and construction
        // synchronizes successful device transfers before returning.
        assert_eq!(dataset_upload_calls() - before_uploads, 2);
        let first_losses = session
            .selection_losses(&first)
            .expect("mandatory first resident CUDA evaluation");
        let second_losses = session
            .selection_losses(&second)
            .expect("mandatory second resident CUDA evaluation");
        assert_eq!(dataset_upload_calls() - before_uploads, 2);
        assert_eq!(first_losses.len(), 2);
        assert_eq!(second_losses.len(), 3);
        let bits = |loss: &(f64, f64, f64)| [loss.0.to_bits(), loss.1.to_bits(), loss.2.to_bits()];
        assert_eq!(bits(&second_losses[1]), bits(&first_losses[1]));
        assert_eq!(bits(&second_losses[2]), bits(&first_losses[0]));
        // The all-zero network has independent analytic loss ln(3), no L2.
        for loss in [first_losses[0].0, first_losses[0].1, first_losses[0].2] {
            assert!((loss - 3.0_f64.ln()).abs() < 2.0e-6);
        }
        assert!(
            session
                .selection_losses(&[vec![0.0; param_dim - 1]])
                .is_err()
        );
        assert!(session.selection_losses(&[]).unwrap().is_empty());
        assert_eq!(dataset_upload_calls() - before_uploads, 2);
        // The existing one-shot adapter exercises the same real kernels with
        // a fresh dataset, independent of the run-owned session's handles.
        let expected = try_selection_losses_cuda(
            &second,
            &train_features,
            &train_labels,
            &val_features,
            &val_labels,
            input_dim,
            hidden_dim,
            param_dim,
            "gpu:0",
        )
        .expect("mandatory one-shot comparison CUDA evaluation");
        assert_eq!(
            second_losses.iter().map(bits).collect::<Vec<_>>(),
            expected.iter().map(bits).collect::<Vec<_>>()
        );
        drop(session);

        // A subsequent run with the same shape owns different dataset bytes;
        // no shape/global cache can silently reuse the prior run's labels.
        let different_labels = [2_usize, 2, 2, 2];
        let mut fresh = CrfmnesCudaSession::new(
            &train_features,
            &different_labels,
            &val_features,
            &[],
            input_dim,
            hidden_dim,
            param_dim,
            "gpu:0",
        )
        .expect("mandatory fresh CUDA session with no validation split");
        let fresh_losses = fresh
            .selection_losses(&second)
            .expect("mandatory fresh CUDA evaluation");
        assert_eq!(dataset_upload_calls() - before_uploads, 5);
        assert!(fresh_losses.iter().all(
            |loss| loss.0.to_bits() == loss.1.to_bits() && loss.1.to_bits() == loss.2.to_bits()
        ));
        assert_ne!(fresh_losses[0].1.to_bits(), second_losses[0].1.to_bits());
    }

    #[test]
    fn crfmnes_cuda_row_tiles_match_original_kernel_on_device() {
        use cubecl::prelude::CubeElement;
        let (input, hidden) = (2, 3);
        let params = input * hidden + hidden + hidden * 3 + 3;
        let features = ndarray::Array2::from_shape_fn((9, input), |(row, col)| {
            (row as f32 - 4.0) * 0.125 * if col == 0 { 1.0 } else { -1.0 }
        });
        let labels = [0_usize, 1, 2, 0, 2, 1, 2, 0, 1];
        let validation = ndarray::Array2::from_shape_fn((3, input), |(row, col)| {
            (row as f32 + col as f32) * 0.03125
        });
        let mut extremes = vec![0.0_f64; params];
        extremes[params - 3] = 40.0;
        extremes[params - 2] = -40.0;
        let candidates = vec![
            vec![0.0; params],
            (0..params)
                .map(|index| (index as f64 - 7.0) * 0.015625)
                .collect(),
            extremes,
        ];
        let session = CrfmnesCudaSession::new(
            &features,
            &labels,
            &validation,
            &[2, 1, 0],
            input,
            hidden,
            params,
            "gpu:0",
        )
        .expect("mandatory row-parallel CUDA session");
        let flat = flatten_candidates(&candidates, params).unwrap();
        let candidate_handle = session.client.create_from_slice(f32::as_bytes(&flat));
        let baseline_train = super::launch_loss_kernel(
            &session.client,
            &candidate_handle,
            flat.len(),
            &session.train,
            candidates.len(),
            input,
            hidden,
            params,
        )
        .unwrap();
        let baseline_val = super::launch_loss_kernel(
            &session.client,
            &candidate_handle,
            flat.len(),
            session.validation.as_ref().unwrap(),
            candidates.len(),
            input,
            hidden,
            params,
        )
        .unwrap();
        for tile_rows in [1, 2, 4, 9, 32] {
            let scratch_len = candidates.len() * tile_rows;
            let scratch = session.client.empty(super::f32_bytes(scratch_len).unwrap());
            let accumulator = session
                .client
                .empty(super::f32_bytes(candidates.len() * 3).unwrap());
            let train_counts = session
                .launch_dataset_tiles(
                    &candidate_handle,
                    flat.len(),
                    &session.train,
                    &scratch,
                    scratch_len,
                    &accumulator,
                    candidates.len(),
                    tile_rows,
                    0,
                )
                .unwrap();
            let val_counts = session
                .launch_dataset_tiles(
                    &candidate_handle,
                    flat.len(),
                    session.validation.as_ref().unwrap(),
                    &scratch,
                    scratch_len,
                    &accumulator,
                    candidates.len(),
                    tile_rows,
                    1,
                )
                .unwrap();
            assert_eq!(train_counts, (9_usize.div_ceil(tile_rows) as u64, 27));
            assert_eq!(val_counts, (3_usize.div_ceil(tile_rows) as u64, 9));
            let bytes = session.client.read_one(accumulator).unwrap();
            let losses = f32::from_bytes(&bytes);
            for index in 0..candidates.len() {
                assert_eq!(
                    losses[index].to_bits(),
                    baseline_train[index].to_bits(),
                    "train tile_rows={tile_rows}"
                );
                assert_eq!(
                    losses[candidates.len() + index].to_bits(),
                    baseline_val[index].to_bits(),
                    "validation tile_rows={tile_rows}"
                );
            }
        }
        // Empty row views exercise L2-only behavior without creating a zero-byte
        // allocation or accessing the nonempty backing feature/label handles.
        let empty = super::ResidentLossInputs {
            features: session.train.features.clone(),
            labels: session.train.labels.clone(),
            feature_count: 0,
            rows: 0,
        };
        let baseline_empty = super::launch_loss_kernel(
            &session.client,
            &candidate_handle,
            flat.len(),
            &empty,
            candidates.len(),
            input,
            hidden,
            params,
        )
        .unwrap();
        let scratch = session
            .client
            .empty(super::f32_bytes(candidates.len()).unwrap());
        let accumulators = session
            .client
            .empty(super::f32_bytes(candidates.len() * 3).unwrap());
        assert_eq!(
            session
                .launch_dataset_tiles(
                    &candidate_handle,
                    flat.len(),
                    &empty,
                    &scratch,
                    candidates.len(),
                    &accumulators,
                    candidates.len(),
                    1,
                    0
                )
                .unwrap(),
            (1, 0)
        );
        let bytes = session.client.read_one(accumulators).unwrap();
        for (actual, expected) in f32::from_bytes(&bytes)
            .iter()
            .take(candidates.len())
            .zip(baseline_empty)
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}
