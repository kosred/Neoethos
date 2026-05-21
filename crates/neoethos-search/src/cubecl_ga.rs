use anyhow::{Context, Result, bail};
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use neoethos_core::TrainingPrecision;
use half::bf16;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::discovery_gpu::GpuDiscoveryConfig;
use crate::genetic::select_parent_index;

#[derive(Debug)]
struct CudaReproductionBatch {
    dim: usize,
    crossover_mask: Vec<u32>,
    parent_a: Vec<f32>,
    parent_b: Vec<f32>,
    noise: Vec<f32>,
}

impl CudaReproductionBatch {
    fn with_capacity(child_count: usize, dim: usize) -> Self {
        let flat_capacity = child_count.saturating_mul(dim);
        Self {
            dim,
            crossover_mask: Vec::with_capacity(child_count),
            parent_a: Vec::with_capacity(flat_capacity),
            parent_b: Vec::with_capacity(flat_capacity),
            noise: Vec::with_capacity(flat_capacity),
        }
    }

    fn push_crossover(&mut self, a: &[f32], b: &[f32]) {
        self.crossover_mask.push(1);
        self.parent_a.extend_from_slice(a);
        self.parent_b.extend_from_slice(b);
    }

    fn push_mean_only(&mut self) {
        self.crossover_mask.push(0);
        self.parent_a
            .resize(self.parent_a.len().saturating_add(self.dim), 0.0);
        self.parent_b
            .resize(self.parent_b.len().saturating_add(self.dim), 0.0);
    }

    fn push_noise_row<R: Rng + ?Sized>(
        &mut self,
        std: &[f32],
        sigma: f64,
        rng: &mut R,
        normal: &Normal<f64>,
    ) {
        for value in std {
            let sample = (*value as f64) * normal.sample(rng) * sigma;
            self.noise.push(sample as f32);
        }
    }

    fn child_count(&self) -> usize {
        self.crossover_mask.len()
    }

    fn validate(&self, mean: &[f32]) -> Result<()> {
        if self.dim == 0 {
            bail!("cuda reproduction batch requires a positive genome dimension");
        }
        if mean.len() != self.dim {
            bail!(
                "cuda reproduction mean vector length mismatch: expected {}, received {}",
                self.dim,
                mean.len()
            );
        }
        let expected_flat = self.child_count().saturating_mul(self.dim);
        if self.parent_a.len() != expected_flat
            || self.parent_b.len() != expected_flat
            || self.noise.len() != expected_flat
        {
            bail!(
                "cuda reproduction flat buffer mismatch: expected {} elements, received a={}, b={}, noise={}",
                expected_flat,
                self.parent_a.len(),
                self.parent_b.len(),
                self.noise.len()
            );
        }
        Ok(())
    }
}

#[cube(launch)]
fn blend_mutate_kernel<F: Float + CubeElement>(
    parent_a: &Array<F>,
    parent_b: &Array<F>,
    mean: &Array<F>,
    noise: &Array<F>,
    crossover_mask: &Array<u32>,
    output: &mut Array<F>,
    dim: u32,
) {
    if ABSOLUTE_POS < output.len() {
        let pos = ABSOLUTE_POS;
        let dim = dim as usize;
        let child = pos / dim;
        let feature = pos % dim;
        let blended = (parent_a[pos] + parent_b[pos]) * F::new(0.5);
        let base = select(crossover_mask[child] == 1, blended, mean[feature]);
        let low = F::new(-1.0);
        let high = F::new(1.0);
        let value = base + noise[pos];
        let value = select(value < low, low, value);
        output[pos] = select(value > high, high, value);
    }
}

fn kernel_units(client: &ComputeClient<CudaRuntime>) -> u32 {
    let max_units = client.properties().hardware.max_units_per_cube.max(1);
    std::env::var("FOREX_BOT_SEARCH_GPU_KERNEL_UNITS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(max_units)
        .min(max_units)
        .max(1)
}

fn launch_cuda_kernel<F>(
    client: &ComputeClient<CudaRuntime>,
    batch: &CudaReproductionBatch,
    mean: &[F],
    parent_a: &[F],
    parent_b: &[F],
    noise: &[F],
) -> Result<Vec<F>>
where
    F: Float + CubeElement,
{
    let total_len = batch.child_count().saturating_mul(batch.dim);
    if total_len == 0 {
        return Ok(Vec::new());
    }

    let parent_a_handle = client.create_from_slice(F::as_bytes(parent_a));
    let parent_b_handle = client.create_from_slice(F::as_bytes(parent_b));
    let mean_handle = client.create_from_slice(F::as_bytes(mean));
    let noise_handle = client.create_from_slice(F::as_bytes(noise));
    let mask_handle = client.create_from_slice(u32::as_bytes(&batch.crossover_mask));
    let output_handle = client.empty(total_len.saturating_mul(std::mem::size_of::<F>()));

    let units = kernel_units(client);
    let cubes = (total_len as u32).div_ceil(units);
    blend_mutate_kernel::launch::<F, CudaRuntime>(
        client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts::<F>(&parent_a_handle, total_len, 1) },
        unsafe { ArrayArg::from_raw_parts::<F>(&parent_b_handle, total_len, 1) },
        unsafe { ArrayArg::from_raw_parts::<F>(&mean_handle, batch.dim, 1) },
        unsafe { ArrayArg::from_raw_parts::<F>(&noise_handle, total_len, 1) },
        unsafe { ArrayArg::from_raw_parts::<u32>(&mask_handle, batch.child_count(), 1) },
        unsafe { ArrayArg::from_raw_parts::<F>(&output_handle, total_len, 1) },
        ScalarArg::new(batch.dim as u32),
    )
    .context("launch cuda blend/mutate kernel")?;

    let bytes = client.read_one(output_handle);
    Ok(F::from_bytes(&bytes).to_vec())
}

fn to_bf16_vec(values: &[f32]) -> Vec<bf16> {
    values.iter().map(|value| bf16::from_f32(*value)).collect()
}

fn flatten_rows(values: &[f32], dim: usize) -> Vec<Vec<f32>> {
    values.chunks(dim).map(|row| row.to_vec()).collect()
}

fn prefers_bf16(requested: TrainingPrecision) -> bool {
    matches!(
        requested,
        TrainingPrecision::Bf16 | TrainingPrecision::Fp8 | TrainingPrecision::Bf4
    )
}

pub(crate) fn cuda_reproduction_kernel_enabled() -> bool {
    !matches!(
        std::env::var("FOREX_BOT_SEARCH_CUDA_REPRO_KERNEL")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "0" | "false" | "off" | "disable" | "disabled")
    )
}

fn execute_cuda_batch(
    batch: &CudaReproductionBatch,
    mean: &[f32],
    requested_precision: TrainingPrecision,
    device_id: usize,
) -> Result<Vec<Vec<f32>>> {
    batch.validate(mean)?;

    let device = CudaDevice::new(device_id);
    let client = CudaRuntime::client(&device);

    if prefers_bf16(requested_precision) {
        let mean_bf16 = to_bf16_vec(mean);
        let parent_a_bf16 = to_bf16_vec(&batch.parent_a);
        let parent_b_bf16 = to_bf16_vec(&batch.parent_b);
        let noise_bf16 = to_bf16_vec(&batch.noise);

        match launch_cuda_kernel::<bf16>(
            &client,
            batch,
            &mean_bf16,
            &parent_a_bf16,
            &parent_b_bf16,
            &noise_bf16,
        ) {
            Ok(values) => {
                let flat = values
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>();
                return Ok(flatten_rows(&flat, batch.dim));
            }
            Err(err) => {
                tracing::debug!(
                    "cuda bf16 reproduction kernel unavailable, falling back to fp32: {err}"
                );
            }
        }
    }

    let flat = launch_cuda_kernel::<f32>(
        &client,
        batch,
        mean,
        &batch.parent_a,
        &batch.parent_b,
        &batch.noise,
    )
    .context("launch fp32 cuda reproduction kernel")?;
    Ok(flatten_rows(&flat, batch.dim))
}

pub(crate) fn try_generate_children_cuda<R: Rng>(
    population: &[&[f32]],
    score_vector: &[f64],
    parent_indices: &[usize],
    mean: &[f32],
    std: &[f32],
    child_count: usize,
    config: &GpuDiscoveryConfig,
    rng: &mut R,
    normal: &Normal<f64>,
    device_id: i64,
) -> Result<Vec<Vec<f32>>> {
    if child_count == 0 {
        return Ok(Vec::new());
    }
    if population.is_empty() {
        bail!("cuda reproduction kernel requires a non-empty population");
    }
    if mean.is_empty() {
        bail!("cuda reproduction kernel requires a non-empty mean vector");
    }
    if std.len() != mean.len() {
        bail!(
            "cuda reproduction std length mismatch: expected {}, received {}",
            mean.len(),
            std.len()
        );
    }
    if population.iter().any(|genome| genome.len() != mean.len()) {
        bail!("cuda reproduction kernel received inconsistent genome dimensions");
    }
    if device_id < 0 {
        bail!("cuda reproduction kernel received invalid device id {device_id}");
    }

    let mut batch = CudaReproductionBatch::with_capacity(child_count, mean.len());
    for _ in 0..child_count {
        let use_cross = rng.random_bool(config.crossover_rate);
        if use_cross && parent_indices.len() >= 2 {
            let a_idx = select_parent_index(
                score_vector,
                parent_indices,
                config.parent_selection,
                config.tournament_size,
                config.selection_temperature,
                rng,
            );
            let mut b_idx = select_parent_index(
                score_vector,
                parent_indices,
                config.parent_selection,
                config.tournament_size,
                config.selection_temperature,
                rng,
            );
            if parent_indices.len() > 1 {
                let mut retries = 0usize;
                while b_idx == a_idx && retries < 4 {
                    b_idx = select_parent_index(
                        score_vector,
                        parent_indices,
                        config.parent_selection,
                        config.tournament_size,
                        config.selection_temperature,
                        rng,
                    );
                    retries += 1;
                }
            }
            batch.push_crossover(population[a_idx], population[b_idx]);
        } else {
            batch.push_mean_only();
        }
        batch.push_noise_row(std, config.sigma, rng, normal);
    }

    execute_cuda_batch(&batch, mean, config.precision, device_id as usize)
}
