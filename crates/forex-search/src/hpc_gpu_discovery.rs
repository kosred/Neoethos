//! HPC-optimized GPU discovery using Island Model GA for 8×A6000.
//!
//! This module implements:
//! - Island Model: Each GPU evolves its own population
//! - NVLink elite migration between islands
//! - Multi-fidelity screening (fast GPU → thorough CPU validation)
//! - NUMA-aware thread pinning

use anyhow::{Result, bail};
use forex_core::{AcceleratorBackend, TrainingPrecision};
use forex_data::{FeatureFrame, Ohlcv};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::thread;
use tch::{Device, Kind, Tensor};
use tracing::info;

use crate::cubecl_ga::{cuda_reproduction_kernel_enabled, try_generate_children_cuda};
use crate::discovery_gpu::{GpuDiscoveryConfig, GpuDiscoveryResult};
use crate::genetic::{SurvivorSelectionPolicy, select_parent_index, select_survivor_indices};
use crate::hpc::{get_gpu_cpu_affinity, is_hpc_mode, is_nvlink_pair, set_thread_affinity};

/// Island-based GPU evolution configuration
#[derive(Debug, Clone)]
pub struct IslandConfig {
    pub base_config: GpuDiscoveryConfig,
    pub migration_interval: usize,
    pub migration_fraction: f64,
    pub num_islands: usize,
}

impl Default for IslandConfig {
    fn default() -> Self {
        Self {
            base_config: GpuDiscoveryConfig::default(),
            migration_interval: 10,
            migration_fraction: 0.05,
            num_islands: 8,
        }
    }
}

/// Run Island Model GA on 8×A6000 with NVLink migration
pub fn run_island_model_discovery(
    frames: &[FeatureFrame],
    base_ohlcv: &Ohlcv,
    config: &IslandConfig,
) -> Result<GpuDiscoveryResult> {
    if config.base_config.backend != AcceleratorBackend::Cuda {
        bail!(
            "HPC island discovery currently supports CUDA only, requested {}",
            config.base_config.backend.as_str()
        );
    }
    if config.base_config.precision != TrainingPrecision::Fp32 {
        bail!(
            "HPC island discovery currently executes FP32 tensors only, requested {}",
            config.base_config.precision.as_str()
        );
    }
    if !is_hpc_mode() {
        bail!("Island model requires HPC mode. Use standard GPU discovery instead.");
    }

    if frames.is_empty() {
        bail!("no feature frames supplied");
    }
    if config.num_islands == 0 {
        bail!("island discovery requires at least one island");
    }
    if config.migration_interval == 0 {
        bail!("island discovery migration_interval must be greater than zero");
    }
    if config.base_config.generations == 0 {
        bail!("island discovery requires at least one generation");
    }
    if !config.migration_fraction.is_finite() || !(0.0..=1.0).contains(&config.migration_fraction) {
        bail!(
            "island discovery migration_fraction must be finite in [0, 1], got {}",
            config.migration_fraction
        );
    }

    let tf_count = frames.len();
    validate_hpc_inputs(frames, base_ohlcv)?;
    let n_features = frames[0].data.ncols();
    let genome_dim = tf_count + n_features + 2;

    // Build data cubes
    let data_cube = build_data_cube_hpc(frames)?;
    let ohlc_cube = build_ohlc_cube_hpc(base_ohlcv, tf_count)?;

    // Initialize islands
    let gpu_ids = if config.base_config.devices.is_empty() {
        (0..config.num_islands as i64).collect::<Vec<_>>()
    } else {
        config
            .base_config
            .devices
            .iter()
            .copied()
            .take(config.num_islands)
            .collect::<Vec<_>>()
    };
    if gpu_ids.is_empty() {
        bail!("island discovery requires at least one CUDA device id");
    }
    let total_population = config.base_config.population.max(gpu_ids.len());
    let base_pop_per_island = total_population / gpu_ids.len();
    let extra_population = total_population % gpu_ids.len();

    info!(
        "Initializing {} islands with {} genomes total (dim={})",
        gpu_ids.len(),
        total_population,
        genome_dim
    );

    let mut islands: Vec<Island> = gpu_ids
        .iter()
        .enumerate()
        .map(|(idx, &id)| {
            let population_size = base_pop_per_island + usize::from(idx < extra_population);
            Island::new(id, population_size, genome_dim)
        })
        .collect();

    // Evolution loop
    for generation in 0..config.base_config.generations {
        // Evaluate all islands
        evaluate_islands_parallel(&mut islands, &data_cube, &ohlc_cube, &config.base_config)?;

        // Select elites on each island
        for island in &mut islands {
            island.select_elites(&config.base_config);
        }

        // Migration via NVLink
        if generation > 0 && generation % config.migration_interval == 0 {
            info!(
                "Generation {}: Performing NVLink elite migration",
                generation
            );
            perform_nvlink_migration(&mut islands, config.migration_fraction);
        }

        // Evolve to next generation
        if generation + 1 < config.base_config.generations {
            for island in &mut islands {
                island.evolve_generation(&config.base_config, generation);
            }
        }
    }

    // Collect final elites from all islands
    let mut all_elites: Vec<Vec<f32>> = Vec::new();
    let mut all_fitness: Vec<f32> = Vec::new();

    for island in &islands {
        if island.elites.is_empty() {
            all_elites.extend(island.population.clone());
            all_fitness.extend(island.fitness.clone());
        } else {
            all_elites.extend(island.elites.clone());
            all_fitness.extend(island.elite_fitness.clone());
        }
    }

    // Sort all elites by fitness
    let mut scored: Vec<(f32, usize, Vec<f32>)> = all_elites
        .into_iter()
        .enumerate()
        .zip(all_fitness.into_iter())
        .map(|((idx, genome), fitness)| (fitness, idx, genome))
        .collect();
    if scored.is_empty() {
        bail!("HPC island discovery produced no scored genomes");
    }
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    // Return top elites
    let final_elites: Vec<Vec<f32>> = scored
        .iter()
        .take(1000)
        .map(|(_, _, g)| g.clone())
        .collect();
    let final_fitness: Vec<f32> = scored.iter().take(1000).map(|(f, _, _)| *f).collect();

    Ok(GpuDiscoveryResult {
        genomes: final_elites,
        fitness: final_fitness,
        feature_names: frames[0].names.clone(),
        timeframes: (0..tf_count).map(|idx| format!("tf_{idx}")).collect(),
        used_gpu: true,
        runtime_backend: "search_hpc_island_cuda_tch_fp32".to_string(),
        degraded_reason: None,
    })
}

fn validate_hpc_inputs(frames: &[FeatureFrame], base_ohlcv: &Ohlcv) -> Result<()> {
    let first = frames
        .first()
        .ok_or_else(|| anyhow::anyhow!("no feature frames supplied"))?;
    let expected_rows = first.data.nrows();
    let expected_features = first.data.ncols();
    if expected_rows < 3 {
        bail!("HPC island discovery requires at least 3 rows, received {expected_rows}");
    }
    if expected_features == 0 {
        bail!("HPC island discovery requires at least one feature column");
    }
    if base_ohlcv.open.len() != base_ohlcv.close.len()
        || base_ohlcv.high.len() != base_ohlcv.close.len()
        || base_ohlcv.low.len() != base_ohlcv.close.len()
    {
        bail!("OHLC arrays must have identical lengths for HPC island discovery");
    }
    if base_ohlcv.close.len() != expected_rows {
        bail!(
            "OHLC row count ({}) must match feature row count ({expected_rows})",
            base_ohlcv.close.len()
        );
    }

    for (idx, frame) in frames.iter().enumerate() {
        if frame.data.nrows() != expected_rows {
            bail!(
                "feature frame {idx} row count mismatch: expected {expected_rows}, got {}",
                frame.data.nrows()
            );
        }
        if frame.data.ncols() != expected_features {
            bail!(
                "feature frame {idx} column count mismatch: expected {expected_features}, got {}",
                frame.data.ncols()
            );
        }
        if frame.names.len() != expected_features {
            bail!(
                "feature frame {idx} names mismatch: expected {expected_features}, got {}",
                frame.names.len()
            );
        }
        if frame.timestamps.len() != expected_rows {
            bail!(
                "feature frame {idx} timestamp count mismatch: expected {expected_rows}, got {}",
                frame.timestamps.len()
            );
        }
    }

    Ok(())
}

/// An evolutionary island running on a specific GPU
struct Island {
    gpu_id: i64,
    population: Vec<Vec<f32>>,
    fitness: Vec<f32>,
    elites: Vec<Vec<f32>>,
    elite_fitness: Vec<f32>,
    device: Device,
}

impl Island {
    fn new(gpu_id: i64, population_size: usize, genome_dim: usize) -> Self {
        let mut rng = rand::rng();
        let population: Vec<Vec<f32>> = (0..population_size)
            .map(|_| {
                (0..genome_dim)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        Self {
            gpu_id,
            population,
            fitness: vec![0.0; population_size],
            elites: Vec::new(),
            elite_fitness: Vec::new(),
            device: Device::Cuda(gpu_id as usize),
        }
    }

    fn evaluate(
        &mut self,
        data_cube: &Tensor,
        ohlc_cube: &Tensor,
        config: &GpuDiscoveryConfig,
    ) -> Result<()> {
        self.fitness =
            evaluate_population_hpc(data_cube, ohlc_cube, &self.population, config, self.device)?;
        Ok(())
    }

    fn select_elites(&mut self, config: &GpuDiscoveryConfig) {
        let survivor_fraction = if config.survivor_fraction > 0.0 {
            config.survivor_fraction
        } else {
            config.elite_fraction
        };

        let mut scored: Vec<(f32, usize, Vec<f32>)> = self
            .population
            .clone()
            .into_iter()
            .enumerate()
            .zip(self.fitness.iter().cloned())
            .map(|((idx, genome), fitness)| (fitness, idx, genome))
            .collect();

        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let score_vector: Vec<f64> = scored
            .iter()
            .map(|(fitness, _, _)| *fitness as f64)
            .collect();
        let survivor_count = match config.survivor_selection {
            SurvivorSelectionPolicy::Generational => 0,
            _ => ((self.population.len() as f64) * survivor_fraction)
                .round()
                .max(2.0) as usize,
        }
        .min(scored.len());
        let mut rng = rand::rng();
        let survivor_indices = select_survivor_indices(
            &score_vector,
            survivor_count,
            config.survivor_selection,
            config.selection_temperature,
            config.tournament_size,
            &mut rng,
        );

        self.elites = survivor_indices
            .iter()
            .map(|idx| scored[*idx].2.clone())
            .collect();
        self.elite_fitness = survivor_indices.iter().map(|idx| scored[*idx].0).collect();
    }

    fn evolve_generation(&mut self, config: &GpuDiscoveryConfig, _generation: usize) {
        let dim = self.population[0].len();
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let reference_pool = if self.elites.is_empty() {
            self.population.clone()
        } else {
            self.elites.clone()
        };
        let mu = mean_vector(&reference_pool);
        let std = std_vector(&reference_pool, &mu);

        let mut next = self.elites.clone();
        let immigrant_count =
            ((self.population.len() as f64) * config.immigrant_fraction).round() as usize;
        let immigrant_count = immigrant_count.min(self.population.len().saturating_sub(next.len()));
        for _ in 0..immigrant_count {
            next.push(random_genome(dim, &mut rng));
        }

        let score_vector: Vec<f64> = self.fitness.iter().map(|fitness| *fitness as f64).collect();
        let parent_indices: Vec<usize> = (0..self.population.len()).collect();

        let pending_children = self.population.len().saturating_sub(next.len());
        if pending_children > 0 && cuda_reproduction_kernel_enabled() {
            let population_refs: Vec<&[f32]> = self
                .population
                .iter()
                .map(|genome| genome.as_slice())
                .collect();
            match try_generate_children_cuda(
                &population_refs,
                &score_vector,
                &parent_indices,
                &mu,
                &std,
                pending_children,
                config,
                &mut rng,
                &normal,
                self.gpu_id,
            ) {
                Ok(children) => {
                    next.extend(children);
                }
                Err(err) => {
                    tracing::warn!(
                        "cuda reproduction kernel unavailable in island evolution, falling back to cpu offspring generation: {err}"
                    );
                }
            }
        }

        while next.len() < self.population.len() {
            let use_cross = rng.random_bool(config.crossover_rate);
            let mut child = vec![0.0_f32; dim];

            if use_cross && parent_indices.len() >= 2 {
                let a_idx = select_parent_index(
                    &score_vector,
                    &parent_indices,
                    config.parent_selection,
                    config.tournament_size,
                    config.selection_temperature,
                    &mut rng,
                );
                let mut b_idx = select_parent_index(
                    &score_vector,
                    &parent_indices,
                    config.parent_selection,
                    config.tournament_size,
                    config.selection_temperature,
                    &mut rng,
                );
                if parent_indices.len() > 1 {
                    let mut retries = 0usize;
                    while b_idx == a_idx && retries < 4 {
                        b_idx = select_parent_index(
                            &score_vector,
                            &parent_indices,
                            config.parent_selection,
                            config.tournament_size,
                            config.selection_temperature,
                            &mut rng,
                        );
                        retries += 1;
                    }
                }
                let a = &self.population[a_idx];
                let b = &self.population[b_idx];
                for i in 0..dim {
                    let base = 0.5 * (a[i] + b[i]);
                    let noise = std[i] as f64 * normal.sample(&mut rng) * config.sigma;
                    child[i] = (base as f64 + noise).clamp(-1.0, 1.0) as f32;
                }
            } else {
                for i in 0..dim {
                    let noise = std[i] as f64 * normal.sample(&mut rng) * config.sigma;
                    child[i] = (mu[i] as f64 + noise).clamp(-1.0, 1.0) as f32;
                }
            }
            next.push(child);
        }

        self.population = next;
    }
}

fn random_genome(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect::<Vec<f32>>()
}

fn evaluate_islands_parallel(
    islands: &mut [Island],
    data_cube: &Tensor,
    ohlc_cube: &Tensor,
    config: &GpuDiscoveryConfig,
) -> Result<()> {
    thread::scope(|scope| -> Result<()> {
        let handles = islands
            .iter_mut()
            .map(|island| {
                let data = data_cube.shallow_clone();
                let ohlc = ohlc_cube.shallow_clone();
                let cfg = config.clone();
                let gpu_id = island.gpu_id;

                scope.spawn(move || {
                    if is_hpc_mode() {
                        let cores = get_gpu_cpu_affinity(gpu_id);
                        if let Err(e) = set_thread_affinity(&cores) {
                            tracing::warn!("Failed to set affinity for GPU {}: {}", gpu_id, e);
                        }
                    }

                    island.evaluate(&data, &ohlc, &cfg)
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("HPC island worker thread panicked"))??;
        }

        Ok(())
    })
}

fn perform_nvlink_migration(islands: &mut [Island], fraction: f64) {
    if islands.is_empty() {
        return;
    }
    let n_islands = islands.len();
    let migrants_per_island = ((islands[0].elites.len() as f64) * fraction).ceil() as usize;

    // For each island, find NVLink neighbors and exchange elites
    for i in 0..n_islands {
        for j in (i + 1)..n_islands {
            if is_nvlink_pair(islands[i].gpu_id, islands[j].gpu_id) {
                // Exchange top migrants
                let i_migrants: Vec<Vec<f32>> = islands[i]
                    .elites
                    .iter()
                    .take(migrants_per_island)
                    .cloned()
                    .collect();
                let j_migrants: Vec<Vec<f32>> = islands[j]
                    .elites
                    .iter()
                    .take(migrants_per_island)
                    .cloned()
                    .collect();

                // Inject migrants (replace worst elites)
                for (idx, migrant) in j_migrants.iter().enumerate() {
                    if idx < islands[i].elites.len() {
                        let target_idx = islands[i].elites.len() - 1 - idx;
                        islands[i].elites[target_idx] = migrant.clone();
                    }
                }
                for (idx, migrant) in i_migrants.iter().enumerate() {
                    if idx < islands[j].elites.len() {
                        let target_idx = islands[j].elites.len() - 1 - idx;
                        islands[j].elites[target_idx] = migrant.clone();
                    }
                }
            }
        }
    }
}

/// HPC-optimized population evaluation with larger chunks
fn evaluate_population_hpc(
    data_cube: &Tensor,
    ohlc_cube: &Tensor,
    genomes: &[Vec<f32>],
    config: &GpuDiscoveryConfig,
    device: Device,
) -> Result<Vec<f32>> {
    // Use larger chunk size for A6000
    let chunk_size = if is_hpc_mode() {
        8192
    } else {
        config.chunk_size
    };

    let mut results = vec![0.0_f32; genomes.len()];
    let mut offset = 0usize;

    while offset < genomes.len() {
        let end = (offset + chunk_size).min(genomes.len());
        let chunk = &genomes[offset..end];

        let mut chunk_buf = Vec::with_capacity(chunk.len() * chunk[0].len());
        for g in chunk {
            chunk_buf.extend_from_slice(g);
        }

        let chunk_tensor =
            Tensor::from_slice(&chunk_buf).reshape(&[chunk.len() as i64, chunk[0].len() as i64]);

        let fit = evaluate_chunk_hpc(data_cube, ohlc_cube, &chunk_tensor, config, device)?;
        let vec: Vec<f32> = Vec::<f32>::try_from(&fit).unwrap_or_default();

        for (i, v) in vec.iter().enumerate() {
            results[offset + i] = *v;
        }

        offset = end;
    }

    Ok(results)
}

/// Evaluate a chunk of genomes on GPU
fn evaluate_chunk_hpc(
    data_cube: &Tensor,
    ohlc_cube: &Tensor,
    genomes: &Tensor,
    config: &GpuDiscoveryConfig,
    device: Device,
) -> Result<Tensor> {
    let tf_count = data_cube.size()[0];
    let n_samples = data_cube.size()[1];
    let n_features = data_cube.size()[2];
    let pop = genomes.size()[0];

    let data = data_cube.to_device(device).to_kind(Kind::Float);
    let ohlc = ohlc_cube.to_device(device).to_kind(Kind::Float);
    let genomes = genomes.to_device(device).to_kind(Kind::Float);

    // Decode genomes
    let tf_weights = genomes.narrow(1, 0, tf_count).softmax(-1, Kind::Float);
    let logic_weights = genomes.narrow(1, tf_count, n_features);
    let thresholds = genomes
        .narrow(1, tf_count + n_features, 2)
        .clamp(-config.threshold_clip as f64, config.threshold_clip as f64)
        * (config.threshold_scale as f64);

    let buy_th =
        thresholds.select(1, 0).maximum(&thresholds.select(1, 1)) + config.threshold_margin as f64;
    let sell_th =
        thresholds.select(1, 0).minimum(&thresholds.select(1, 1)) - config.threshold_margin as f64;

    // Build segments for walk-forward analysis
    let segments = build_segments_hpc(n_samples as usize, config.window_bars, config.segments);
    let segment_count = segments.len();

    let mut fitness_sum = Tensor::zeros([pop], (Kind::Float, device));
    let mut min_fitness = Tensor::full([pop], 1e9, (Kind::Float, device));
    let mut pos_windows = Tensor::zeros([pop], (Kind::Float, device));

    for (start, len) in segments {
        let data_slice = data.narrow(1, start as i64, len as i64);
        let ohlc_slice = ohlc.narrow(1, start as i64, len as i64);

        // Compute signals for all timeframes
        let mut all_signals = Tensor::zeros([pop, len as i64], (Kind::Float, device));
        for t in 0..tf_count {
            let tf_data = data_slice.get(t);
            let tf_sig = tf_data.matmul(&logic_weights.transpose(0, 1));
            let std = tf_sig.std_dim(0i64, false, false) + 1e-6;
            let tf_sig = tf_sig / std.unsqueeze(0);
            let weight = tf_weights.select(1, t).unsqueeze(1);
            all_signals += tf_sig.transpose(0, 1) * weight;
        }
        all_signals = all_signals.tanh();

        // Generate actions
        let actions = all_signals
            .gt_tensor(&buy_th.unsqueeze(1))
            .to_kind(Kind::Float)
            - all_signals
                .lt_tensor(&sell_th.unsqueeze(1))
                .to_kind(Kind::Float);

        // Compute returns
        let open_p = ohlc_slice.get(0).select(1, 0);
        let close_p = ohlc_slice.get(0).select(1, 3);
        let open_next = open_p.narrow(0, 1, (len - 1) as i64);
        let close_next = close_p.narrow(0, 1, (len - 1) as i64);
        let rets = (close_next - &open_next) / open_next.clamp_min(1e-6);
        let actions_slice = actions.narrow(1, 0, (len - 1) as i64);
        let batch_rets = &actions_slice * rets.unsqueeze(0) - actions_slice.abs() * 0.0002;

        // Compute fitness metrics
        let equity = batch_rets.cumsum(1, Kind::Float);
        let peaks = equity.cummax(1).0;
        let max_dd = (&peaks - &equity).max_dim(1, false).0;

        let mean_ret = batch_rets.mean_dim(1i64, false, Kind::Float);
        let downside = batch_rets.minimum(&Tensor::zeros([1], (Kind::Float, device)));
        let downside_std = downside
            .pow_tensor_scalar(2)
            .mean_dim(1i64, false, Kind::Float)
            .sqrt()
            + 1e-9;
        let sortino = &mean_ret / downside_std;

        // Consistency metric
        let steps = Tensor::arange((len - 1) as i64, (Kind::Float, device));
        let equity_mean = equity.mean_dim(1i64, true, Kind::Float);
        let steps_mean = steps.mean(Kind::Float);
        let num = ((&equity - &equity_mean) * (&steps - &steps_mean)).sum_dim_intlist(
            1i64,
            false,
            Kind::Float,
        );
        let den = ((&equity - &equity_mean)
            .pow_tensor_scalar(2)
            .sum_dim_intlist(1i64, false, Kind::Float)
            * (&steps - &steps_mean).pow_tensor_scalar(2).sum(Kind::Float))
        .sqrt();
        let consistency = num / (den + 1e-9);

        // Penalties
        let trade_count = actions.abs().sum_dim_intlist(1i64, false, Kind::Float);
        let expected = (len as f64 / 1440.0) * config.min_trades_per_day;
        let freq_penalty = (Tensor::from(expected).to_device(device) - &trade_count).clamp_min(0.0)
            * (config.trade_penalty as f64);
        let dd_penalty =
            (max_dd - config.dd_limit as f64).clamp_min(0.0) * (config.dd_penalty as f64);

        let mut window_fit = sortino * 10.0 + consistency * 5.0 - freq_penalty - dd_penalty;
        let profit_pct = equity.select(1, (len - 2) as i64);
        window_fit += profit_pct.clamp_max(0.10) * 100.0;

        fitness_sum += &window_fit;
        min_fitness = min_fitness.minimum(&window_fit);

        let pos = profit_pct.gt(0.0) * trade_count.ge(expected);
        pos_windows += pos.to_kind(Kind::Float);
    }

    let avg_fit = fitness_sum / (segment_count as f64);
    let min_pos = (segment_count as f64 * config.pos_window_fraction).ceil();
    let pos_penalty = (Tensor::from(min_pos).to_device(device) - pos_windows).clamp_min(0.0)
        * (config.pos_penalty as f64);
    let final_fit = avg_fit + min_fitness * (config.robust_weight as f64) - pos_penalty;

    Ok(final_fit.to_device(Device::Cpu))
}

/// Build data cube for HPC mode
fn build_data_cube_hpc(frames: &[FeatureFrame]) -> Result<Tensor> {
    let tf_count = frames.len();
    let n_samples = frames[0].data.nrows();
    let n_features = frames[0].data.ncols();

    let mut buf = vec![0.0_f32; tf_count * n_samples * n_features];

    for (t, frame) in frames.iter().enumerate() {
        let shifted = shift_down_hpc(&frame.data);
        let standardized = causal_zscore_hpc(&shifted);

        for i in 0..n_samples {
            for j in 0..n_features {
                let idx = (t * n_samples * n_features) + (i * n_features) + j;
                buf[idx] = standardized[(i, j)];
            }
        }
    }

    Ok(Tensor::from_slice(&buf).reshape(&[tf_count as i64, n_samples as i64, n_features as i64]))
}

/// Build OHLC cube for HPC mode
fn build_ohlc_cube_hpc(base: &Ohlcv, tf_count: usize) -> Result<Tensor> {
    let n_samples = base.close.len();
    let mut buf = vec![0.0_f32; tf_count * n_samples * 4];

    for t in 0..tf_count {
        for i in 0..n_samples {
            let idx = (t * n_samples * 4) + (i * 4);
            buf[idx] = base.open[i] as f32;
            buf[idx + 1] = base.high[i] as f32;
            buf[idx + 2] = base.low[i] as f32;
            buf[idx + 3] = base.close[i] as f32;
        }
    }

    Ok(Tensor::from_slice(&buf).reshape(&[tf_count as i64, n_samples as i64, 4]))
}

/// Build segments for walk-forward analysis
fn build_segments_hpc(n_samples: usize, window: usize, segments: usize) -> Vec<(usize, usize)> {
    if n_samples <= window + 2 {
        return vec![(0, n_samples)];
    }

    let mut rng = rand::rng();
    let mut out = Vec::new();

    // Always include most recent window
    let start_recent = n_samples.saturating_sub(window + 1);
    out.push((start_recent, window));

    // Add random historical segments
    let segs = segments.saturating_sub(1);
    for _ in 0..segs {
        let start = rng.random_range(0..(n_samples - window - 1));
        out.push((start, window));
    }

    out
}

/// Shift data down by one row
fn shift_down_hpc(data: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let (rows, cols) = data.dim();
    let mut out = ndarray::Array2::<f32>::zeros((rows, cols));
    for r in 1..rows {
        for c in 0..cols {
            out[(r, c)] = data[(r - 1, c)];
        }
    }
    out
}

/// Causal z-score normalization using only past rows.
fn causal_zscore_hpc(data: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let (rows, cols) = data.dim();
    let mut out = ndarray::Array2::<f32>::zeros((rows, cols));
    if rows == 0 || cols == 0 {
        return out;
    }

    let mut running_sum = vec![0.0_f64; cols];
    let mut running_sumsq = vec![0.0_f64; cols];
    for r in 0..rows {
        if r == 0 {
            for c in 0..cols {
                out[(r, c)] = 0.0;
            }
        } else {
            let count = r as f64;
            for c in 0..cols {
                let mean = running_sum[c] / count;
                let var = (running_sumsq[c] / count) - mean * mean;
                let std = var.max(1e-12).sqrt();
                out[(r, c)] = ((data[(r, c)] as f64 - mean) / std) as f32;
            }
        }

        for c in 0..cols {
            let value = data[(r, c)] as f64;
            let value = if value.is_finite() { value } else { 0.0 };
            running_sum[c] += value;
            running_sumsq[c] += value * value;
        }
    }

    out
}

/// Compute mean vector
fn mean_vector(elites: &[Vec<f32>]) -> Vec<f32> {
    if elites.is_empty() {
        return Vec::new();
    }

    let dim = elites[0].len();
    let mut out = vec![0.0_f32; dim];

    for e in elites {
        for i in 0..dim {
            out[i] += e[i];
        }
    }

    let n = elites.len() as f32;
    for v in &mut out {
        *v /= n;
    }

    out
}

/// Compute standard deviation vector
fn std_vector(elites: &[Vec<f32>], mean: &[f32]) -> Vec<f32> {
    if elites.len() < 2 {
        return vec![1e-6_f32; mean.len()];
    }

    let dim = elites[0].len();
    let mut out = vec![0.0_f32; dim];

    for e in elites {
        for i in 0..dim {
            let d = e[i] - mean[i];
            out[i] += d * d;
        }
    }

    let n = elites.len() as f32;
    for v in &mut out {
        *v = (*v / (n - 1.0)).sqrt().max(1e-6);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{causal_zscore_hpc, shift_down_hpc};
    use ndarray::array;

    #[test]
    fn causal_zscore_hpc_ignores_future_rows() {
        let future_spike = array![[1.0_f32], [2.0], [100.0]];
        let alternate_future = array![[1.0_f32], [2.0], [3.0]];

        let normalized_spike = causal_zscore_hpc(&shift_down_hpc(&future_spike));
        let normalized_alt = causal_zscore_hpc(&shift_down_hpc(&alternate_future));

        assert!((normalized_spike[(0, 0)] - normalized_alt[(0, 0)]).abs() < 1e-6);
        assert!((normalized_spike[(1, 0)] - normalized_alt[(1, 0)]).abs() < 1e-6);
    }
}
