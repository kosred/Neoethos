use crate::artifact_io::write_json_atomic;
use crate::scheduler_assignment::accelerator_backend_from_assignment;
use anyhow::{Result, bail};
use neoethos_core::{AcceleratorBackend, ResolvedWorkloadAssignment, TrainingPrecision, WorkloadKind};
use neoethos_data::{
    FeatureCache, FeatureFrame, FeatureProfile, Ohlcv, SymbolDataset, compute_hpc_feature_frame,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::cubecl_ga::{cuda_reproduction_kernel_enabled, try_generate_children_cuda};
use crate::genetic::{
    ParentSelectionPolicy, SurvivorSelectionPolicy, select_parent_index, select_survivor_indices,
};

#[derive(Debug, Clone)]
pub struct GpuDiscoveryConfig {
    pub population: usize,
    pub generations: usize,
    pub elite_fraction: f64,
    pub survivor_fraction: f64,
    pub immigrant_fraction: f64,
    pub parent_selection: ParentSelectionPolicy,
    pub survivor_selection: SurvivorSelectionPolicy,
    pub selection_temperature: f64,
    pub tournament_size: usize,
    pub sigma: f64,
    pub crossover_rate: f64,
    pub threshold_scale: f64,
    pub threshold_margin: f64,
    pub threshold_clip: f64,
    pub window_bars: usize,
    pub segments: usize,
    pub min_trades_per_day: f64,
    pub trade_penalty: f64,
    pub dd_limit: f64,
    pub dd_penalty: f64,
    pub robust_weight: f64,
    pub pos_window_fraction: f64,
    pub pos_penalty: f64,
    pub chunk_size: usize,
    pub devices: Vec<i64>,
    pub backend: AcceleratorBackend,
    pub precision: TrainingPrecision,
    /// Optional deterministic seed. When `Some`, the GA + segment selection use a
    /// reproducible RNG sequence so a given (config, dataset) pair yields the same
    /// genomes/segments across runs and across CPU/GPU paths.
    pub seed: Option<u64>,
}

impl Default for GpuDiscoveryConfig {
    fn default() -> Self {
        Self {
            population: 24000,
            generations: 200,
            elite_fraction: 0.05,
            survivor_fraction: 0.10,
            immigrant_fraction: 0.20,
            parent_selection: ParentSelectionPolicy::RankWeighted,
            survivor_selection: SurvivorSelectionPolicy::RankWeighted,
            selection_temperature: 0.75,
            tournament_size: 4,
            sigma: 0.5,
            crossover_rate: 0.35,
            threshold_scale: 0.10,
            threshold_margin: 0.02,
            threshold_clip: 0.30,
            window_bars: 1440 * 22 * 6,
            segments: 4,
            min_trades_per_day: 1.0,
            trade_penalty: 25.0,
            dd_limit: 0.04,
            dd_penalty: 200.0,
            robust_weight: 0.2,
            pos_window_fraction: 0.5,
            pos_penalty: 15.0,
            chunk_size: 2048,
            devices: Vec::new(),
            backend: AcceleratorBackend::Cuda,
            precision: TrainingPrecision::Fp32,
            seed: None,
        }
    }
}

impl GpuDiscoveryConfig {
    pub fn apply_scheduler_assignment(
        &mut self,
        assignment: &ResolvedWorkloadAssignment,
    ) -> &mut Self {
        if assignment.workload != WorkloadKind::StrategySearch {
            return self;
        }
        self.backend = accelerator_backend_from_assignment(assignment);
        self.devices = assignment
            .device_assignment
            .device_ids
            .iter()
            .map(|id| *id as i64)
            .collect();
        self.precision = assignment.precision_policy.precision;
        if assignment.batch_size > 0 {
            self.chunk_size = assignment.batch_size;
        }
        self
    }

    pub fn with_scheduler_assignment(mut self, assignment: &ResolvedWorkloadAssignment) -> Self {
        self.apply_scheduler_assignment(assignment);
        self
    }
}

fn make_rng(config: &GpuDiscoveryConfig, salt: u64) -> StdRng {
    let seed = match config.seed {
        Some(s) => s.wrapping_add(salt),
        None => rand::rng().random::<u64>(),
    };
    StdRng::seed_from_u64(seed)
}

#[derive(Debug, Clone)]
pub struct GpuDiscoveryResult {
    pub genomes: Vec<Vec<f32>>,
    pub fitness: Vec<f32>,
    pub feature_names: Vec<String>,
    pub timeframes: Vec<String>,
    pub used_gpu: bool,
    pub runtime_backend: String,
    pub degraded_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct GenomeExport<'a> {
    fitness: f32,
    genome: &'a [f32],
}

fn append_degraded_reason(primary: Option<String>, secondary: Option<String>) -> Option<String> {
    match (primary, secondary) {
        (Some(primary), Some(secondary)) => Some(format!("{primary}; {secondary}")),
        (Some(primary), None) => Some(primary),
        (None, Some(secondary)) => Some(secondary),
        (None, None) => None,
    }
}

fn resolve_execution_mode(config: &GpuDiscoveryConfig) -> (Vec<i64>, String, Option<String>) {
    let requested_precision_reason = (config.precision != TrainingPrecision::Fp32).then(|| {
        format!(
            "requested_search_precision_unavailable({})",
            config.precision.as_str()
        )
    });

    match config.backend {
        AcceleratorBackend::Cpu => (
            Vec::new(),
            "search_cpu_fp32".to_string(),
            requested_precision_reason,
        ),
        AcceleratorBackend::Cuda => {
            let device_ids = if config.devices.is_empty() {
                let count = tch::Cuda::device_count();
                (0..count).collect::<Vec<_>>()
            } else {
                config.devices.clone()
            };
            let backend_reason = device_ids
                .is_empty()
                .then(|| "requested_search_cuda_unavailable".to_string());
            if device_ids.is_empty() {
                // The user asked for CUDA but we can't find any device. This is
                // the failure mode that turns a 1-day GPU search into a 1500-year
                // CPU run; emit a loud, structured log so the operator notices
                // *during* the run, not three weeks later.
                tracing::error!(
                    target: "neoethos_search::gpu",
                    backend = ?config.backend,
                    devices = ?config.devices,
                    "CUDA backend requested for strategy search but no CUDA devices are available; \
                     falling back to CPU. Set FOREX_BOT_REQUIRE_GPU=1 to fail fast instead."
                );
                if std::env::var("FOREX_BOT_REQUIRE_GPU")
                    .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
                    .unwrap_or(false)
                {
                    panic!(
                        "FOREX_BOT_REQUIRE_GPU is set but no CUDA devices were detected for the \
                         strategy search. Refusing to silently fall back to CPU."
                    );
                }
                (
                    device_ids,
                    "search_cpu_fp32".to_string(),
                    append_degraded_reason(backend_reason, requested_precision_reason),
                )
            } else {
                tracing::info!(
                    target: "neoethos_search::gpu",
                    devices = ?device_ids,
                    "Strategy search running on {} CUDA device(s)",
                    device_ids.len()
                );
                (
                    device_ids,
                    "search_cuda_tch_fp32".to_string(),
                    requested_precision_reason,
                )
            }
        }
        other => (
            Vec::new(),
            "search_cpu_fp32".to_string(),
            append_degraded_reason(
                Some(format!(
                    "requested_search_backend_unavailable({})",
                    other.as_str()
                )),
                requested_precision_reason,
            ),
        ),
    }
}

pub fn save_gpu_genomes(path: impl AsRef<Path>, result: &GpuDiscoveryResult) -> Result<()> {
    let mut payload = Vec::new();
    for (g, f) in result.genomes.iter().zip(result.fitness.iter()) {
        payload.push(GenomeExport {
            fitness: *f,
            genome: g,
        });
    }
    write_json_atomic(path, &payload)
}

pub fn build_feature_cube(
    dataset: &SymbolDataset,
    base_tf: &str,
    timeframes: &[&str],
    cache: Option<&FeatureCache>,
) -> Result<(Vec<FeatureFrame>, Vec<String>, Ohlcv)> {
    let base_tf = if dataset.frames.contains_key(base_tf) {
        base_tf.to_string()
    } else if dataset.frames.contains_key("M5") {
        "M5".to_string()
    } else if dataset.frames.contains_key("M1") {
        "M1".to_string()
    } else {
        dataset
            .frames
            .keys()
            .next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no timeframes available"))?
    };

    let base_ohlcv = dataset
        .frames
        .get(&base_tf)
        .ok_or_else(|| anyhow::anyhow!("base timeframe missing"))?;

    let base_key = format!("{}_{}_base", dataset.symbol, base_tf);
    let base_frame = if let Some(cache) = cache {
        if let Some(frame) = cache.load(&base_key)? {
            frame
        } else {
            let frame = compute_hpc_feature_frame(base_ohlcv, FeatureProfile::Standard)?;
            cache.store(&base_key, &frame)?;
            frame
        }
    } else {
        compute_hpc_feature_frame(base_ohlcv, FeatureProfile::Standard)?
    };

    let base_ts = base_frame.timestamps.clone();
    let base_names = base_frame.names.clone();
    let base_aligned = base_frame.data.clone();

    let mut targets: Vec<String> = if timeframes.is_empty() {
        dataset
            .frames
            .keys()
            .filter(|tf| *tf != &base_tf)
            .cloned()
            .collect()
    } else {
        timeframes.iter().map(|tf| tf.to_string()).collect()
    };
    targets.sort();

    let mut frames = Vec::new();
    frames.push(FeatureFrame {
        timestamps: base_ts.clone(),
        names: base_names.clone(),
        data: base_aligned,
    });

    for tf in targets.iter() {
        if tf == &base_tf {
            continue;
        }
        let htf = match dataset.frames.get(tf) {
            Some(v) => v,
            None => continue,
        };
        let key = format!("{}_{}_htf", dataset.symbol, tf);
        let htf_frame = if let Some(cache) = cache {
            if let Some(frame) = cache.load(&key)? {
                frame
            } else {
                let frame = compute_hpc_feature_frame(htf, FeatureProfile::Standard)?;
                cache.store(&key, &frame)?;
                frame
            }
        } else {
            compute_hpc_feature_frame(htf, FeatureProfile::Standard)?
        };

        let aligned = align_features(&base_ts, &htf_frame.timestamps, &htf_frame.data);
        let mapped = map_feature_columns(&base_names, &htf_frame.names, &aligned);
        frames.push(FeatureFrame {
            timestamps: base_ts.clone(),
            names: base_names.clone(),
            data: mapped,
        });
    }

    Ok((frames, base_names, base_ohlcv.clone()))
}

/// Tensor-based GPU strategy search (tch / CUDA path).
///
/// **Design note — fitness parity:** this entry point uses a *returns-based*
/// fitness (cumulative `action * (close_next - open_next)/open_next` minus a
/// flat 0.0002 cost) and does NOT model SL/TP, spread, or commission. It is
/// not equivalent to the CPU GA driven by [`crate::evolve_search`]. If you
/// need an SL/TP-faithful GPU search use `evolve_search` with the `gpu`
/// feature enabled — that path uses the cubecl backtest kernel.
pub fn run_gpu_discovery(
    frames: &[FeatureFrame],
    base_ohlcv: &Ohlcv,
    config: &GpuDiscoveryConfig,
) -> Result<GpuDiscoveryResult> {
    if frames.is_empty() {
        bail!("no feature frames supplied");
    }

    let tf_count = frames.len();
    let n_samples = frames[0].data.nrows();
    let n_features = frames[0].data.ncols();
    if n_samples == 0 || n_features == 0 {
        bail!("empty feature frame");
    }

    let data_cube = build_data_cube(frames)?;
    let ohlc_cube = build_ohlc_cube(base_ohlcv, tf_count)?;

    let (device_ids, runtime_backend, degraded_reason) = resolve_execution_mode(config);
    let used_gpu = !device_ids.is_empty();

    let dim = tf_count + n_features + 2;
    // Seedable RNG so genome init / mutation / segment selection are reproducible
    // when `config.seed` is set. This is required for CPU/GPU parity checks.
    let mut rng = make_rng(config, 0xA5A5_A5A5);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut genomes: Vec<Vec<f32>> = (0..config.population)
        .map(|_| random_genome(dim, &mut rng))
        .collect();

    // Build segments once with a deterministic RNG so every chunk/device sees
    // the SAME windows for the SAME genomes. Previously each call to
    // `build_segments` made its own unseeded RNG → results changed per call.
    let mut seg_rng = make_rng(config, 0x5A5A_5A5A);
    let segments = build_segments(n_samples, config.window_bars, config.segments, &mut seg_rng);

    let mut best_genomes = Vec::new();
    let mut best_scores = Vec::new();

    for generation in 0..config.generations {
        let fitness = evaluate_population_multi_gpu(
            &data_cube,
            &ohlc_cube,
            &genomes,
            config,
            &device_ids,
            &segments,
        )?;

        let mut scored: Vec<(f32, usize, Vec<f32>)> = genomes
            .into_iter()
            .enumerate()
            .zip(fitness.into_iter())
            .map(|((idx, g), f)| (f, idx, g))
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let effective_survivor_fraction = if config.survivor_fraction > 0.0 {
            config.survivor_fraction
        } else {
            config.elite_fraction
        };
        let return_count = ((config.population as f64)
            * effective_survivor_fraction
                .max(config.elite_fraction)
                .max(0.05))
        .round()
        .max(2.0) as usize;
        let return_count = return_count.min(scored.len());
        let best_candidates: Vec<Vec<f32>> = scored
            .iter()
            .take(return_count)
            .map(|(_, _, g)| g.clone())
            .collect();
        let best_candidate_scores: Vec<f32> = scored
            .iter()
            .take(return_count)
            .map(|(f, _, _)| *f)
            .collect();

        if generation + 1 == config.generations {
            best_genomes = best_candidates.clone();
            best_scores = best_candidate_scores.clone();
            break;
        }

        let score_vector: Vec<f64> = scored
            .iter()
            .map(|(fitness, _, _)| *fitness as f64)
            .collect();
        let survivor_count = match config.survivor_selection {
            SurvivorSelectionPolicy::Generational => 0,
            _ => ((config.population as f64) * effective_survivor_fraction)
                .round()
                .max(2.0) as usize,
        }
        .min(scored.len());
        let survivor_indices = select_survivor_indices(
            &score_vector,
            survivor_count,
            config.survivor_selection,
            config.selection_temperature,
            config.tournament_size,
            &mut rng,
        );
        let survivors: Vec<Vec<f32>> = survivor_indices
            .iter()
            .map(|idx| scored[*idx].2.clone())
            .collect();

        let parent_indices: Vec<usize> = (0..scored.len()).collect();
        let reference_pool: Vec<Vec<f32>> = if survivors.is_empty() {
            best_candidates.clone()
        } else {
            survivors.clone()
        };
        let mu = mean_vector(&reference_pool);
        let std = std_vector(&reference_pool, &mu);

        let mut next = survivors;
        let immigrant_count =
            ((config.population as f64) * config.immigrant_fraction).round() as usize;
        let immigrant_count = immigrant_count.min(config.population.saturating_sub(next.len()));
        for _ in 0..immigrant_count {
            next.push(random_genome(dim, &mut rng));
        }
        let pending_children = config.population.saturating_sub(next.len());
        if pending_children > 0 && !device_ids.is_empty() && cuda_reproduction_kernel_enabled() {
            let ranked_population: Vec<&[f32]> = scored
                .iter()
                .map(|(_, _, genome)| genome.as_slice())
                .collect();
            match try_generate_children_cuda(
                &ranked_population,
                &score_vector,
                &parent_indices,
                &mu,
                &std,
                pending_children,
                config,
                &mut rng,
                &normal,
                device_ids[0],
            ) {
                Ok(children) => {
                    next.extend(children);
                }
                Err(err) => {
                    tracing::warn!(
                        "cuda reproduction kernel unavailable in discovery_gpu, falling back to cpu offspring generation: {err}"
                    );
                }
            }
        }
        while next.len() < config.population {
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
                let a = &scored[a_idx].2;
                let b = &scored[b_idx].2;
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
        genomes = next;
    }

    Ok(GpuDiscoveryResult {
        genomes: best_genomes,
        fitness: best_scores,
        feature_names: frames[0].names.clone(),
        timeframes: (0..tf_count).map(|idx| format!("tf_{idx}")).collect(),
        used_gpu,
        runtime_backend,
        degraded_reason,
    })
}

fn random_genome(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect::<Vec<f32>>()
}

fn build_data_cube(frames: &[FeatureFrame]) -> Result<Tensor> {
    let tf_count = frames.len();
    let n_samples = frames[0].data.nrows();
    let n_features = frames[0].data.ncols();
    let mut buf = vec![0.0_f32; tf_count * n_samples * n_features];

    for (t, frame) in frames.iter().enumerate() {
        let shifted = shift_down(&frame.data);
        let standardized = causal_zscore(&shifted);
        for i in 0..n_samples {
            for j in 0..n_features {
                let idx = (t * n_samples * n_features) + (i * n_features) + j;
                buf[idx] = standardized[(i, j)];
            }
        }
    }

    Ok(Tensor::from_slice(&buf).reshape(&[tf_count as i64, n_samples as i64, n_features as i64]))
}

fn build_ohlc_cube(base: &Ohlcv, tf_count: usize) -> Result<Tensor> {
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

fn evaluate_population_multi_gpu(
    data_cube: &Tensor,
    ohlc_cube: &Tensor,
    genomes: &[Vec<f32>],
    config: &GpuDiscoveryConfig,
    device_ids: &[i64],
    segments: &[(usize, usize)],
) -> Result<Vec<f32>> {
    let mut results = vec![0.0_f32; genomes.len()];

    if device_ids.is_empty() {
        let mut offset = 0usize;
        while offset < genomes.len() {
            let end = (offset + config.chunk_size).min(genomes.len());
            let chunk = &genomes[offset..end];
            let mut chunk_buf = Vec::with_capacity(chunk.len() * chunk[0].len());
            for genome in chunk {
                chunk_buf.extend_from_slice(genome);
            }
            let chunk_tensor = Tensor::from_slice(&chunk_buf)
                .reshape(&[chunk.len() as i64, chunk[0].len() as i64]);
            let fitness = evaluate_population_gpu(
                data_cube,
                ohlc_cube,
                &chunk_tensor,
                config,
                Device::Cpu,
                segments,
            )?;
            for (idx, value) in Vec::<f32>::try_from(&fitness)
                .unwrap_or_default()
                .into_iter()
                .enumerate()
            {
                results[offset + idx] = value;
            }
            offset = end;
        }
        return Ok(results);
    }

    // Keep static cubes resident per GPU to avoid repeated host->device copies per chunk.
    let mut per_device_cubes: Vec<(Device, Tensor, Tensor)> = Vec::with_capacity(device_ids.len());
    for &device_id_i64 in device_ids {
        let device = Device::Cuda(device_id_i64 as usize);
        let data_dev = data_cube.to_device(device).to_kind(Kind::Float);
        let ohlc_dev = ohlc_cube.to_device(device).to_kind(Kind::Float);
        per_device_cubes.push((device, data_dev, ohlc_dev));
    }

    let mut offset = 0usize;
    while offset < genomes.len() {
        let end = (offset + config.chunk_size).min(genomes.len());
        let chunk = &genomes[offset..end];
        let mut chunk_buf = Vec::with_capacity(chunk.len() * chunk[0].len());
        for g in chunk {
            chunk_buf.extend_from_slice(g);
        }
        let chunk_tensor =
            Tensor::from_slice(&chunk_buf).reshape(&[chunk.len() as i64, chunk[0].len() as i64]);

        let mut per_device = Vec::new();
        let split = split_tensor(&chunk_tensor, device_ids.len());
        for (i, part) in split.into_iter().enumerate() {
            let (device, data_dev, ohlc_dev) = &per_device_cubes[i];
            let fit =
                evaluate_population_gpu(data_dev, ohlc_dev, &part, config, *device, segments)?;
            per_device.push(fit);
        }

        let mut idx = offset;
        for fit in per_device {
            let vec: Vec<f32> = Vec::<f32>::try_from(&fit).unwrap_or_default();
            for v in vec {
                results[idx] = v;
                idx += 1;
            }
        }
        offset = end;
    }
    Ok(results)
}

fn split_tensor(t: &Tensor, parts: usize) -> Vec<Tensor> {
    let n = t.size()[0] as usize;
    if parts <= 1 || n <= 1 {
        return vec![t.shallow_clone()];
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    for i in 0..parts {
        let remaining = n - start;
        let take = if i == parts - 1 {
            remaining
        } else {
            (n / parts).max(1)
        };
        let len = take.min(remaining);
        if len == 0 {
            break;
        }
        out.push(t.narrow(0, start as i64, len as i64));
        start += len;
    }
    out
}

fn evaluate_population_gpu(
    data_cube: &Tensor,
    ohlc_cube: &Tensor,
    genomes: &Tensor,
    config: &GpuDiscoveryConfig,
    device: Device,
    segments: &[(usize, usize)],
) -> Result<Tensor> {
    let tf_count = data_cube.size()[0];
    let _n_samples = data_cube.size()[1];
    let n_features = data_cube.size()[2];
    let pop = genomes.size()[0];

    let data = if data_cube.device() == device {
        data_cube.shallow_clone()
    } else {
        data_cube.to_device(device).to_kind(Kind::Float)
    };
    let ohlc = if ohlc_cube.device() == device {
        ohlc_cube.shallow_clone()
    } else {
        ohlc_cube.to_device(device).to_kind(Kind::Float)
    };
    let genomes = genomes.to_device(device).to_kind(Kind::Float);

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

    // Segments are pre-built once with a deterministic RNG by the caller — every
    // chunk and every device evaluates the SAME windows. This is required for
    // CPU/GPU parity (same windows ⇒ comparable fitness).
    let segments_owned: Vec<(usize, usize)> = segments.to_vec();

    let mut fitness_sum = Tensor::zeros([pop], (Kind::Float, device));
    let mut min_fitness = Tensor::full([pop], 1e9, (Kind::Float, device));
    let mut pos_windows = Tensor::zeros([pop], (Kind::Float, device));

    for (start, len) in &segments_owned {
        let start = *start;
        let len = *len;
        let data_slice = data.narrow(1, start as i64, len as i64);
        let ohlc_slice = ohlc.narrow(1, start as i64, len as i64);
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

        let actions = all_signals
            .gt_tensor(&buy_th.unsqueeze(1))
            .to_kind(Kind::Float)
            - all_signals
                .lt_tensor(&sell_th.unsqueeze(1))
                .to_kind(Kind::Float);

        let open_p = ohlc_slice.get(0).select(1, 0);
        let close_p = ohlc_slice.get(0).select(1, 3);
        let open_next = open_p.narrow(0, 1, (len - 1) as i64);
        let close_next = close_p.narrow(0, 1, (len - 1) as i64);
        let rets = (close_next - &open_next) / open_next.clamp_min(1e-6);
        let actions_slice = actions.narrow(1, 0, (len - 1) as i64);
        let batch_rets = &actions_slice * rets.unsqueeze(0) - actions_slice.abs() * 0.0002;

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

    let avg_fit = fitness_sum / (segments_owned.len() as f64);
    let min_pos = (segments_owned.len() as f64 * config.pos_window_fraction).ceil();
    let pos_penalty = (Tensor::from(min_pos).to_device(device) - pos_windows).clamp_min(0.0)
        * (config.pos_penalty as f64);
    let final_fit = avg_fit + min_fitness * (config.robust_weight as f64) - pos_penalty;
    Ok(final_fit.to_device(Device::Cpu))
}

fn build_segments(
    n_samples: usize,
    window: usize,
    segments: usize,
    rng: &mut impl Rng,
) -> Vec<(usize, usize)> {
    if n_samples <= window + 2 {
        return vec![(0, n_samples)];
    }
    let mut out = Vec::new();
    // Most-recent window must end at the LAST bar (inclusive).
    let start_recent = n_samples.saturating_sub(window);
    out.push((start_recent, window));
    let segs = segments.saturating_sub(1);
    for _ in 0..segs {
        let start = rng.random_range(0..(n_samples - window));
        out.push((start, window));
    }
    out
}

fn align_features(
    base_ts: &[i64],
    htf_ts: &[i64],
    htf_data: &ndarray::Array2<f32>,
) -> ndarray::Array2<f32> {
    let n_base = base_ts.len();
    let n_htf = htf_ts.len();
    let n_cols = htf_data.ncols();
    let mut out = ndarray::Array2::<f32>::zeros((n_base, n_cols));
    if n_htf == 0 || n_base == 0 {
        return out;
    }
    let mut j = 0usize;
    for i in 0..n_base {
        let target = base_ts[i];
        while j + 1 < n_htf && htf_ts[j + 1] <= target {
            j += 1;
        }
        if htf_ts[j] > target {
            continue;
        }
        if j == 0 {
            continue;
        }
        let src = j - 1;
        for c in 0..n_cols {
            out[(i, c)] = htf_data[(src, c)];
        }
    }
    out
}

fn map_feature_columns(
    base_names: &[String],
    htf_names: &[String],
    aligned: &ndarray::Array2<f32>,
) -> ndarray::Array2<f32> {
    let n_rows = aligned.nrows();
    let n_cols = base_names.len();
    let mut out = ndarray::Array2::<f32>::zeros((n_rows, n_cols));
    let mut index_map = HashMap::with_capacity(htf_names.len());
    for (idx, name) in htf_names.iter().enumerate() {
        index_map.insert(name.as_str(), idx);
    }
    for (col_idx, name) in base_names.iter().enumerate() {
        if let Some(src_idx) = index_map.get(name.as_str()) {
            for row in 0..n_rows {
                out[(row, col_idx)] = aligned[(row, *src_idx)];
            }
        }
    }
    out
}

fn shift_down(data: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let (rows, cols) = data.dim();
    let mut out = ndarray::Array2::<f32>::zeros((rows, cols));
    for r in 1..rows {
        for c in 0..cols {
            out[(r, c)] = data[(r - 1, c)];
        }
    }
    out
}

fn causal_zscore(data: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
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

fn mean_vector(elites: &[Vec<f32>]) -> Vec<f32> {
    let dim = elites[0].len();
    let mut out = vec![0.0_f32; dim];
    for e in elites {
        for i in 0..dim {
            out[i] += e[i];
        }
    }
    let n = elites.len().max(1) as f32;
    for v in &mut out {
        *v /= n;
    }
    out
}

fn std_vector(elites: &[Vec<f32>], mean: &[f32]) -> Vec<f32> {
    let dim = elites[0].len();
    let mut out = vec![0.0_f32; dim];
    for e in elites {
        for i in 0..dim {
            let d = e[i] - mean[i];
            out[i] += d * d;
        }
    }
    let n = elites.len().max(2) as f32;
    for v in &mut out {
        *v = (*v / (n - 1.0)).sqrt();
        if *v < 1e-6 {
            *v = 1e-6;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{causal_zscore, shift_down};
    use ndarray::array;

    #[test]
    fn causal_zscore_ignores_future_rows() {
        let future_spike = array![[1.0_f32], [2.0], [100.0]];
        let alternate_future = array![[1.0_f32], [2.0], [3.0]];

        let normalized_spike = causal_zscore(&shift_down(&future_spike));
        let normalized_alt = causal_zscore(&shift_down(&alternate_future));

        assert!((normalized_spike[(0, 0)] - normalized_alt[(0, 0)]).abs() < 1e-6);
        assert!((normalized_spike[(1, 0)] - normalized_alt[(1, 0)]).abs() < 1e-6);
    }

    #[test]
    fn causal_zscore_is_past_only_and_finite() {
        let data = array![[10.0_f32, 2.0], [12.0, 4.0], [50.0, 100.0]];
        let normalized = causal_zscore(&data);

        assert_eq!(normalized[(0, 0)], 0.0);
        assert_eq!(normalized[(0, 1)], 0.0);
        assert!(normalized[(1, 0)].is_finite());
        assert!(normalized[(1, 1)].is_finite());
    }
}
