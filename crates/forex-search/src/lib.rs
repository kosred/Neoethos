pub mod challenge;
#[cfg(feature = "gpu")]
mod cubecl_eval;
#[cfg(feature = "gpu")]
mod cubecl_ga;
pub mod discovery;
#[cfg(feature = "gpu")]
pub mod discovery_gpu;
#[cfg(not(feature = "gpu"))]
pub mod discovery_gpu {
    use crate::eval::{BacktestSettings, fast_evaluate_strategy_core};
    use crate::genetic::strategy_gene::infer_market_cost_profile;
    use crate::genetic::{
        ParentSelectionPolicy, SurvivorSelectionPolicy, month_day_indices, select_parent_index,
        select_survivor_indices,
    };
    use anyhow::{Result, bail};
    use forex_core::{AcceleratorBackend, TrainingPrecision};
    use forex_data::{FeatureCache, FeatureFrame, Ohlcv, SymbolDataset, compute_hpc_features};
    use ndarray::Array2;
    use rand::Rng;
    use serde::Serialize;
    use std::cmp::Ordering;
    use std::collections::HashMap;
    use std::path::Path;

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
            }
        }
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

    fn append_degraded_reason(
        primary: Option<String>,
        secondary: Option<String>,
    ) -> Option<String> {
        match (primary, secondary) {
            (Some(primary), Some(secondary)) => Some(format!("{primary}; {secondary}")),
            (Some(primary), None) => Some(primary),
            (None, Some(secondary)) => Some(secondary),
            (None, None) => None,
        }
    }

    fn resolve_cpu_fallback_runtime(config: &GpuDiscoveryConfig) -> (String, Option<String>) {
        let backend_reason = match config.backend {
            AcceleratorBackend::Cpu => None,
            AcceleratorBackend::Cuda => Some("requested_search_cuda_unavailable".to_string()),
            other => Some(format!(
                "requested_search_backend_unavailable({})",
                other.as_str()
            )),
        };
        let precision_reason = (config.precision != TrainingPrecision::Fp32).then(|| {
            format!(
                "requested_search_precision_unavailable({})",
                config.precision.as_str()
            )
        });
        (
            "search_cpu_fp32".to_string(),
            append_degraded_reason(backend_reason, precision_reason),
        )
    }

    pub fn save_gpu_genomes(path: impl AsRef<Path>, result: &GpuDiscoveryResult) -> Result<()> {
        let mut payload = Vec::new();
        for (g, f) in result.genomes.iter().zip(result.fitness.iter()) {
            payload.push(GenomeExport {
                fitness: *f,
                genome: g,
            });
        }
        let json = serde_json::to_string_pretty(&payload)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn base_timeframe(dataset: &SymbolDataset, requested: &str) -> Result<String> {
        if dataset.frames.contains_key(requested) {
            return Ok(requested.to_string());
        }
        if dataset.frames.contains_key("M5") {
            return Ok("M5".to_string());
        }
        if dataset.frames.contains_key("M1") {
            return Ok("M1".to_string());
        }
        dataset
            .frames
            .keys()
            .next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no timeframes available"))
    }

    fn feature_frame_for_ohlcv(
        ohlcv: &Ohlcv,
        cache_key: &str,
        cache: Option<&FeatureCache>,
    ) -> Result<FeatureFrame> {
        if let Some(cache) = cache {
            if let Some(frame) = cache.load(cache_key)? {
                return Ok(frame);
            }
        }

        let frame = compute_hpc_features(ohlcv)?;
        if let Some(cache) = cache {
            cache.store(cache_key, &frame)?;
        }
        Ok(frame)
    }

    fn align_features(base_ts: &[i64], htf_ts: &[i64], htf_data: &Array2<f32>) -> Array2<f32> {
        let n_base = base_ts.len();
        let n_htf = htf_ts.len();
        let n_cols = htf_data.ncols();
        let mut out = Array2::<f32>::zeros((n_base, n_cols));
        if n_htf == 0 || n_base == 0 {
            return out;
        }

        let mut j = 0usize;
        for i in 0..n_base {
            let target = base_ts[i];
            while j + 1 < n_htf && htf_ts[j + 1] <= target {
                j += 1;
            }
            if htf_ts[j] > target || j == 0 {
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
        aligned: &Array2<f32>,
    ) -> Array2<f32> {
        let n_rows = aligned.nrows();
        let n_cols = base_names.len();
        let mut out = Array2::<f32>::zeros((n_rows, n_cols));
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

    fn build_segments(n_samples: usize, window: usize, segments: usize) -> Vec<(usize, usize)> {
        if n_samples <= window + 2 {
            return vec![(0, n_samples)];
        }
        let mut rng = rand::rng();
        let mut out = Vec::new();
        let start_recent = n_samples.saturating_sub(window + 1);
        out.push((start_recent, window));
        for _ in 0..segments.saturating_sub(1) {
            let start = rng.random_range(0..(n_samples - window - 1));
            out.push((start, window));
        }
        out
    }

    fn random_genome(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
        (0..dim)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect()
    }

    fn softmax_weights(values: &[f32]) -> Vec<f32> {
        if values.is_empty() {
            return Vec::new();
        }
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = values.iter().map(|v| (v - max).exp().max(1e-6)).collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return vec![1.0 / values.len() as f32; values.len()];
        }
        exps.into_iter().map(|v| v / sum).collect()
    }

    fn build_signals_for_genome(
        genome: &[f32],
        frames: &[FeatureFrame],
        start: usize,
        len: usize,
        config: &GpuDiscoveryConfig,
    ) -> Vec<i8> {
        let tf_count = frames.len();
        let n_features = frames[0].data.ncols();
        let tf_weights = softmax_weights(&genome[..tf_count]);
        let feature_weights = &genome[tf_count..tf_count + n_features];
        let thr_a = genome[tf_count + n_features]
            .clamp(-config.threshold_clip as f32, config.threshold_clip as f32)
            * config.threshold_scale as f32;
        let thr_b = genome[tf_count + n_features + 1]
            .clamp(-config.threshold_clip as f32, config.threshold_clip as f32)
            * config.threshold_scale as f32;
        let buy_th = thr_a.max(thr_b) + config.threshold_margin as f32;
        let sell_th = thr_a.min(thr_b) - config.threshold_margin as f32;

        let mut signals = vec![0_i8; len];
        for (row, signal) in signals.iter_mut().enumerate().take(len) {
            let mut composite = 0.0_f32;
            for (tf_idx, frame) in frames.iter().enumerate() {
                let frame_row = start + row;
                if frame_row >= frame.data.nrows() {
                    continue;
                }
                let mut tf_score = 0.0_f32;
                for (feat_idx, weight) in feature_weights.iter().enumerate().take(n_features) {
                    tf_score += *weight * frame.data[(frame_row, feat_idx)];
                }
                composite += tf_weights[tf_idx] * tf_score;
            }
            *signal = if composite > buy_th {
                1
            } else if composite < sell_th {
                -1
            } else {
                0
            };
        }
        signals
    }

    fn evaluate_genome_cpu(
        genome: &[f32],
        frames: &[FeatureFrame],
        ohlcv: &Ohlcv,
        config: &GpuDiscoveryConfig,
        months: &[i64],
        days: &[i64],
    ) -> f32 {
        if frames.is_empty() || frames[0].data.nrows() == 0 {
            return f32::NEG_INFINITY;
        }

        let n_samples = frames[0].data.nrows().min(ohlcv.close.len());
        let segments = build_segments(n_samples, config.window_bars, config.segments);
        let mut fitness_sum = 0.0_f32;
        let mut min_fitness = f32::INFINITY;
        let mut pos_windows = 0.0_f32;

        let market_profile =
            infer_market_cost_profile("", "", ohlcv.close.last().copied(), None, None);

        for &(start, len) in &segments {
            let end = (start + len).min(n_samples);
            if end <= start + 1 {
                continue;
            }
            let signals = build_signals_for_genome(genome, frames, start, end - start, config);
            let metrics = fast_evaluate_strategy_core(
                &ohlcv.close[start..end],
                &ohlcv.high[start..end],
                &ohlcv.low[start..end],
                &signals,
                &months[start..end],
                &days[start..end],
                &BacktestSettings {
                    sl_pips: 20.0,
                    tp_pips: 40.0,
                    max_hold_bars: 0,
                    trailing_enabled: false,
                    trailing_atr_multiplier: 1.0,
                    trailing_be_trigger_r: 1.0,
                    pip_value: market_profile.pip_value,
                    spread_pips: market_profile.spread_pips,
                    commission_per_trade: market_profile.commission_per_trade,
                    pip_value_per_lot: market_profile.pip_value_per_lot,
                },
            );

            let net_profit = metrics[0] as f32;
            let sharpe = metrics[1] as f32;
            let max_dd = metrics[3] as f32;
            let trade_count = metrics[8] as f32;
            let consistency = metrics[9] as f32;
            let profit_pct = (net_profit / 100_000.0).clamp(-1.0, 1.0);
            let expected =
                (end.saturating_sub(start) as f32 / 1440.0) * config.min_trades_per_day as f32;
            let freq_penalty = (expected - trade_count).max(0.0) * config.trade_penalty as f32;
            let dd_penalty = (max_dd - config.dd_limit as f32).max(0.0) * config.dd_penalty as f32;
            let window_fit = sharpe * 10.0 + consistency * 5.0 - freq_penalty - dd_penalty
                + profit_pct.clamp(-1.0, 0.10) * 100.0;

            fitness_sum += window_fit;
            min_fitness = min_fitness.min(window_fit);
            if profit_pct > 0.0 && trade_count >= expected {
                pos_windows += 1.0;
            }
        }

        if !fitness_sum.is_finite() {
            return f32::NEG_INFINITY;
        }

        let avg_fit = fitness_sum / (segments.len().max(1) as f32);
        let min_pos = (segments.len() as f32 * config.pos_window_fraction as f32).ceil();
        let pos_penalty = (min_pos - pos_windows).max(0.0) * config.pos_penalty as f32;
        avg_fit + min_fitness * config.robust_weight as f32 - pos_penalty
    }

    fn refine_genome_cpu(
        genome: &[f32],
        frames: &[FeatureFrame],
        ohlcv: &Ohlcv,
        config: &GpuDiscoveryConfig,
        months: &[i64],
        days: &[i64],
    ) -> (Vec<f32>, f32) {
        let mut rng = rand::rng();
        let mut best_genome = genome.to_vec();
        let mut best_score = evaluate_genome_cpu(&best_genome, frames, ohlcv, config, months, days);
        let mutation_scale = config.sigma.max(0.05) as f32 * 0.5;

        for _ in 0..24 {
            let mut candidate = best_genome.clone();
            for value in &mut candidate {
                let delta = rng.random_range(-mutation_scale..mutation_scale);
                *value = (*value + delta).clamp(-1.0, 1.0);
            }
            let score = evaluate_genome_cpu(&candidate, frames, ohlcv, config, months, days);
            if score.is_finite() && (score > best_score || !best_score.is_finite()) {
                best_genome = candidate;
                best_score = score;
            }
        }

        (best_genome, best_score)
    }

    pub fn build_feature_cube(
        dataset: &SymbolDataset,
        base_tf: &str,
        timeframes: &[&str],
        cache: Option<&FeatureCache>,
    ) -> Result<(Vec<FeatureFrame>, Vec<String>, Ohlcv)> {
        let base_tf = base_timeframe(dataset, base_tf)?;
        let base_ohlcv = dataset
            .frames
            .get(&base_tf)
            .ok_or_else(|| anyhow::anyhow!("base timeframe missing"))?;

        let base_key = format!("{}_{}_base", dataset.symbol, base_tf);
        let base_frame = feature_frame_for_ohlcv(base_ohlcv, &base_key, cache)?;
        let base_ts = base_frame.timestamps.clone();
        let base_names = base_frame.names.clone();
        let mut frames = vec![FeatureFrame {
            timestamps: base_ts.clone(),
            names: base_names.clone(),
            data: base_frame.data.clone(),
        }];

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

        for tf in targets {
            if tf == base_tf {
                continue;
            }
            let Some(htf) = dataset.frames.get(&tf) else {
                continue;
            };
            let key = format!("{}_{}_htf", dataset.symbol, tf);
            let htf_frame = feature_frame_for_ohlcv(htf, &key, cache)?;
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

    pub fn run_gpu_discovery(
        frames: &[FeatureFrame],
        ohlcv: &Ohlcv,
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
        if ohlcv.close.len() < n_samples
            || ohlcv.high.len() < n_samples
            || ohlcv.low.len() < n_samples
        {
            bail!("ohlcv length does not match feature frame rows");
        }
        let (runtime_backend, degraded_reason) = resolve_cpu_fallback_runtime(config);

        let dim = tf_count + n_features + 2;
        let mut rng = rand::rng();
        let population = config.population.max(8);
        let generations = config.generations.max(1);
        let mut genomes: Vec<Vec<f32>> = (0..population)
            .map(|_| random_genome(dim, &mut rng))
            .collect();
        let (months, days) = match ohlcv.timestamp.as_ref() {
            Some(ts) if ts.len() >= n_samples => month_day_indices(&ts[..n_samples]),
            _ => (vec![0_i64; n_samples], vec![0_i64; n_samples]),
        };

        let mut best_genomes = Vec::new();
        let mut best_scores = Vec::new();

        for generation in 0..generations {
            let fitness: Vec<f32> = genomes
                .iter()
                .map(|genome| evaluate_genome_cpu(genome, frames, ohlcv, config, &months, &days))
                .collect();

            let mut scored: Vec<(f32, Vec<f32>)> = genomes
                .into_iter()
                .zip(fitness.into_iter())
                .map(|(g, f)| (f, g))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

            let effective_survivor_fraction = if config.survivor_fraction > 0.0 {
                config.survivor_fraction
            } else {
                config.elite_fraction
            };
            let return_count = ((population as f64)
                * effective_survivor_fraction
                    .max(config.elite_fraction)
                    .max(0.05))
            .round()
            .max(2.0) as usize;
            let return_count = return_count.min(scored.len());
            let best_candidates: Vec<Vec<f32>> = scored
                .iter()
                .take(return_count)
                .map(|(_, g)| g.clone())
                .collect();
            let best_candidate_scores: Vec<f32> =
                scored.iter().take(return_count).map(|(f, _)| *f).collect();

            if generation + 1 == generations {
                best_genomes = best_candidates;
                best_scores = best_candidate_scores;
                break;
            }

            let score_vector: Vec<f64> =
                scored.iter().map(|(fitness, _)| *fitness as f64).collect();
            let survivor_count = match config.survivor_selection {
                SurvivorSelectionPolicy::Generational => 0,
                _ => ((population as f64) * effective_survivor_fraction)
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
                .map(|idx| scored[*idx].1.clone())
                .collect();

            let mut next = survivors;
            let immigrant_count =
                ((population as f64) * config.immigrant_fraction).round() as usize;
            let immigrant_count = immigrant_count.min(population.saturating_sub(next.len()));
            for _ in 0..immigrant_count {
                next.push(random_genome(dim, &mut rng));
            }

            let parent_indices: Vec<usize> = (0..scored.len()).collect();
            while next.len() < population {
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

                let a = &scored[a_idx].1;
                let b = &scored[b_idx].1;
                let mut child = Vec::with_capacity(dim);
                for i in 0..dim {
                    let base = if rng.random_bool(config.crossover_rate) {
                        a[i]
                    } else {
                        b[i]
                    };
                    let noise = rng.random_range(-config.sigma as f32..config.sigma as f32);
                    child.push((base + noise).clamp(-1.0, 1.0));
                }
                next.push(child);
            }

            genomes = next;
        }

        let mut refined: Vec<(Vec<f32>, f32)> = best_genomes
            .into_iter()
            .zip(best_scores)
            .map(|(genome, score)| {
                let (refined_genome, refined_score) =
                    refine_genome_cpu(&genome, frames, ohlcv, config, &months, &days);
                if refined_score > score {
                    (refined_genome, refined_score)
                } else {
                    (genome, score)
                }
            })
            .collect();
        refined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        Ok(GpuDiscoveryResult {
            genomes: refined.iter().map(|(genome, _)| genome.clone()).collect(),
            fitness: refined.iter().map(|(_, score)| *score).collect(),
            feature_names: frames[0].names.clone(),
            timeframes: (0..tf_count).map(|idx| format!("tf_{idx}")).collect(),
            used_gpu: false,
            runtime_backend,
            degraded_reason,
        })
    }
}

// HPC-specific modules - only compiled on GPU-enabled builds.
#[cfg(feature = "gpu")]
pub mod hpc;
#[cfg(feature = "gpu")]
pub mod hpc_gpu_discovery;
#[cfg(feature = "gpu")]
pub mod hpc_simd;

// Re-export HPC functions when GPU feature is enabled.
#[cfg(feature = "gpu")]
pub use hpc::{
    detect_hyperstack_n3, force_hpc_mode, get_gpu_cpu_affinity, get_optimal_chunk_size,
    get_optimal_population, get_validation_cpu_cores, is_hpc_mode, is_nvlink_pair,
    print_hpc_config, set_thread_affinity,
};
#[cfg(feature = "gpu")]
pub use hpc_gpu_discovery::{IslandConfig, run_island_model_discovery};
#[cfg(feature = "gpu")]
pub use hpc_simd::{batch_evaluate_simd, compute_sharpe_ratio, has_avx2};

pub mod eval;
pub mod gauntlet;
pub mod genetic;
pub mod orchestration;
pub mod portfolio;
pub mod quality;
pub mod stop_target;
pub mod validation;

pub use challenge::{ChallengeOptimizer, ChallengeTarget};
pub use discovery::{
    DiscoveryConfig, DiscoveryProgress, DiscoveryResult, DiscoveryRunProfile, LoggedStrategyTrades,
    build_discovery_profile, ensure_non_empty_portfolio, run_discovery_cycle,
    run_discovery_cycle_with_progress, save_discovery_profile_json, save_portfolio_json,
    save_quality_report_json, save_trade_log_json,
};
pub use discovery_gpu::{
    GpuDiscoveryConfig, GpuDiscoveryResult, build_feature_cube, run_gpu_discovery, save_gpu_genomes,
};
pub use eval::{
    BacktestSettings, evaluate_population_core, fast_evaluate_strategy_core, simulate_trades_core,
};
pub use gauntlet::{GauntletConfig, StrategyGauntlet};
pub use genetic::{
    EvaluationConfig, EvolutionSearchPolicy, FilteringConfig, Gene, ParentSelectionPolicy,
    SearchResult, SurvivorSelectionPolicy, evaluate_genes, evolve_search,
    evolve_search_with_progress, evolve_search_with_progress_and_limits, month_day_indices,
    random_search, signals_for_gene,
};
pub use orchestration::{BatchDiscoverySummary, DiscoveryOrchestrator};
pub use portfolio::{AllocationResult, PortfolioOptimizer, SymbolMetrics};
pub use quality::{StrategyMetrics, StrategyQualityAnalyzer, StrategyRanker, Trade};
pub use stop_target::{StopTargetSettings, compute_stop_distance_series, infer_stop_target_pips};
pub use validation::{
    CombinatorialPurgedCV, WalkforwardSplitResult, WalkforwardSummary,
    embargoed_walkforward_backtest,
};
