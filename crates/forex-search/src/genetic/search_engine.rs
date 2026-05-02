use super::evolution_math::{
    EvolutionSearchPolicy, ParentSelectionPolicy, SeenSignatureMemory, SurvivorSelectionPolicy,
    apply_metrics, crossover, generate_random_genes, mutate, new_random_gene, select_parent_index,
    select_survivor_indices, unique_candidate_or_retry,
};
use super::smc_indicators::{SmcSearchConfig, build_smc_arrays, enforce_population_smc_ratio};
use super::strategy_gene::{EvaluationConfig, Gene, SearchResult};
use crate::eval::BacktestSettings;
use crate::stop_target::{StopTargetSettings, infer_stop_target_pips};
use anyhow::{Result, anyhow, bail};
use chrono::{Datelike, TimeZone, Utc};
use forex_data::{FeatureFrame, Ohlcv};
use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Build a deterministic RNG seeded from `FOREX_BOT_SEARCH_SEED` (if set) or
/// from the OS RNG otherwise. Used by the GA so that runs are reproducible
/// when the operator pins a seed (matching the parity work in
/// `discovery_gpu`/`lib.rs`).
fn build_search_rng() -> StdRng {
    let seed = std::env::var("FOREX_BOT_SEARCH_SEED")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or_else(|| rand::rng().random::<u64>());
    StdRng::seed_from_u64(seed)
}

type GeneArrays = (Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Holds data that is stable across all generations (OHLCV + features don't change).
/// Computing this once outside the generation loop saves ~5-15% eval time.
pub struct EvalDataCache {
    pub indicators: ndarray::Array2<f32>,
    pub months: Vec<i64>,
    pub days: Vec<i64>,
    pub smc_data: Vec<crate::eval::SmcRow>,
}

impl EvalDataCache {
    pub fn build(features: &FeatureFrame, ohlcv: &Ohlcv) -> Self {
        let indicators = transpose_features(features);
        let (months, days) = month_day_indices(&features.timestamps);
        let n_samples = features.data.nrows();
        let (ob, fvg, liq, trend, prem, ind, bos, choch, eqh, eql, disp) =
            build_smc_arrays(features, ohlcv);
        let mut smc_data = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            smc_data.push([
                ob[i], fvg[i], liq[i], trend[i], prem[i], ind[i], bos[i], choch[i], eqh[i], eql[i],
                disp[i],
            ]);
        }
        Self {
            indicators,
            months,
            days,
            smc_data,
        }
    }
}

pub fn month_day_indices(timestamps: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let mut months = Vec::with_capacity(timestamps.len());
    let mut days = Vec::with_capacity(timestamps.len());
    for ts in timestamps {
        if let Some(dt) = Utc.timestamp_millis_opt(*ts).single() {
            let month_key = (dt.year() as i64) * 12 + dt.month() as i64;
            let day_key = (dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64;
            months.push(month_key);
            days.push(day_key);
        } else {
            months.push(0);
            days.push(0);
        }
    }
    (months, days)
}

fn build_gene_arrays(genes: &[Gene]) -> GeneArrays {
    let mut offsets = Vec::with_capacity(genes.len() + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let mut long_thr = Vec::with_capacity(genes.len());
    let mut short_thr = Vec::with_capacity(genes.len());
    offsets.push(0);
    for gene in genes {
        long_thr.push(gene.long_threshold);
        short_thr.push(gene.short_threshold);
        for (idx, weight) in gene.indices.iter().zip(gene.weights.iter()) {
            indices.push(*idx as i32);
            weights.push(*weight);
        }
        offsets.push(indices.len() as i32);
    }
    (offsets, indices, weights, long_thr, short_thr)
}

fn transpose_features(frame: &FeatureFrame) -> Array2<f32> {
    frame.data.t().to_owned()
}

pub fn signals_for_gene(features: &FeatureFrame, gene: &Gene) -> Vec<i8> {
    let n_samples = features.data.nrows();
    let mut combined = vec![0.0_f32; n_samples];
    for (idx, weight) in gene.indices.iter().zip(gene.weights.iter()) {
        if *idx >= features.data.ncols() {
            continue;
        }
        let col = features.data.column(*idx);
        for (i, v) in col.iter().enumerate() {
            combined[i] += *weight * *v;
        }
    }
    let mut signals = vec![0_i8; n_samples];
    for i in 0..n_samples {
        let v = combined[i];
        if v >= gene.long_threshold {
            signals[i] = 1;
        } else if v <= gene.short_threshold {
            signals[i] = -1;
        }
    }
    signals
}

/// Evaluate genes using a pre-built EvalDataCache (avoids recomputing stable arrays each generation).
pub fn evaluate_genes_cached(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    genes: &[Gene],
    config: &EvaluationConfig,
    cache: &EvalDataCache,
) -> Result<Vec<[f64; 11]>> {
    if genes.is_empty() {
        return Ok(Vec::new());
    }
    let (offsets, indices, weights, long_thr, short_thr) = build_gene_arrays(genes);
    let (sl_pips, tp_pips) = resolve_stop_target_arrays(genes, ohlcv, config);
    let mut gene_smc_flags = Vec::with_capacity(genes.len());
    for g in genes {
        gene_smc_flags.push([
            g.use_ob as i8,
            g.use_fvg as i8,
            g.use_liq_sweep as i8,
            g.mtf_confirmation as i8,
            g.use_premium_discount as i8,
            g.use_inducement as i8,
            g.use_bos as i8,
            g.use_choch as i8,
            g.use_eqh as i8,
            g.use_eql as i8,
            g.use_displacement as i8,
        ]);
    }

    let smc_weights = [
        config.smc_weight_ob,
        config.smc_weight_fvg,
        config.smc_weight_liq,
        config.smc_weight_mtf,
        config.smc_weight_premium,
        config.smc_weight_inducement,
        config.smc_weight_bos,
        config.smc_weight_choch,
        config.smc_weight_eqh,
        config.smc_weight_eql,
        config.smc_weight_displacement,
    ];

    let b_settings = BacktestSettings {
        max_hold_bars: config.max_hold_bars,
        trailing_enabled: config.trailing_enabled,
        trailing_atr_multiplier: config.trailing_atr_multiplier,
        trailing_be_trigger_r: config.trailing_be_trigger_r,
        pip_value: config.pip_value,
        spread_pips: config.spread_pips,
        commission_per_trade: config.commission_per_trade,
        pip_value_per_lot: config.pip_value_per_lot,
        ..Default::default()
    };

    crate::eval::evaluate_population_core(crate::eval::PopulationEvalInputs {
        close: &ohlcv.close,
        high: &ohlcv.high,
        low: &ohlcv.low,
        indicators: cache.indicators.view(),
        gene_offsets: &offsets,
        gene_indices: &indices,
        gene_weights: &weights,
        long_thr: &long_thr,
        short_thr: &short_thr,
        month_idx: &cache.months,
        day_idx: &cache.days,
        timestamps: &features.timestamps,
        sl_pips: &sl_pips,
        tp_pips: &tp_pips,
        smc_data: &cache.smc_data,
        gene_smc_flags: &gene_smc_flags,
        gate_threshold: config.smc_gate_threshold,
        weights: &smc_weights,
        settings: &b_settings,
    })
    .map_err(|e| anyhow!(e))
}

pub fn evaluate_genes(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    genes: &[Gene],
    config: &EvaluationConfig,
) -> Result<Vec<[f64; 11]>> {
    if features.data.nrows() == 0 || features.data.ncols() == 0 {
        bail!("empty feature matrix");
    }
    let n_samples = features.data.nrows();
    if ohlcv.close.len() != n_samples {
        bail!("ohlcv length does not match feature rows");
    }

    let indicators = transpose_features(features);
    let (offsets, indices, weights, long_thr, short_thr) = build_gene_arrays(genes);
    let (sl_pips, tp_pips) = resolve_stop_target_arrays(genes, ohlcv, config);
    let (months, days) = month_day_indices(&features.timestamps);

    let (ob, fvg, liq, trend, prem, ind, bos, choch, eqh, eql, disp) =
        build_smc_arrays(features, ohlcv);
    let mut smc_data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        smc_data.push([
            ob[i], fvg[i], liq[i], trend[i], prem[i], ind[i], bos[i], choch[i], eqh[i], eql[i],
            disp[i],
        ]);
    }
    let mut gene_smc_flags = Vec::with_capacity(genes.len());
    for g in genes {
        gene_smc_flags.push([
            g.use_ob as i8,
            g.use_fvg as i8,
            g.use_liq_sweep as i8,
            g.mtf_confirmation as i8,
            g.use_premium_discount as i8,
            g.use_inducement as i8,
            g.use_bos as i8,
            g.use_choch as i8,
            g.use_eqh as i8,
            g.use_eql as i8,
            g.use_displacement as i8,
        ]);
    }

    let smc_weights = [
        config.smc_weight_ob,
        config.smc_weight_fvg,
        config.smc_weight_liq,
        config.smc_weight_mtf,
        config.smc_weight_premium,
        config.smc_weight_inducement,
        config.smc_weight_bos,
        config.smc_weight_choch,
        config.smc_weight_eqh,
        config.smc_weight_eql,
        config.smc_weight_displacement,
    ];

    let b_settings = BacktestSettings {
        max_hold_bars: config.max_hold_bars,
        trailing_enabled: config.trailing_enabled,
        trailing_atr_multiplier: config.trailing_atr_multiplier,
        trailing_be_trigger_r: config.trailing_be_trigger_r,
        pip_value: config.pip_value,
        spread_pips: config.spread_pips,
        commission_per_trade: config.commission_per_trade,
        pip_value_per_lot: config.pip_value_per_lot,
        ..Default::default()
    };

    crate::eval::evaluate_population_core(crate::eval::PopulationEvalInputs {
        close: &ohlcv.close,
        high: &ohlcv.high,
        low: &ohlcv.low,
        indicators: indicators.view(),
        gene_offsets: &offsets,
        gene_indices: &indices,
        gene_weights: &weights,
        long_thr: &long_thr,
        short_thr: &short_thr,
        month_idx: &months,
        day_idx: &days,
        timestamps: &features.timestamps,
        sl_pips: &sl_pips,
        tp_pips: &tp_pips,
        smc_data: &smc_data,
        gene_smc_flags: &gene_smc_flags,
        gate_threshold: config.smc_gate_threshold,
        weights: &smc_weights,
        settings: &b_settings,
    })
    .map_err(|e| anyhow!(e))
}

fn resolve_stop_target_arrays(
    genes: &[Gene],
    ohlcv: &Ohlcv,
    config: &EvaluationConfig,
) -> (Vec<f64>, Vec<f64>) {
    let pip_size = if config.pip_value.is_finite() && config.pip_value > 0.0 {
        config.pip_value
    } else {
        0.0001
    };
    let default = infer_stop_target_pips(
        &ohlcv.open,
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
        &StopTargetSettings::default(),
        pip_size,
        0,
    );
    let (default_sl, default_tp) = default
        .map(|(sl, tp, _rr)| (sl, tp))
        .unwrap_or((20.0, 40.0));

    let mut sl_pips = Vec::with_capacity(genes.len());
    let mut tp_pips = Vec::with_capacity(genes.len());
    for gene in genes {
        sl_pips.push(if gene.sl_pips.is_finite() && gene.sl_pips > 0.0 {
            gene.sl_pips
        } else {
            default_sl
        });
        tp_pips.push(if gene.tp_pips.is_finite() && gene.tp_pips > 0.0 {
            gene.tp_pips
        } else {
            default_tp
        });
    }
    (sl_pips, tp_pips)
}

pub fn random_search(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    n_genes: usize,
    max_indicators: usize,
) -> Result<SearchResult> {
    let n_indicators = features.data.ncols();
    let smc_cfg = SmcSearchConfig::from_env();
    let mut rng = build_search_rng();
    let mut genes =
        generate_random_genes(n_genes, n_indicators, max_indicators, 0, &smc_cfg, &mut rng);
    enforce_population_smc_ratio(&mut genes, &smc_cfg);
    for gene in genes.iter_mut() {
        gene.normalize(n_indicators, 1);
    }
    let metrics = evaluate_genes(features, ohlcv, &genes, &EvaluationConfig::default())?;
    Ok(SearchResult { genes, metrics })
}

pub fn evolve_search(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    population: usize,
    generations: usize,
    max_indicators: usize,
) -> Result<SearchResult> {
    evolve_search_with_progress(
        features,
        ohlcv,
        population,
        generations,
        max_indicators,
        None,
        |_, _, _, _, _| {},
    )
}

#[allow(clippy::too_many_arguments)]
pub fn evolve_search_with_progress_and_limits<F>(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    population: usize,
    generations: usize,
    max_indicators: usize,
    max_runtime: Option<Duration>,
    eval_config: Option<EvaluationConfig>,
    progress_fn: F,
) -> Result<SearchResult>
where
    F: FnMut(usize, usize, f64, usize, usize),
{
    evolve_search_with_progress_impl(
        features,
        ohlcv,
        population,
        generations,
        max_indicators,
        max_runtime,
        eval_config,
        progress_fn,
    )
}

pub fn evolve_search_with_progress<F>(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    population: usize,
    generations: usize,
    max_indicators: usize,
    eval_config: Option<EvaluationConfig>,
    progress_fn: F,
) -> Result<SearchResult>
where
    F: FnMut(usize, usize, f64, usize, usize),
{
    evolve_search_with_progress_impl(
        features,
        ohlcv,
        population,
        generations,
        max_indicators,
        None,
        eval_config,
        progress_fn,
    )
}

#[allow(clippy::too_many_arguments)]
fn evolve_search_with_progress_impl<F>(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    population: usize,
    generations: usize,
    max_indicators: usize,
    max_runtime: Option<Duration>,
    eval_config: Option<EvaluationConfig>,
    mut progress_fn: F,
) -> Result<SearchResult>
where
    F: FnMut(usize, usize, f64, usize, usize),
{
    if population == 0 {
        bail!("population must be > 0");
    }
    let n_indicators = features.data.ncols();
    let smc_cfg = SmcSearchConfig::from_env();

    let env_f32 = |n, d| {
        std::env::var(n)
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(d)
    };
    let gate_start = env_f32(
        "FOREX_BOT_PROP_SMC_GATE_START",
        env_f32("FOREX_BOT_PROP_SMC_GATE", 0.75),
    );
    let gate_end = env_f32("FOREX_BOT_PROP_SMC_GATE_END", 0.35);
    let gate_curve = env_f32("FOREX_BOT_PROP_SMC_GATE_CURVE", 1.0).max(0.1);
    let gate_stagnation_step = env_f32("FOREX_BOT_PROP_SMC_GATE_STAGNATION_STEP", 0.03).max(0.0);
    let (gate_lo, gate_hi) = (gate_start.min(gate_end), gate_start.max(gate_end));

    let mut eval_cfg = eval_config.unwrap_or_default();
    eval_cfg.smc_gate_threshold = gate_start.clamp(gate_lo, gate_hi);

    let seen_retry_attempts = std::env::var("FOREX_BOT_PROP_SEEN_RETRY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16)
        .max(1);
    let mut seen_memory = SeenSignatureMemory::from_env();
    let mut rng = build_search_rng();
    let mut genes = generate_random_genes(
        population,
        n_indicators,
        max_indicators,
        0,
        &smc_cfg,
        &mut rng,
    );
    enforce_population_smc_ratio(&mut genes, &smc_cfg);

    genes = genes
        .into_iter()
        .map(|g| {
            unique_candidate_or_retry(
                g,
                &mut seen_memory,
                n_indicators,
                max_indicators,
                0,
                seen_retry_attempts,
                &smc_cfg,
                &mut rng,
            )
        })
        .collect();

    let mut best_metrics = Vec::new();
    let mut profitable_archive: Vec<(Gene, [f64; 11], usize)> = Vec::new();
    let mut archive_seq = 0usize;
    let mut seen_strategy_ids: HashSet<String> = HashSet::new();

    let env_str = |n, d: &str| {
        std::env::var(n)
            .unwrap_or_else(|_| d.to_string())
            .to_ascii_lowercase()
    };
    let env_f64 = |n, d| {
        std::env::var(n)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(d)
    };
    let archive_mode = env_str("FOREX_BOT_PROP_ARCHIVE_MODE", "net");
    let (archive_min_net, archive_min_pf, archive_min_sharpe) = (
        env_f64("FOREX_BOT_PROP_ARCHIVE_MIN_NET", 0.0),
        env_f64("FOREX_BOT_PROP_ARCHIVE_MIN_PF", 1.0),
        env_f64("FOREX_BOT_PROP_ARCHIVE_MIN_SHARPE", 0.0),
    );
    // Cap archive to prevent memory explosion on large HPC runs (Remove #3)
    let archive_cap = std::env::var("FOREX_BOT_PROP_ARCHIVE_CAP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or((population * generations.max(1)).min(50_000))
        .max(population)
        .min(200_000);
    let base_immigrant_ratio = env_f64("FOREX_BOT_PROP_RANDOM_IMMIGRANTS", 0.25).clamp(0.0, 0.95);
    let base_survivor_fraction = env_f64(
        "FOREX_BOT_PROP_SURVIVOR_FRACTION",
        env_f64("FOREX_BOT_PROP_ELITE_FRACTION", 0.10),
    )
    .clamp(0.0, 0.95);
    let parent_selection =
        ParentSelectionPolicy::parse(&env_str("FOREX_BOT_PROP_PARENT_SELECTION", "rank"));
    let survivor_selection =
        SurvivorSelectionPolicy::parse(&env_str("FOREX_BOT_PROP_SURVIVOR_SELECTION", "rank"));
    let selection_temperature = env_f64("FOREX_BOT_PROP_SELECTION_TEMPERATURE", 0.75).max(1e-3);
    let tournament_size = std::env::var("FOREX_BOT_PROP_TOURNAMENT_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or((population / 12).max(3))
        .max(2);
    let search_policy = EvolutionSearchPolicy::new(
        base_survivor_fraction,
        base_immigrant_ratio,
        parent_selection,
        survivor_selection,
        selection_temperature,
        tournament_size,
    );
    let stagnation_patience = std::env::var("FOREX_BOT_PROP_STAGNATION_GENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2)
        .max(1);

    // Perf #1: read env vars once before the generation loop
    let novelty_weight: f64 = std::env::var("FOREX_BOT_NOVELTY_WEIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0); // Default OFF to avoid O(n²) cost; set > 0 only for large populations

    // Perf #3: build stable eval data cache ONCE before the generation loop
    let eval_cache = EvalDataCache::build(features, ohlcv);

    let started_at = Instant::now();
    let mut best_score_seen = f64::NEG_INFINITY;
    let mut stagnant_gens = 0usize;

    if generations == 0 {
        let metrics = evaluate_genes_cached(features, ohlcv, &genes, &eval_cfg, &eval_cache)?;
        apply_metrics(&mut genes, &metrics);
        seen_memory.flush();
        return Ok(SearchResult { genes, metrics });
    }

    for generation in 0..generations {
        let progress = (generation as f32) / ((generations - 1) as f32).max(1.0);
        let mut gate_now = gate_start + (gate_end - gate_start) * progress.powf(gate_curve);
        if stagnant_gens >= stagnation_patience {
            gate_now -= gate_stagnation_step * (stagnant_gens as f32);
        }
        eval_cfg.smc_gate_threshold = gate_now.clamp(gate_lo, gate_hi);

        let metrics = evaluate_genes_cached(features, ohlcv, &genes, &eval_cfg, &eval_cache)?;
        apply_metrics(&mut genes, &metrics);

        let mut scored: Vec<(f64, usize, Gene, [f64; 11])> = genes
            .iter()
            .cloned()
            .zip(metrics)
            .enumerate()
            .map(|(idx, (g, m))| (g.fitness, idx, g, m))
            .collect();

        // --- Novelty Search: Behavioral Diversity ---
        // Pre-compute all HashSets once and run the O(n²) Jaccard pass in
        // parallel — turns a single-threaded bottleneck into Ncores× faster.
        if novelty_weight > 0.0 && scored.len() > 1 {
            use rayon::prelude::*;
            let n_pop = scored.len();
            let index_sets: Vec<HashSet<usize>> = scored
                .iter()
                .map(|(_, _, g, _)| g.indices.iter().copied().collect())
                .collect();

            // Parallel: each row i computes its mean Jaccard distance to the
            // remaining population. Each pair is touched twice (i→j and j→i),
            // matching the previous semantics exactly while running in parallel.
            let novelty_scores: Vec<f64> = (0..n_pop)
                .into_par_iter()
                .map(|i| {
                    let sig_i = &index_sets[i];
                    let mut dist_sum = 0.0;
                    for (j, sig_j) in index_sets.iter().enumerate() {
                        if i == j {
                            continue;
                        }
                        let intersection = sig_i.intersection(sig_j).count() as f64;
                        let union = sig_i.union(sig_j).count() as f64;
                        let jaccard_dist = if union > 0.0 {
                            1.0 - (intersection / union)
                        } else {
                            0.0
                        };
                        dist_sum += jaccard_dist;
                    }
                    dist_sum / (n_pop as f64 - 1.0)
                })
                .collect();

            // Normalize and blend
            let min_fit = scored
                .iter()
                .map(|(f, _, _, _)| *f)
                .filter(|f| f.is_finite())
                .fold(f64::INFINITY, f64::min);
            let max_fit = scored
                .iter()
                .map(|(f, _, _, _)| *f)
                .filter(|f| f.is_finite())
                .fold(f64::NEG_INFINITY, f64::max);
            let fit_range = (max_fit - min_fit).max(1e-9);
            let max_nov = novelty_scores
                .iter()
                .copied()
                .fold(0.0_f64, f64::max)
                .max(1e-9);

            for i in 0..n_pop {
                if !scored[i].0.is_finite() {
                    continue;
                }
                let norm_fit = (scored[i].0 - min_fit) / fit_range;
                let norm_nov = novelty_scores[i] / max_nov;
                // Modify the sorting score purely for the tournament survival/elites
                scored[i].0 = (1.0 - novelty_weight) * norm_fit + novelty_weight * norm_nov;
            }
        }
        // ------------------------------------------

        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let top_score = scored.first().map(|x| x.0).unwrap_or(f64::NEG_INFINITY);
        if top_score > best_score_seen + 1e-12 {
            best_score_seen = top_score;
            stagnant_gens = 0;
        } else {
            stagnant_gens += 1;
        }

        for (_score, _, gene, m) in scored.iter() {
            if profitable_archive.len() >= archive_cap {
                break;
            }
            let (net, sharpe, pf, trades) = (m[0], m[1], m[5], m[8]);
            if !net.is_finite() || !sharpe.is_finite() || !pf.is_finite() || !trades.is_finite() {
                continue;
            }
            let keep = match archive_mode.as_str() {
                "active" => trades > 0.0,
                "pf" | "profit_factor" => trades > 0.0 && pf > archive_min_pf,
                "sharpe" => trades > 0.0 && sharpe > archive_min_sharpe,
                _ => trades > 0.0 && net > archive_min_net,
            };
            if !keep {
                continue;
            }
            let sid = if gene.strategy_id.is_empty() {
                format!(
                    "{:?}|{:?}|{:.3}|{:.3}",
                    gene.indices, gene.weights, gene.long_threshold, gene.short_threshold
                )
            } else {
                gene.strategy_id.clone()
            };
            if !seen_strategy_ids.insert(sid) {
                continue;
            }
            profitable_archive.push((gene.clone(), *m, archive_seq));
            archive_seq += 1;
        }

        progress_fn(
            generation + 1,
            generations,
            top_score,
            stagnant_gens,
            profitable_archive.len(),
        );

        if let Some(max_runtime) = max_runtime
            && started_at.elapsed() >= max_runtime
        {
            let best_return_count = population
                .clamp(2, (population / 2).clamp(100, 500))
                .min(scored.len());
            let top_candidates: Vec<Gene> = scored
                .iter()
                .take(best_return_count)
                .map(|(_, _, g, _)| g.clone())
                .collect();
            let top_metrics: Vec<[f64; 11]> = scored
                .iter()
                .take(best_return_count)
                .map(|(_, _, _, m)| *m)
                .collect();
            seen_memory.flush();
            if !profitable_archive.is_empty() {
                profitable_archive.sort_by(|a, b| {
                    b.1[0]
                        .partial_cmp(&a.1[0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.2.cmp(&b.2))
                });
                return Ok(SearchResult {
                    genes: profitable_archive
                        .iter()
                        .map(|(g, _, _)| g.clone())
                        .collect(),
                    metrics: profitable_archive.iter().map(|(_, m, _)| *m).collect(),
                });
            }
            return Ok(SearchResult {
                genes: top_candidates,
                metrics: top_metrics,
            });
        }

        let best_return_count = population
            .clamp(2, (population / 2).clamp(100, 500))
            .min(scored.len());
        let top_candidates: Vec<Gene> = scored
            .iter()
            .take(best_return_count)
            .map(|(_, _, g, _)| g.clone())
            .collect();
        best_metrics = scored
            .iter()
            .take(best_return_count)
            .map(|(_, _, _, m)| *m)
            .collect();

        if generation + 1 == generations {
            seen_memory.flush();
            if !profitable_archive.is_empty() {
                profitable_archive.sort_by(|a, b| {
                    b.1[0]
                        .partial_cmp(&a.1[0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.2.cmp(&b.2))
                });
                return Ok(SearchResult {
                    genes: profitable_archive
                        .iter()
                        .map(|(g, _, _)| g.clone())
                        .collect(),
                    metrics: profitable_archive.iter().map(|(_, m, _)| *m).collect(),
                });
            }
            return Ok(SearchResult {
                genes: top_candidates,
                metrics: best_metrics,
            });
        }

        // Reuse the seeded RNG built at the top of `evolve_search_with_progress_impl`
        // (was `let mut rng = rand::rng();` here, which shadowed the seeded one and
        // broke the determinism work in the GPU path). `rng` is available in scope.
        let score_vector: Vec<f64> = scored.iter().map(|(score, _, _, _)| *score).collect();
        let survivor_fraction = if stagnant_gens >= stagnation_patience {
            (search_policy.survivor_fraction * 0.75).clamp(0.0, 0.5)
        } else {
            search_policy.survivor_fraction
        };
        let survivor_count = ((population as f64) * survivor_fraction).round() as usize;
        let survivor_count = match search_policy.survivor_selection {
            SurvivorSelectionPolicy::Generational => 0,
            _ => survivor_count.clamp(2, scored.len()),
        };
        let survivor_indices = select_survivor_indices(
            &score_vector,
            survivor_count,
            search_policy.survivor_selection,
            search_policy.selection_temperature,
            search_policy.tournament_size,
            &mut rng,
        );
        let survivors: Vec<Gene> = survivor_indices
            .iter()
            .map(|idx| scored[*idx].2.clone())
            .collect();

        let mut next = Vec::with_capacity(population);
        next.extend(survivors);
        let immigrant_ratio = if stagnant_gens >= stagnation_patience {
            search_policy.immigrant_fraction.max(0.5)
        } else {
            search_policy.immigrant_fraction
        };
        let immigrant_count = ((population as f64) * immigrant_ratio).round() as usize;
        let immigrant_count = immigrant_count.min(population - next.len());
        for _ in 0..immigrant_count {
            let immigrant = new_random_gene(
                n_indicators,
                max_indicators,
                generation + 1,
                &smc_cfg,
                &mut rng,
            );
            next.push(unique_candidate_or_retry(
                immigrant,
                &mut seen_memory,
                n_indicators,
                max_indicators,
                generation + 1,
                seen_retry_attempts,
                &smc_cfg,
                &mut rng,
            ));
        }

        let parent_indices: Vec<usize> = (0..scored.len()).collect();
        while next.len() < population {
            let a_idx = select_parent_index(
                &score_vector,
                &parent_indices,
                search_policy.parent_selection,
                search_policy.tournament_size,
                search_policy.selection_temperature,
                &mut rng,
            );
            let mut b_idx = select_parent_index(
                &score_vector,
                &parent_indices,
                search_policy.parent_selection,
                search_policy.tournament_size,
                search_policy.selection_temperature,
                &mut rng,
            );
            if parent_indices.len() > 1 {
                let mut retries = 0usize;
                while b_idx == a_idx && retries < 4 {
                    b_idx = select_parent_index(
                        &score_vector,
                        &parent_indices,
                        search_policy.parent_selection,
                        search_policy.tournament_size,
                        search_policy.selection_temperature,
                        &mut rng,
                    );
                    retries += 1;
                }
            }
            let a = &scored[a_idx].2;
            let b = &scored[b_idx].2;
            let crossed = crossover(a, b, generation + 1, &mut rng);
            let mutated = mutate(
                &crossed,
                n_indicators,
                max_indicators,
                generation + 1,
                &smc_cfg,
                stagnant_gens,
                &mut rng,
            );
            next.push(unique_candidate_or_retry(
                mutated,
                &mut seen_memory,
                n_indicators,
                max_indicators,
                generation + 1,
                seen_retry_attempts,
                &smc_cfg,
                &mut rng,
            ));
        }
        enforce_population_smc_ratio(&mut next, &smc_cfg);
        genes = next;
        seen_memory.flush();
    }
    seen_memory.flush();
    Ok(SearchResult {
        genes,
        metrics: best_metrics,
    })
}
