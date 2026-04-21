use crate::utils::build_ohlcv;
use forex_data::{FeatureProfile, compute_hpc_feature_frame};
use forex_search::genetic::{ParentSelectionPolicy, SurvivorSelectionPolicy};
use forex_search::{GpuDiscoveryConfig, evolve_search, run_gpu_discovery};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pythonize::pythonize;
use std::collections::HashSet;

fn fmt_f64(v: f64) -> String {
    format!("{v:.10}")
}

fn fmt_f32(v: f32) -> String {
    format!("{v:.8}")
}

fn parse_parent_selection(raw: &str) -> PyResult<ParentSelectionPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "uniform" => Ok(ParentSelectionPolicy::Uniform),
        "rank" | "rank_weighted" | "rank-weighted" => Ok(ParentSelectionPolicy::RankWeighted),
        "softmax" | "boltzmann" => Ok(ParentSelectionPolicy::Softmax),
        "tournament" => Ok(ParentSelectionPolicy::Tournament),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported parent_selection `{other}`; expected one of uniform, rank, softmax, tournament"
        ))),
    }
}

fn parse_survivor_selection(raw: &str) -> PyResult<SurvivorSelectionPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "elitist" | "elite" => Ok(SurvivorSelectionPolicy::Elitist),
        "rank" | "rank_weighted" | "rank-weighted" => Ok(SurvivorSelectionPolicy::RankWeighted),
        "tournament" => Ok(SurvivorSelectionPolicy::Tournament),
        "generational" | "none" | "non_elitist" | "non-elitist" => {
            Ok(SurvivorSelectionPolicy::Generational)
        }
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported survivor_selection `{other}`; expected one of elitist, rank, tournament, generational"
        ))),
    }
}

fn pythonize_genes(py: Python, genes: &[forex_search::Gene]) -> PyResult<Vec<Py<PyAny>>> {
    genes
        .iter()
        .map(|gene| {
            pythonize(py, gene)
                .map(|obj| obj.unbind())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, timestamps=None, volume=None, population=64, generations=20, max_indicators=12, include_raw=true))]
#[allow(clippy::too_many_arguments)]
pub fn search_evolve_ohlcv(
    py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    timestamps: Option<PyReadonlyArray1<i64>>,
    volume: Option<PyReadonlyArray1<f64>>,
    population: usize,
    generations: usize,
    max_indicators: usize,
    include_raw: bool,
) -> PyResult<Py<PyAny>> {
    let ohlcv = build_ohlcv(
        &open,
        &high,
        &low,
        &close,
        timestamps.as_ref(),
        volume.as_ref(),
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let (features, result) = py
        .detach(|| {
            let prof = if include_raw {
                FeatureProfile::Full
            } else {
                FeatureProfile::Standard
            };
            let features = compute_hpc_feature_frame(&ohlcv, prof)
                .map_err(|e| format!("Feature computation failed: {}", e))?;
            let result = evolve_search(&features, &ohlcv, population, generations, max_indicators)
                .map_err(|e| format!("Search failed: {}", e))?;
            Ok::<_, String>((features, result))
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    let metrics: Vec<Vec<f64>> = result
        .metrics
        .iter()
        .map(|m: &[f64; 11]| m.to_vec())
        .collect();
    let genes_py = pythonize_genes(py, &result.genes)?;

    let dict = PyDict::new(py);
    dict.set_item("genes", genes_py)?;
    dict.set_item("metrics", metrics)?;
    dict.set_item("feature_names", features.names)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (
    open,
    high,
    low,
    close,
    timestamps=None,
    volume=None,
    population=24000,
    generations=200,
    include_raw=true,
    elite_fraction=0.05,
    survivor_fraction=0.10,
    immigrant_fraction=0.20,
    parent_selection="rank",
    survivor_selection="rank",
    selection_temperature=0.75,
    tournament_size=4,
    sigma=0.5,
    crossover_rate=0.35,
    threshold_scale=0.10,
    threshold_margin=0.02,
    threshold_clip=0.30,
    window_bars=190080,
    segments=4,
    min_trades_per_day=1.0,
    trade_penalty=25.0,
    dd_limit=0.04,
    dd_penalty=200.0,
    robust_weight=0.2,
    pos_window_fraction=0.5,
    pos_penalty=15.0,
    chunk_size=2048,
    devices=None
))]
#[allow(clippy::too_many_arguments)]
pub fn search_evolve_gpu_ohlcv(
    py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    timestamps: Option<PyReadonlyArray1<i64>>,
    volume: Option<PyReadonlyArray1<f64>>,
    population: usize,
    generations: usize,
    include_raw: bool,
    elite_fraction: f64,
    survivor_fraction: f64,
    immigrant_fraction: f64,
    parent_selection: &str,
    survivor_selection: &str,
    selection_temperature: f64,
    tournament_size: usize,
    sigma: f64,
    crossover_rate: f64,
    threshold_scale: f64,
    threshold_margin: f64,
    threshold_clip: f64,
    window_bars: usize,
    segments: usize,
    min_trades_per_day: f64,
    trade_penalty: f64,
    dd_limit: f64,
    dd_penalty: f64,
    robust_weight: f64,
    pos_window_fraction: f64,
    pos_penalty: f64,
    chunk_size: usize,
    devices: Option<Vec<i64>>,
) -> PyResult<Py<PyAny>> {
    let ohlcv = build_ohlcv(
        &open,
        &high,
        &low,
        &close,
        timestamps.as_ref(),
        volume.as_ref(),
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let defaults = GpuDiscoveryConfig::default();

    let config = GpuDiscoveryConfig {
        population: population.max(16),
        generations: generations.max(1),
        elite_fraction: elite_fraction.clamp(0.01, 0.50),
        survivor_fraction: survivor_fraction.clamp(0.0, 0.95),
        immigrant_fraction: immigrant_fraction.clamp(0.0, 0.95),
        parent_selection: parse_parent_selection(parent_selection)?,
        survivor_selection: parse_survivor_selection(survivor_selection)?,
        selection_temperature: selection_temperature.max(1e-3),
        tournament_size: tournament_size.max(2),
        sigma: sigma.max(0.01),
        crossover_rate: crossover_rate.clamp(0.0, 1.0),
        threshold_scale: threshold_scale.max(0.001),
        threshold_margin: threshold_margin.max(0.0),
        threshold_clip: threshold_clip.max(0.01),
        window_bars: window_bars.max(128),
        segments: segments.max(1),
        min_trades_per_day: min_trades_per_day.max(0.0),
        trade_penalty: trade_penalty.max(0.0),
        dd_limit: dd_limit.clamp(0.0, 1.0),
        dd_penalty: dd_penalty.max(0.0),
        robust_weight: robust_weight.max(0.0),
        pos_window_fraction: pos_window_fraction.clamp(0.0, 1.0),
        pos_penalty: pos_penalty.max(0.0),
        chunk_size: chunk_size.max(64),
        devices: devices.unwrap_or(defaults.devices),
        backend: defaults.backend,
        precision: defaults.precision,
    };

    let (features, result) = py
        .detach(|| {
            let prof = if include_raw {
                FeatureProfile::Full
            } else {
                FeatureProfile::Standard
            };
            let features = compute_hpc_feature_frame(&ohlcv, prof)
                .map_err(|e| format!("Feature computation failed: {}", e))?;
            let result = {
                #[cfg(feature = "gpu")]
                {
                    let frames_vec = vec![features.clone()];
                    run_gpu_discovery(frames_vec, features.names.clone(), ohlcv.clone(), &config)
                        .map_err(|e| format!("GPU search failed: {}", e))?
                }
                #[cfg(not(feature = "gpu"))]
                {
                    let frames_vec = vec![features.clone()];
                    run_gpu_discovery(frames_vec, features.names.clone(), ohlcv.clone(), &config)
                        .map_err(|e| format!("GPU search failed: {}", e))?
                }
            };
            Ok::<_, String>((features, result))
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    let dict = PyDict::new(py);
    dict.set_item("genomes", result.genomes)?;
    dict.set_item("fitness", result.fitness)?;
    dict.set_item("feature_names", features.names)?;
    dict.set_item("timeframes", result.timeframes)?;
    dict.set_item("gpu", result.used_gpu)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, timestamps=None, volume=None, population=5000, generations=50, max_indicators=8))]
#[allow(clippy::too_many_arguments)]
pub fn search_discovery_ohlcv(
    py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    timestamps: Option<PyReadonlyArray1<i64>>,
    volume: Option<PyReadonlyArray1<f64>>,
    population: usize,
    generations: usize,
    max_indicators: usize,
) -> PyResult<Py<PyAny>> {
    let ohlcv = build_ohlcv(
        &open,
        &high,
        &low,
        &close,
        timestamps.as_ref(),
        volume.as_ref(),
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let config = forex_search::DiscoveryConfig {
        population,
        generations,
        max_indicators,
        ..forex_search::DiscoveryConfig::default()
    };

    let (features, result) = py
        .detach(|| {
            let features = compute_hpc_feature_frame(&ohlcv, FeatureProfile::Standard)
                .map_err(|e| format!("Feature computation failed: {}", e))?;
            let result = forex_search::run_discovery_cycle(&features, &ohlcv, &config)
                .map_err(|e| format!("Discovery failed: {}", e))?;
            Ok::<_, String>((features, result))
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    let genes_py = pythonize_genes(py, &result.portfolio)?;

    let dict = PyDict::new(py);
    dict.set_item("genes", genes_py)?;
    dict.set_item("feature_names", features.names)?;
    Ok(dict.into_any().unbind())
}

pub fn discovery_gene_key(gene: &forex_search::Gene) -> String {
    let sid = gene.strategy_id.trim();
    let strategy_prefix = if sid.is_empty() {
        "id:".to_string()
    } else {
        format!("id:{sid}|")
    };
    let weights = gene
        .weights
        .iter()
        .map(|weight| fmt_f32(*weight))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{strategy_prefix}sig:idx={:?}|w=[{}]|lt={}|st={}|ob={}|fvg={}|liq={}|mtf={}|premium={}|inducement={}|bos={}|choch={}|eqh={}|eql={}|disp={}|tp={}|sl={}",
        gene.indices,
        weights,
        fmt_f32(gene.long_threshold),
        fmt_f32(gene.short_threshold),
        gene.use_ob as u8,
        gene.use_fvg as u8,
        gene.use_liq_sweep as u8,
        gene.mtf_confirmation as u8,
        gene.use_premium_discount as u8,
        gene.use_inducement as u8,
        gene.use_bos as u8,
        gene.use_choch as u8,
        gene.use_eqh as u8,
        gene.use_eql as u8,
        gene.use_displacement as u8,
        fmt_f64(gene.tp_pips),
        fmt_f64(gene.sl_pips)
    )
}

pub fn rank_dedupe_genes(genes: &[forex_search::Gene]) -> Vec<forex_search::Gene> {
    let mut ranked = genes.to_vec();
    ranked.sort_by(|a, b| {
        let fitness = b
            .fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal);
        if fitness != std::cmp::Ordering::Equal {
            return fitness;
        }
        let sharpe = b
            .sharpe_ratio
            .partial_cmp(&a.sharpe_ratio)
            .unwrap_or(std::cmp::Ordering::Equal);
        if sharpe != std::cmp::Ordering::Equal {
            return sharpe;
        }
        b.win_rate
            .partial_cmp(&a.win_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::with_capacity(ranked.len());
    for gene in ranked {
        let key = discovery_gene_key(&gene);
        if seen.insert(key) {
            out.push(gene);
        }
    }
    out
}

pub fn select_ranked_discovery_genes(
    candidates: &[forex_search::Gene],
    min_keep: usize,
    cap: usize,
    cfg: &forex_search::FilteringConfig,
) -> (Vec<forex_search::Gene>, usize, usize) {
    let ranked_all = rank_dedupe_genes(candidates);
    let ranked_filtered_input: Vec<forex_search::Gene> = candidates
        .iter()
        .filter(|g| g.passes_filter(cfg))
        .cloned()
        .collect();
    let ranked_filtered = rank_dedupe_genes(&ranked_filtered_input);

    let mut selected = ranked_filtered.clone();
    if min_keep > 0 && selected.len() < min_keep {
        let mut seen: HashSet<String> = selected.iter().map(discovery_gene_key).collect();
        for gene in &ranked_all {
            let key = discovery_gene_key(gene);
            if seen.insert(key) {
                selected.push(gene.clone());
                if selected.len() >= min_keep {
                    break;
                }
            }
        }
    }

    if selected.is_empty() {
        selected = ranked_all.clone();
    }
    if cap > 0 && selected.len() > cap {
        selected.truncate(cap);
    }

    (selected, ranked_filtered.len(), ranked_all.len())
}

#[cfg(test)]
mod tests {
    use super::discovery_gene_key;
    use forex_search::Gene;

    #[test]
    fn discovery_gene_key_distinguishes_weight_changes() {
        let a = Gene {
            indices: vec![1, 2, 3],
            weights: vec![0.2, 0.3, 0.5],
            long_threshold: 0.12345678,
            short_threshold: -0.12345678,
            tp_pips: 12.5,
            sl_pips: 8.0,
            ..Gene::default()
        };

        let mut b = a.clone();
        b.weights = vec![0.5, 0.3, 0.2];

        assert_ne!(discovery_gene_key(&a), discovery_gene_key(&b));
    }

    #[test]
    fn discovery_gene_key_includes_strategy_id_without_collapsing_structure() {
        let a = Gene {
            strategy_id: "alpha".to_string(),
            indices: vec![1, 2],
            weights: vec![0.4, 0.6],
            ..Gene::default()
        };

        let mut b = a.clone();
        b.weights = vec![0.6, 0.4];

        assert_ne!(discovery_gene_key(&a), discovery_gene_key(&b));
        assert!(discovery_gene_key(&a).starts_with("id:alpha|"));
    }
}
