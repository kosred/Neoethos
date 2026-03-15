use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::HashSet;
use pythonize::pythonize;
use forex_data::compute_talib_feature_frame;
use forex_search::{evolve_search, run_gpu_discovery, GpuDiscoveryConfig};
use crate::utils::{build_ohlcv};

#[pyfunction]
#[pyo3(signature = (open, high, low, close, timestamps=None, volume=None, population=64, generations=20, max_indicators=12, include_raw=true))]
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
    .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;

    let (features, result) = py
        .detach(|| {
            let features = compute_talib_feature_frame(&ohlcv, include_raw)
                .map_err(|e| format!("Feature computation failed: {}", e))?;
            let result = evolve_search(&features, &ohlcv, population, generations, max_indicators)
                .map_err(|e| format!("Search failed: {}", e))?;
            Ok::<_, String>((features, result))
        })
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

    let metrics: Vec<Vec<f64>> = result.metrics.iter().map(|m| m.to_vec()).collect();
    let genes_py: Vec<Py<PyAny>> = result
        .genes
        .iter()
        .map(|g| {
            pythonize(py, g)
                .map(|obj| obj.into())
                .unwrap_or_else(|_| py.None())
        })
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("genes", genes_py)?;
    dict.set_item("metrics", metrics)?;
    dict.set_item("feature_names", features.names)?;
    Ok(dict.into_any().into())
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
    .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;

    let mut config = GpuDiscoveryConfig::default();
    config.population = population.max(16);
    config.generations = generations.max(1);
    config.elite_fraction = elite_fraction.clamp(0.01, 0.50);
    config.sigma = sigma.max(0.01);
    config.crossover_rate = crossover_rate.clamp(0.0, 1.0);
    config.threshold_scale = threshold_scale.max(0.001);
    config.threshold_margin = threshold_margin.max(0.0);
    config.threshold_clip = threshold_clip.max(0.01);
    config.window_bars = window_bars.max(128);
    config.segments = segments.max(1);
    config.min_trades_per_day = min_trades_per_day.max(0.0);
    config.trade_penalty = trade_penalty.max(0.0);
    config.dd_limit = dd_limit.clamp(0.0, 1.0);
    config.dd_penalty = dd_penalty.max(0.0);
    config.robust_weight = robust_weight.max(0.0);
    config.pos_window_fraction = pos_window_fraction.clamp(0.0, 1.0);
    config.pos_penalty = pos_penalty.max(0.0);
    config.chunk_size = chunk_size.max(64);
    config.devices = devices.unwrap_or_default();

    let (features, result) = py
        .detach(|| {
            let features = compute_talib_feature_frame(&ohlcv, include_raw)
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
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

    let dict = PyDict::new(py);
    dict.set_item("genomes", result.genomes)?;
    dict.set_item("fitness", result.fitness)?;
    dict.set_item("feature_names", features.names)?;
    dict.set_item("timeframes", result.timeframes)?;
    dict.set_item("gpu", true)?;
    Ok(dict.into_any().into())
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, timestamps=None, volume=None, population=5000, generations=50, max_indicators=8))]
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
    .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;

    let mut config = forex_search::DiscoveryConfig::default();
    config.population = population;
    config.generations = generations;
    config.max_indicators = max_indicators;

    let (features, result) = py
        .detach(|| {
            let features = compute_talib_feature_frame(&ohlcv, true)
                .map_err(|e| format!("Feature computation failed: {}", e))?;
            let result = forex_search::run_discovery_cycle(&features, &ohlcv, &config)
                .map_err(|e| format!("Discovery failed: {}", e))?;
            Ok::<_, String>((features, result))
        })
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

    let genes_py: Vec<Py<PyAny>> = result
        .portfolio
        .iter()
        .map(|g| {
            pythonize(py, g)
                .map(|obj| obj.into())
                .unwrap_or_else(|_| py.None())
        })
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("genes", genes_py)?;
    dict.set_item("feature_names", features.names)?;
    Ok(dict.into_any().into())
}

pub fn discovery_gene_key(gene: &forex_search::Gene) -> String {
    let sid = gene.strategy_id.trim();
    if !sid.is_empty() {
        return format!("id:{sid}");
    }
    format!(
        "sig:{:?}|{:.6}|{:.6}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{:.2}|{:.2}",
        gene.indices,
        gene.long_threshold,
        gene.short_threshold,
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
        gene.tp_pips,
        gene.sl_pips
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
