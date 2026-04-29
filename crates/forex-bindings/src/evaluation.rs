use crate::indicators::causal_tanh_zscore_column;
use crate::utils::{
    build_ohlcv, map_indicator_index, normalize_indicator_name, vec_from_py_f64, vec_from_py_i8,
    vec_from_py_i64,
};
use forex_data::{FeatureProfile, compute_hpc_feature_frame};
use forex_search::{EvaluationConfig, Gene, evaluate_genes};
use ndarray::Array2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::HashMap;
#[pyfunction]
#[pyo3(signature = (signals, close, initial_balance=10000.0, leverage=1.0))]
pub fn quick_backtest_metrics(
    signals: PyReadonlyArray1<'_, i8>,
    close: PyReadonlyArray1<'_, f64>,
    initial_balance: f64,
    leverage: f64,
) -> PyResult<(f64, f64, f64, i64)> {
    let sig = vec_from_py_i8(&signals);
    let cl = vec_from_py_f64(&close);
    let n = sig.len().min(cl.len());
    if n <= 1 {
        return Ok((0.0, 0.0, 0.0, 0));
    }

    let mut balance = initial_balance;
    let mut pnl_sum = 0.0;
    let mut wins = 0;
    let mut trades = 0;
    let mut correct = 0;

    for i in 0..(n - 1) {
        let s = sig[i];
        if s == 0 {
            continue;
        }
        let ret = (cl[i + 1] - cl[i]) / cl[i];
        trades += 1;
        let trade_pnl = balance * leverage * (s as f64) * ret;
        pnl_sum += trade_pnl;
        balance += trade_pnl;

        if (s > 0 && ret > 0.0) || (s < 0 && ret < 0.0) {
            correct += 1;
        }
        if trade_pnl > 0.0 {
            wins += 1;
        }
    }

    let steps = trades as f64;
    let accuracy = if steps > 0.0 {
        correct as f64 / steps
    } else {
        0.0
    };
    let win_rate = if steps > 0.0 {
        wins as f64 / steps
    } else {
        0.0
    };
    Ok((accuracy, pnl_sum, win_rate, trades as i64))
}

#[pyfunction]
#[pyo3(signature = (
    open,
    high,
    low,
    close,
    indicator_names,
    weights=None,
    long_threshold=0.66,
    short_threshold=-0.66,
    timestamps=None,
    volume=None,
    include_raw=true
))]
#[allow(clippy::too_many_arguments)]
pub fn fast_evaluate_strategy(
    py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    indicator_names: Vec<String>,
    weights: Option<Vec<f64>>,
    long_threshold: f32,
    short_threshold: f32,
    timestamps: Option<PyReadonlyArray1<i64>>,
    volume: Option<PyReadonlyArray1<f64>>,
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

    let prof = if include_raw {
        FeatureProfile::Full
    } else {
        FeatureProfile::Standard
    };
    let features = compute_hpc_feature_frame(&ohlcv, prof).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Feature computation failed: {}",
            e
        ))
    })?;
    let mut indices = Vec::new();
    let mut w_vec = Vec::new();
    for (idx, name) in indicator_names.iter().enumerate() {
        if let Some(col) = map_indicator_index(name, &features.names) {
            indices.push(col);
            let w = weights
                .as_ref()
                .and_then(|v| v.get(idx))
                .copied()
                .unwrap_or(1.0);
            w_vec.push(w as f32);
        }
    }
    if indices.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No indicators found in feature frame",
        ));
    }

    let gene = Gene {
        indices,
        weights: w_vec,
        long_threshold,
        short_threshold,
        ..Default::default()
    };

    let config = EvaluationConfig::default();
    let res_vec = evaluate_genes(&features, &ohlcv, &[gene], &config).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Evaluation failed: {}", e))
    })?;
    let result = res_vec[0];

    let dict = PyDict::new(py);
    dict.set_item("fitness", result[0])?;
    dict.set_item("sharpe_ratio", result[1])?;
    dict.set_item("win_rate", result[4])?;
    dict.set_item("max_drawdown", result[3])?;
    dict.set_item("trades_count", result[8])?;
    dict.set_item("consistency", result[9])?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (
    open,
    high,
    low,
    close,
    strategies,
    timestamps=None,
    volume=None,
    include_raw=true
))]
#[allow(clippy::too_many_arguments)]
pub fn batch_evaluate_strategies(
    py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    strategies: Vec<Py<PyAny>>,
    timestamps: Option<PyReadonlyArray1<i64>>,
    volume: Option<PyReadonlyArray1<f64>>,
    include_raw: bool,
) -> PyResult<Vec<Py<PyAny>>> {
    let ohlcv = build_ohlcv(
        &open,
        &high,
        &low,
        &close,
        timestamps.as_ref(),
        volume.as_ref(),
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let mut genes = Vec::with_capacity(strategies.len());
    for s in strategies {
        let gene: Gene = pythonize::depythonize(s.bind(py)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse strategy: {}",
                e
            ))
        })?;
        genes.push(gene);
    }

    let prof = if include_raw {
        FeatureProfile::Full
    } else {
        FeatureProfile::Standard
    };
    let features = compute_hpc_feature_frame(&ohlcv, prof).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Feature computation failed: {}",
            e
        ))
    })?;

    let config = EvaluationConfig::default();
    let results = evaluate_genes(&features, &ohlcv, &genes, &config).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Evaluation failed: {}", e))
    })?;

    let mut out = Vec::new();
    for res in results {
        let dict = PyDict::new(py);
        dict.set_item("fitness", res[0])?;
        dict.set_item("sharpe_ratio", res[1])?;
        dict.set_item("win_rate", res[4])?;
        dict.set_item("max_drawdown", res[3])?;
        dict.set_item("trades_count", res[8])?;
        out.push(dict.into_any().unbind());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    open,
    high,
    low,
    close,
    indicator_sets,
    weight_sets=None,
    long_thresholds=None,
    short_thresholds=None,
    timestamps=None,
    volume=None,
    include_raw=true
))]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_population_vector_ta_ohlcv(
    py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    indicator_sets: Vec<Vec<String>>,
    weight_sets: Option<Vec<Vec<f64>>>,
    long_thresholds: Option<Vec<f64>>,
    short_thresholds: Option<Vec<f64>>,
    timestamps: Option<PyReadonlyArray1<i64>>,
    volume: Option<PyReadonlyArray1<f64>>,
    include_raw: bool,
) -> PyResult<Vec<Py<PyAny>>> {
    let ohlcv = build_ohlcv(
        &open,
        &high,
        &low,
        &close,
        timestamps.as_ref(),
        volume.as_ref(),
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let prof = if include_raw {
        FeatureProfile::Full
    } else {
        FeatureProfile::Standard
    };
    let features = compute_hpc_feature_frame(&ohlcv, prof).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Feature computation failed: {}",
            e
        ))
    })?;
    let mut genes = Vec::new();
    for (i, indicators) in indicator_sets.iter().enumerate() {
        let mut indices = Vec::new();
        let mut w_vec = Vec::new();
        for (j, name) in indicators.iter().enumerate() {
            if let Some(col) = map_indicator_index(name, &features.names) {
                indices.push(col);
                let w = weight_sets
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|v| v.get(j))
                    .copied()
                    .unwrap_or(1.0);
                w_vec.push(w as f32);
            }
        }
        let long = long_thresholds
            .as_ref()
            .and_then(|v| v.get(i))
            .copied()
            .unwrap_or(0.66);
        let short = short_thresholds
            .as_ref()
            .and_then(|v| v.get(i))
            .copied()
            .unwrap_or(-0.66);

        genes.push(Gene {
            indices,
            weights: w_vec,
            long_threshold: long as f32,
            short_threshold: short as f32,
            ..Default::default()
        });
    }

    let config = EvaluationConfig::default();
    let results = evaluate_genes(&features, &ohlcv, &genes, &config).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Evaluation failed: {}", e))
    })?;

    let mut out = Vec::new();
    for res in results {
        let dict = PyDict::new(py);
        dict.set_item("fitness", res[0])?;
        dict.set_item("sharpe_ratio", res[1])?;
        dict.set_item("win_rate", res[4])?;
        dict.set_item("max_drawdown", res[3])?;
        dict.set_item("trades_count", res[8])?;
        out.push(dict.into_any().unbind());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (features, ohlcv_dict, genes_dict))]
pub fn evaluate_population_core_py(
    py: Python,
    features: Py<PyAny>,
    ohlcv_dict: Py<PyAny>,
    genes_dict: Py<PyAny>,
) -> PyResult<Vec<Py<PyAny>>> {
    let frame_res = {
        let features_bound = features.bind(py);
        let data: PyReadonlyArray2<f32> = features_bound.getattr("data")?.extract()?;
        let names: Vec<String> = features_bound.getattr("names")?.extract()?;
        let timestamps: PyReadonlyArray1<i64> = features_bound.getattr("timestamps")?.extract()?;
        forex_data::FeatureFrame {
            data: data.as_array().to_owned(),
            names,
            timestamps: vec_from_py_i64(&timestamps),
        }
    };

    let ohlcv_res = {
        let ohlcv_bound = ohlcv_dict.bind(py);
        let open: PyReadonlyArray1<f64> = ohlcv_bound.getattr("open")?.extract()?;
        let high: PyReadonlyArray1<f64> = ohlcv_bound.getattr("high")?.extract()?;
        let low: PyReadonlyArray1<f64> = ohlcv_bound.getattr("low")?.extract()?;
        let close: PyReadonlyArray1<f64> = ohlcv_bound.getattr("close")?.extract()?;
        let timestamps: Option<PyReadonlyArray1<i64>> =
            ohlcv_bound.getattr("timestamp")?.extract().ok();
        let volume: Option<PyReadonlyArray1<f64>> = ohlcv_bound.getattr("volume")?.extract().ok();
        build_ohlcv(
            &open,
            &high,
            &low,
            &close,
            timestamps.as_ref(),
            volume.as_ref(),
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?
    };

    let genes: Vec<Gene> = {
        let list: Vec<Py<PyAny>> = genes_dict.bind(py).extract()?;
        let mut out = Vec::with_capacity(list.len());
        for item in list {
            let gene: Gene = pythonize::depythonize(item.bind(py)).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid gene in genes_dict: {}",
                    e
                ))
            })?;
            out.push(gene);
        }
        out
    };

    let config = EvaluationConfig::default();
    let results = evaluate_genes(&frame_res, &ohlcv_res, &genes, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut out = Vec::new();
    for res in results {
        let dict = PyDict::new(py);
        dict.set_item("fitness", res[0])?;
        dict.set_item("sharpe_ratio", res[1])?;
        dict.set_item("win_rate", res[4])?;
        dict.set_item("max_drawdown", res[3])?;
        dict.set_item("trades_count", res[8])?;
        dict.set_item("consistency", res[9])?;
        out.push(dict.into_any().unbind());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (prop_signals, prop_returns, prop_masks))]
pub fn aggregate_prop_score_metrics<'py>(
    py: Python<'py>,
    prop_signals: PyReadonlyArray2<'py, i8>,
    prop_returns: PyReadonlyArray2<'py, f64>,
    prop_masks: PyReadonlyArray2<'py, bool>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let sigs = prop_signals.as_array();
    let rets = prop_returns.as_array();
    let masks = prop_masks.as_array();
    let n_rows = sigs.nrows();
    let n_cols = sigs.ncols();

    let mut out_score = vec![0.0_f64; n_rows];
    let mut out_certainty = vec![0.0_f64; n_rows];

    for r in 0..n_rows {
        let mut sum_score = 0.0;
        let mut sum_weight = 0.0;
        for c in 0..n_cols {
            if masks[(r, c)] {
                let s = sigs[(r, c)] as f64;
                let ret = rets[(r, c)];
                sum_score += s * ret;
                sum_weight += 1.0;
            }
        }
        if sum_weight > 0.0 {
            out_score[r] = sum_score / sum_weight;
            out_certainty[r] = sum_weight / (n_cols as f64);
        }
    }

    Ok((
        out_score.to_pyarray(py).into_any().unbind(),
        out_certainty.to_pyarray(py).into_any().unbind(),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    open,
    high,
    low,
    close,
    indicator_sets,
    weight_sets=None,
    long_thresholds=None,
    short_thresholds=None,
    timestamps=None,
    volume=None,
    include_raw=false,
    causal_min_bars=30
))]
#[allow(clippy::too_many_arguments)]
pub fn vector_ta_bulk_signals_ohlcv(
    py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    indicator_sets: Vec<Vec<String>>,
    weight_sets: Option<Vec<Vec<f64>>>,
    long_thresholds: Option<Vec<f64>>,
    short_thresholds: Option<Vec<f64>>,
    timestamps: Option<PyReadonlyArray1<i64>>,
    volume: Option<PyReadonlyArray1<f64>>,
    include_raw: bool,
    causal_min_bars: usize,
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

    let prof = if include_raw {
        FeatureProfile::Full
    } else {
        FeatureProfile::Standard
    };
    let frame = compute_hpc_feature_frame(&ohlcv, prof).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Feature computation failed: {}",
            e
        ))
    })?;
    let n_rows = frame.data.nrows();
    let n_genes = indicator_sets.len();
    let mut out = Array2::<i8>::zeros((n_rows, n_genes));
    if n_rows == 0 || n_genes == 0 {
        return Ok(out.to_pyarray(py).into_any().unbind());
    }

    let mut idx_map: HashMap<String, usize> = HashMap::with_capacity(frame.names.len() * 2);
    for (idx, name) in frame.names.iter().enumerate() {
        let key = normalize_indicator_name(name.strip_prefix("ta_").unwrap_or(name));
        if !key.is_empty() {
            idx_map.entry(key).or_insert(idx);
        }
    }

    for g in 0..n_genes {
        let indicators = &indicator_sets[g];
        if indicators.is_empty() {
            continue;
        }
        let weights = weight_sets.as_ref().and_then(|v| v.get(g));
        let long_thr = long_thresholds
            .as_ref()
            .and_then(|v| v.get(g))
            .copied()
            .unwrap_or(0.66);
        let short_thr = short_thresholds
            .as_ref()
            .and_then(|v| v.get(g))
            .copied()
            .unwrap_or(-0.66);

        let mut votes = vec![0.0_f64; n_rows];
        let mut weight_total = 0.0_f64;

        for (k, indicator) in indicators.iter().enumerate() {
            let norm = normalize_indicator_name(indicator);
            let col_idx = idx_map
                .get(&norm)
                .copied()
                .or_else(|| map_indicator_index(indicator, &frame.names));
            let Some(col_idx) = col_idx else {
                continue;
            };

            let w = weights.and_then(|w| w.get(k)).copied().unwrap_or(1.0);
            if !w.is_finite() || w.abs() <= 0.0 {
                continue;
            }

            let score = causal_tanh_zscore_column(&frame.data, col_idx, causal_min_bars);
            for r in 0..n_rows {
                votes[r] += w * score[r];
            }
            weight_total += w.abs();
        }

        let denom = if weight_total > 0.0 {
            weight_total
        } else {
            1.0
        };
        for r in 0..n_rows {
            let combined = votes[r] / denom;
            out[(r, g)] = if combined > long_thr {
                1
            } else if combined < short_thr {
                -1
            } else {
                0
            };
        }
    }

    Ok(out.to_pyarray(py).into_any().unbind())
}

#[allow(unused_variables)]
#[pyfunction]
#[pyo3(signature = (close, timestamps=None))]
pub fn triple_barrier_labels<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timestamps: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Py<PyAny>> {
    let cl = vec_from_py_f64(&close);
    let n = cl.len();
    let mut labels = vec![0_i8; n];
    if n < 30 {
        return Ok(labels.to_pyarray(py).into_any().unbind());
    }

    let lookahead = 20;
    let threshold = 0.001;

    for i in 0..(n - lookahead) {
        let current = cl[i];
        let mut hit = 0;
        for j in 1..=lookahead {
            let ret = (cl[i + j] - current) / current;
            if ret >= threshold {
                hit = 1;
                break;
            } else if ret <= -threshold {
                hit = -1;
                break;
            }
        }
        labels[i] = hit;
    }

    Ok(labels.to_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (journal_df))]
pub fn trade_journal_metrics(py: Python, journal_df: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let pnls = extract_trade_pnls(journal_df.bind(py))?;
    if pnls.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "trade_journal_metrics requires at least one trade pnl value",
        ));
    }

    let trades = pnls.len();
    let wins = pnls.iter().filter(|pnl| **pnl > 0.0).count();
    let losses = pnls.iter().filter(|pnl| **pnl < 0.0).count();
    let gross_profit: f64 = pnls.iter().copied().filter(|pnl| *pnl > 0.0).sum();
    let gross_loss: f64 = pnls
        .iter()
        .copied()
        .filter(|pnl| *pnl < 0.0)
        .map(f64::abs)
        .sum();
    let total_pnl: f64 = pnls.iter().sum();
    let mean = total_pnl / trades as f64;
    let variance = if trades > 1 {
        pnls.iter()
            .map(|pnl| {
                let d = *pnl - mean;
                d * d
            })
            .sum::<f64>()
            / (trades as f64 - 1.0)
    } else {
        0.0
    };
    let std = variance.sqrt();
    let sharpe = if std > 0.0 {
        (mean / std) * (trades as f64).sqrt()
    } else {
        0.0
    };
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let mut equity = 0.0;
    let mut peak = 0.0;
    let mut max_drawdown = 0.0;
    let mut current_loss_streak = 0usize;
    let mut max_consec_losses = 0usize;
    for pnl in &pnls {
        equity += *pnl;
        if equity > peak {
            peak = equity;
        }
        max_drawdown = f64::max(max_drawdown, peak - equity);
        if *pnl < 0.0 {
            current_loss_streak += 1;
            max_consec_losses = max_consec_losses.max(current_loss_streak);
        } else if *pnl > 0.0 {
            current_loss_streak = 0;
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("sharpe", sharpe)?;
    dict.set_item("win_rate", wins as f64 / trades as f64)?;
    dict.set_item("trades", trades)?;
    dict.set_item("wins", wins)?;
    dict.set_item("losses", losses)?;
    dict.set_item("total_pnl", total_pnl)?;
    dict.set_item("avg_pnl", mean)?;
    dict.set_item("profit_factor", profit_factor)?;
    dict.set_item("expectancy", mean)?;
    dict.set_item("max_drawdown", max_drawdown)?;
    dict.set_item("max_consec_losses", max_consec_losses)?;
    Ok(dict.into_any().unbind())
}

fn extract_trade_pnls(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Some(values) = extract_numeric_vec(obj)? {
        return Ok(values);
    }

    if let Ok(dict) = obj.cast::<PyDict>() {
        for key in ["pnl", "profit", "net_pnl", "pnl_pct"] {
            if let Some(value) = dict.get_item(key)?
                && let Some(values) = extract_numeric_vec(&value)?
            {
                return Ok(values);
            }
        }
    }

    for key in ["pnl", "profit", "net_pnl", "pnl_pct"] {
        if let Ok(value) = obj.getattr(key)
            && let Some(values) = extract_numeric_vec(&value)?
        {
            return Ok(values);
        }
    }

    if let Ok(iter) = obj.try_iter() {
        let mut values = Vec::new();
        for item in iter {
            if let Some(pnl) = extract_trade_pnl(&item?)? {
                values.push(pnl);
            }
        }
        if !values.is_empty() {
            return Ok(values);
        }
    }

    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "could not extract trade pnl values from journal; expected a pnl/profit sequence, dict, dataframe column, or iterable of trade records",
    ))
}

fn extract_numeric_vec(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<f64>>> {
    if let Ok(values) = obj.extract::<Vec<f64>>() {
        return Ok(Some(values));
    }
    for method in ["to_list", "tolist"] {
        if let Ok(callable) = obj.getattr(method) {
            let values = callable.call0()?;
            if let Ok(values) = values.extract::<Vec<f64>>() {
                return Ok(Some(values));
            }
        }
    }
    Ok(None)
}

fn extract_trade_pnl(obj: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if let Ok(value) = obj.extract::<f64>() {
        return Ok(Some(value));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        for key in ["pnl", "profit", "net_pnl", "pnl_pct"] {
            if let Some(value) = dict.get_item(key)? {
                return Ok(Some(value.extract::<f64>()?));
            }
        }
    }
    for key in ["pnl", "profit", "net_pnl", "pnl_pct"] {
        if let Ok(value) = obj.getattr(key) {
            return Ok(Some(value.extract::<f64>()?));
        }
    }
    Ok(None)
}

#[allow(unused_variables)]
#[pyfunction]
#[pyo3(signature = (open, high, low, close, window=500))]
pub fn infer_stop_target_pips_ohlcv(
    _py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<(f64, f64)> {
    let cl = vec_from_py_f64(&close);
    let hi = vec_from_py_f64(&high);
    let lo = vec_from_py_f64(&low);
    let n = cl.len();
    if n < window {
        return Ok((10.0, 30.0));
    }

    let mut atr_sum = 0.0;
    for i in (n - window)..n {
        let tr = (hi[i] - lo[i])
            .max((hi[i] - cl[i - 1]).abs())
            .max((lo[i] - cl[i - 1]).abs());
        atr_sum += tr;
    }
    let atr = atr_sum / window as f64;

    let sl_pips = (atr * 1.5) / 0.0001;
    let tp_pips = (atr * 4.5) / 0.0001;

    Ok((sl_pips, tp_pips))
}

pub fn _quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (q * (sorted.len() - 1) as f64) as usize;
    sorted[idx]
}

pub fn _mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

pub fn _sum(v: &[f64]) -> f64 {
    v.iter().sum()
}

pub fn _max(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
