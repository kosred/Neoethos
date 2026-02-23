use forex_core::system::HardwareProbe;
use forex_data::{
    compute_talib_feature_frame, ensure_timeframes_with_resample, load_symbol_dataset,
    load_symbol_dataset_with_timeframes, prepare_multitimeframe_features, FeatureCache, Ohlcv,
    SymbolDataset,
};
#[cfg(feature = "onnx")]
use forex_models::ONNXInferenceEngine;
use forex_search::{
    evolve_search,
    infer_stop_target_pips as infer_stop_target_pips_rs,
    run_gpu_discovery,
    run_discovery_cycle,
    DiscoveryConfig,
    GpuDiscoveryConfig,
    StopTargetSettings,
};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pythonize::pythonize;
use serde::Deserialize;
#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
#[cfg(feature = "onnx")]
use std::sync::Arc;
use std::sync::Mutex;
use std::fs;
use std::path::PathBuf;
#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
use std::path::Path;

#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
use forex_models::base::ExpertModel;
#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
use forex_models::tree_models::ParamValue;
#[cfg(feature = "lightgbm")]
use forex_models::tree_models::LightGBMExpert;

#[cfg(feature = "xgboost")]
use forex_models::tree_models::{XGBoostDARTExpert, XGBoostExpert, XGBoostRFExpert};

#[cfg(feature = "catboost")]
use forex_models::tree_models::{CatBoostExpert, CatBoostAltExpert};

#[pyclass]
struct ForexCore {
    probe: Mutex<HardwareProbe>,
}

#[pymethods]
impl ForexCore {
    #[new]
    fn new() -> Self {
        ForexCore {
            probe: Mutex::new(HardwareProbe::new()),
        }
    }

    fn detect_hardware(&self, py: Python) -> PyResult<Py<PyAny>> {
        let mut probe = self.probe.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let profile = probe.detect();
        let py_profile: Py<PyAny> = pythonize(py, &profile)?.into();
        Ok(py_profile)
    }
}

#[cfg(feature = "onnx")]
#[pyclass]
struct ModelEngine {
    engine: Arc<Mutex<ONNXInferenceEngine>>,
}

#[cfg(feature = "onnx")]
#[pymethods]
impl ModelEngine {
    #[new]
    fn new() -> PyResult<Self> {
        let engine = ONNXInferenceEngine::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to init engine: {}",
                e
            ))
        })?;
        Ok(ModelEngine {
            engine: Arc::new(Mutex::new(engine)),
        })
    }

    fn load_models(&self, py: Python, path: &str) -> PyResult<()> {
        let path = path.to_string();
        let result: Result<(), String> = py.detach(|| {
            let mut engine = self
                .engine
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;
            engine
                .load_models(&path)
                .map_err(|e| format!("Failed to load models: {}", e))?;
            Ok(())
        });
        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        model_name: &str,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();
        let model_name = model_name.to_string();
        let prediction: Array2<f32> = py
            .detach(|| {
                let engine = self
                    .engine
                    .lock()
                    .map_err(|e| format!("Lock poisoned: {}", e))?;
                engine
                    .predict_proba(&model_name, &features_array)
                    .map_err(|e| format!("Prediction failed: {}", e))
            })
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

        Ok(prediction.into_pyarray(py))
    }
}

#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
fn dataframe_from_ndarray(features: &Array2<f64>) -> Result<DataFrame, String> {
    let mut df_data: Vec<Column> = Vec::with_capacity(features.ncols());
    for col_idx in 0..features.ncols() {
        let col_data: Vec<f64> = features.column(col_idx).iter().copied().collect();
        let name = format!("feature_{col_idx}");
        df_data.push(Series::new(name.into(), col_data).into());
    }
    DataFrame::new(df_data).map_err(|e| format!("DataFrame creation failed: {}", e))
}

#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
fn param_value_from_py(value: &Bound<'_, PyAny>) -> Option<ParamValue> {        
    if let Ok(v) = value.extract::<bool>() {
        return Some(ParamValue::Bool(v));
    }
    if let Ok(v) = value.extract::<i64>() {
        return Some(ParamValue::Int(v as i32));
    }
    if let Ok(v) = value.extract::<f64>() {
        return Some(ParamValue::Float(v));
    }
    if let Ok(v) = value.extract::<String>() {
        return Some(ParamValue::String(v));
    }
    None
}

#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
fn params_from_py(
    params: Option<&Bound<'_, PyAny>>,
) -> Result<Option<HashMap<String, ParamValue>>, String> {
    let Some(obj) = params else {
        return Ok(None);
    };
    let dict = obj
        .cast::<PyDict>()
        .map_err(|_| "params must be a dict".to_string())?;
    let mut map = HashMap::new();
    for (k, v) in dict.iter() {
        let key: String = k
            .extract()
            .map_err(|_| "params keys must be strings".to_string())?;
        if let Some(val) = param_value_from_py(&v) {
            map.insert(key, val);
        }
    }
    Ok(Some(map))
}

fn vec_from_py_f64(arr: &PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

fn vec_from_py_i64(arr: &PyReadonlyArray1<i64>) -> Vec<i64> {
    arr.as_array().iter().copied().collect()
}

fn vec_from_py_i8(arr: &PyReadonlyArray1<i8>) -> Vec<i8> {
    arr.as_array().iter().copied().collect()
}

fn build_ohlcv(
    open: &PyReadonlyArray1<f64>,
    high: &PyReadonlyArray1<f64>,
    low: &PyReadonlyArray1<f64>,
    close: &PyReadonlyArray1<f64>,
    timestamps: Option<&PyReadonlyArray1<i64>>,
    volume: Option<&PyReadonlyArray1<f64>>,
) -> Result<Ohlcv, String> {
    let open_vec = vec_from_py_f64(open);
    let high_vec = vec_from_py_f64(high);
    let low_vec = vec_from_py_f64(low);
    let close_vec = vec_from_py_f64(close);

    let n = close_vec.len();
    if open_vec.len() != n || high_vec.len() != n || low_vec.len() != n {
        return Err("OHLC arrays must have equal length".to_string());
    }

    let timestamp_vec = match timestamps {
        Some(ts) => {
            let ts_vec = vec_from_py_i64(ts);
            if ts_vec.len() != n {
                return Err("timestamps length does not match close length".to_string());
            }
            Some(ts_vec)
        }
        None => Some((0..n as i64).collect()),
    };

    let volume_vec = match volume {
        Some(v) => {
            let vol = vec_from_py_f64(v);
            if vol.len() != n {
                return Err("volume length does not match close length".to_string());
            }
            Some(vol)
        }
        None => None,
    };

    Ok(Ohlcv {
        timestamp: timestamp_vec,
        open: open_vec,
        high: high_vec,
        low: low_vec,
        close: close_vec,
        volume: volume_vec,
    })
}

fn resolve_base_tf(dataset: &SymbolDataset, preferred: &str) -> String {        
    if dataset.frames.contains_key(preferred) {
        return preferred.to_string();
    }
    if dataset.frames.contains_key("M5") {
        return "M5".to_string();
    }
    if dataset.frames.contains_key("M1") {
        return "M1".to_string();
    }
    dataset
        .frames
        .keys()
        .next()
        .cloned()
        .unwrap_or_else(|| preferred.to_string())
}

fn normalize_indicator_name(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut underscore = false;
    for ch in input.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            underscore = false;
        } else if !underscore {
            out.push('_');
            underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn map_indicator_index(indicator: &str, feature_names: &[String]) -> Option<usize> {
    let norm = normalize_indicator_name(indicator);
    if norm.is_empty() {
        return None;
    }
    let prefix = format!("ta_{}", norm);
    for (idx, name) in feature_names.iter().enumerate() {
        if name == &prefix || name.starts_with(&format!("{}_", prefix)) {
            return Some(idx);
        }
    }
    None
}

#[derive(Debug, Deserialize)]
struct StrategySpec {
    indicators: Option<Vec<String>>,
    weights: Option<std::collections::HashMap<String, f64>>,
    long_threshold: Option<f64>,
    short_threshold: Option<f64>,
    strategy_id: Option<String>,
    tp_pips: Option<f64>,
    sl_pips: Option<f64>,
    use_ob: Option<bool>,
    use_fvg: Option<bool>,
    use_liq_sweep: Option<bool>,
    mtf_confirmation: Option<bool>,
    use_premium_discount: Option<bool>,
    use_inducement: Option<bool>,
}

fn load_strategy_specs(path: &PathBuf) -> Result<Vec<StrategySpec>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read strategy catalog {}: {}", path.display(), e))?;
    let value: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("Strategy catalog JSON invalid: {}", e))?;

    let genes_value = if value.is_array() {
        value
    } else if let Some(best) = value.get("best_genes") {
        best.clone()
    } else if let Some(best) = value.get("portfolio") {
        best.clone()
    } else if let Some(best) = value.get("genes") {
        best.clone()
    } else {
        serde_json::Value::Array(vec![])
    };

    let genes: Vec<StrategySpec> = serde_json::from_value(genes_value)
        .map_err(|e| format!("Failed to parse strategy catalog genes: {}", e))?;
    Ok(genes)
}

#[pyfunction]
#[pyo3(signature = (
    equity,
    risk_pct,
    stop_loss_pips,
    pip_value,
    max_lot_size=10.0,
    lot_step=0.01,
    min_lot=0.0
))]
fn compute_position_size_lots(
    equity: f64,
    risk_pct: f64,
    stop_loss_pips: f64,
    pip_value: f64,
    max_lot_size: f64,
    lot_step: f64,
    min_lot: f64,
) -> PyResult<f64> {
    if !equity.is_finite()
        || !risk_pct.is_finite()
        || !stop_loss_pips.is_finite()
        || !pip_value.is_finite()
        || equity <= 0.0
        || risk_pct <= 0.0
        || stop_loss_pips <= 0.0
        || pip_value <= 0.0
    {
        return Ok(0.0);
    }

    let risk_amount = equity * risk_pct.max(0.0);
    let denom = (stop_loss_pips * pip_value).max(1e-9);
    let mut lot_size = risk_amount / denom;
    if !lot_size.is_finite() || lot_size <= 0.0 {
        return Ok(0.0);
    }

    let step = if lot_step.is_finite() && lot_step > 0.0 {
        lot_step
    } else {
        0.01
    };
    lot_size = (lot_size / step).floor() * step;

    let cap = if max_lot_size.is_finite() && max_lot_size > 0.0 {
        max_lot_size
    } else {
        lot_size
    };
    let floor = if min_lot.is_finite() {
        min_lot.max(0.0)
    } else {
        0.0
    };
    lot_size = lot_size.max(floor).min(cap);
    if !lot_size.is_finite() || lot_size < floor {
        return Ok(0.0);
    }
    Ok(lot_size.max(0.0))
}

#[pyfunction]
#[pyo3(signature = (symbol, point=None, digits=None))]
fn pip_size_from_symbol(symbol: &str, point: Option<f64>, digits: Option<i64>) -> PyResult<f64> {
    let sym = symbol.to_ascii_uppercase();
    let pip_size = if let (Some(pt), Some(dig)) = (point, digits) {
        let ptv = if pt.is_finite() && pt > 0.0 { pt } else { 0.0001 };
        let d = dig.max(0) as i32;
        if sym.ends_with("JPY") || sym.starts_with("JPY") {
            ptv * if d >= 3 { 10.0 } else { 1.0 }
        } else if sym.starts_with("XAU") || sym.starts_with("XAG") {
            0.01
        } else if sym.contains("BTC") || sym.contains("ETH") || sym.contains("LTC") {
            1.0
        } else {
            ptv * if d >= 4 { 10.0 } else { 1.0 }
        }
    } else if sym.ends_with("JPY") || sym.starts_with("JPY") {
        0.01
    } else if sym.starts_with("XAU") || sym.starts_with("XAG") {
        0.01
    } else if sym.contains("BTC") || sym.contains("ETH") || sym.contains("LTC") {
        1.0
    } else {
        0.0001
    };
    Ok(pip_size.max(1e-9))
}

#[pyfunction]
#[pyo3(signature = (entry_price, signal, sl_pips, rr, pip_size))]
fn compute_order_prices(
    entry_price: f64,
    signal: i8,
    sl_pips: f64,
    rr: f64,
    pip_size: f64,
) -> PyResult<(f64, f64, f64)> {
    if !entry_price.is_finite()
        || !sl_pips.is_finite()
        || !rr.is_finite()
        || !pip_size.is_finite()
        || entry_price <= 0.0
        || sl_pips <= 0.0
        || rr <= 0.0
        || pip_size <= 0.0
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "entry_price/sl_pips/rr/pip_size must be finite positive values",
        ));
    }

    let sl_dist = sl_pips * pip_size;
    if signal > 0 {
        let sl = entry_price - sl_dist;
        let tp = entry_price + (rr * sl_dist);
        Ok((sl, tp, sl_dist))
    } else if signal < 0 {
        let sl = entry_price + sl_dist;
        let tp = entry_price - (rr * sl_dist);
        Ok((sl, tp, sl_dist))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "signal must be +1 or -1",
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (
    sl_pips,
    rr,
    spread_pips,
    slippage_pips,
    commission_per_lot,
    pip_value_per_lot,
    min_edge_multiple
))]
fn evaluate_trade_edge(
    sl_pips: f64,
    rr: f64,
    spread_pips: f64,
    slippage_pips: f64,
    commission_per_lot: f64,
    pip_value_per_lot: f64,
    min_edge_multiple: f64,
) -> PyResult<(bool, f64, f64)> {
    if !sl_pips.is_finite()
        || !rr.is_finite()
        || !spread_pips.is_finite()
        || !slippage_pips.is_finite()
        || !commission_per_lot.is_finite()
        || !pip_value_per_lot.is_finite()
        || !min_edge_multiple.is_finite()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "all inputs must be finite",
        ));
    }

    let commission_pips = commission_per_lot / pip_value_per_lot.max(1e-9);
    let total_cost_pips = (spread_pips.max(0.0) + slippage_pips.max(0.0) + commission_pips.max(0.0)).max(0.0);
    let expected_profit_pips = (sl_pips.max(0.0) * rr.max(0.0)).max(0.0);
    let passed = if min_edge_multiple <= 0.0 {
        true
    } else {
        expected_profit_pips >= (min_edge_multiple * total_cost_pips)
    };
    Ok((passed, expected_profit_pips, total_cost_pips))
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
    include_raw=false
))]
fn talib_bulk_signals_ohlcv(
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

    if let Some(ref wsets) = weight_sets {
        if wsets.len() != indicator_sets.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "weight_sets length must match indicator_sets length",
            ));
        }
    }
    if let Some(ref longs) = long_thresholds {
        if longs.len() != indicator_sets.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "long_thresholds length must match indicator_sets length",
            ));
        }
    }
    if let Some(ref shorts) = short_thresholds {
        if shorts.len() != indicator_sets.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "short_thresholds length must match indicator_sets length",
            ));
        }
    }

    let signals = py
        .detach(|| {
            let frame = compute_talib_feature_frame(&ohlcv, include_raw)
                .map_err(|e| format!("Feature computation failed: {}", e))?;
            let n_rows = frame.data.nrows();
            let n_genes = indicator_sets.len();
            let mut out = Array2::<i8>::zeros((n_rows, n_genes));
            if n_rows == 0 || n_genes == 0 {
                return Ok::<Array2<i8>, String>(out);
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
                    if norm.is_empty() {
                        continue;
                    }
                    let col_idx = idx_map
                        .get(&norm)
                        .copied()
                        .or_else(|| map_indicator_index(indicator, &frame.names));
                    let Some(col_idx) = col_idx else {
                        continue;
                    };

                    let w = weights
                        .and_then(|w| w.get(k))
                        .copied()
                        .unwrap_or(1.0);
                    if !w.is_finite() || w.abs() <= 0.0 {
                        continue;
                    }

                    let mut sum = 0.0_f64;
                    let mut count = 0usize;
                    for r in 0..n_rows {
                        let v = frame.data[(r, col_idx)] as f64;
                        if v.is_finite() {
                            sum += v;
                            count += 1;
                        }
                    }
                    if count == 0 {
                        continue;
                    }
                    let mean = sum / count as f64;
                    let mut var_sum = 0.0_f64;
                    for r in 0..n_rows {
                        let v = frame.data[(r, col_idx)] as f64;
                        if v.is_finite() {
                            let d = v - mean;
                            var_sum += d * d;
                        }
                    }
                    let std = (var_sum / count.max(1) as f64).sqrt();

                    for r in 0..n_rows {
                        let v = frame.data[(r, col_idx)] as f64;
                        let centered = if v.is_finite() {
                            if std <= 1e-9 {
                                v - mean
                            } else {
                                (v - mean) / std
                            }
                        } else {
                            0.0
                        };
                        votes[r] += w * centered.tanh();
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
            Ok::<Array2<i8>, String>(out)
        })
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

    Ok(signals.into_pyarray(py).into_any().into())
}

#[pyfunction]
#[pyo3(signature = (
    root,
    symbol,
    timeframes=None,
    resample_missing=true,
    base_tf="M1"
))]
fn load_symbol_frames(
    py: Python,
    root: String,
    symbol: String,
    timeframes: Option<Vec<String>>,
    resample_missing: bool,
    base_tf: &str,
) -> PyResult<Py<PyAny>> {
    let result: Result<SymbolDataset, String> = py.detach(|| {
        let tfs = timeframes.clone().unwrap_or_default();
        let mut merged = tfs.clone();
        if !merged.is_empty() && !merged.iter().any(|tf| tf.eq_ignore_ascii_case(&base_tf)) {
            merged.push(base_tf.to_string());
        }
        let dataset = if merged.is_empty() {
            load_symbol_dataset(&root, &symbol)
                .map_err(|e| format!("Load dataset failed: {}", e))?
        } else {
            let refs: Vec<&str> = merged.iter().map(|s| s.as_str()).collect();
            load_symbol_dataset_with_timeframes(&root, &symbol, &refs)
                .map_err(|e| format!("Load dataset failed: {}", e))?
        };
        let dataset = if resample_missing && !tfs.is_empty() {
            let refs: Vec<&str> = tfs.iter().map(|s| s.as_str()).collect();
            ensure_timeframes_with_resample(&dataset, &base_tf, &refs)
                .map_err(|e| format!("Resample failed: {}", e))?
        } else {
            dataset
        };
        Ok(dataset)
    });

    let dataset = result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;
    let frames = PyDict::new(py);
    for (tf, ohlcv) in dataset.frames {
        let entry = PyDict::new(py);
        entry.set_item("open", ohlcv.open)?;
        entry.set_item("high", ohlcv.high)?;
        entry.set_item("low", ohlcv.low)?;
        entry.set_item("close", ohlcv.close)?;
        if let Some(volume) = ohlcv.volume {
            entry.set_item("volume", volume)?;
        }
        if let Some(ts) = ohlcv.timestamp {
            entry.set_item("timestamp", ts)?;
        }
        frames.set_item(tf, entry)?;
    }
    Ok(frames.into_any().into())
}

#[pyfunction]
#[pyo3(signature = (
    root,
    symbol,
    base_tf="M1",
    higher_tfs=None,
    include_raw=true,
    cache_dir=None,
    cache_ttl_minutes=0,
    cache_enabled=false,
    resample_missing=true,
    arrow_tensor=false
))]
fn load_symbol_features(
    py: Python,
    root: String,
    symbol: String,
    base_tf: &str,
    higher_tfs: Option<Vec<String>>,
    include_raw: bool,
    cache_dir: Option<String>,
    cache_ttl_minutes: u64,
    cache_enabled: bool,
    resample_missing: bool,
    arrow_tensor: bool,
) -> PyResult<Py<PyAny>> {
    let result: Result<(forex_data::FeatureFrame, String, Ohlcv), String> = py.detach(|| {
        let higher = higher_tfs.clone().unwrap_or_default();
        let dataset = if higher.is_empty() {
            load_symbol_dataset(&root, &symbol)
                .map_err(|e| format!("Load dataset failed: {}", e))?
        } else {
            let mut merged = higher.clone();
            if !merged.iter().any(|tf| tf.eq_ignore_ascii_case(&base_tf)) {
                merged.push(base_tf.to_string());
            }
            let refs: Vec<&str> = merged.iter().map(|s| s.as_str()).collect();
            load_symbol_dataset_with_timeframes(&root, &symbol, &refs)
                .map_err(|e| format!("Load dataset failed: {}", e))?
        };

        let dataset = if resample_missing && !higher.is_empty() {
            let refs: Vec<&str> = higher.iter().map(|s| s.as_str()).collect();
            ensure_timeframes_with_resample(&dataset, &base_tf, &refs)
                .map_err(|e| format!("Resample failed: {}", e))?
        } else {
            dataset
        };

        let base_final = resolve_base_tf(&dataset, &base_tf);
        let refs: Vec<&str> = higher.iter().map(|s| s.as_str()).collect();
        let cache = cache_dir.as_ref().map(|dir| FeatureCache::new(dir, cache_ttl_minutes, cache_enabled));
        let mut frame = prepare_multitimeframe_features(
            &dataset,
            &base_final,
            &refs,
            cache.as_ref(),
        )
        .map_err(|e| format!("Feature computation failed: {}", e))?;

        if !include_raw {
            // Remove raw OHLC/volume columns when requested.
            let mut keep = Vec::new();
            for (idx, name) in frame.names.iter().enumerate() {
                let lower = name.to_ascii_lowercase();
                if lower == "open" || lower == "high" || lower == "low" || lower == "close" || lower == "volume" {
                    continue;
                }
                keep.push(idx);
            }
            if !keep.is_empty() && keep.len() < frame.names.len() {
                let mut names = Vec::with_capacity(keep.len());
                let mut data = Array2::<f32>::zeros((frame.data.nrows(), keep.len()));
                for (col_pos, col_idx) in keep.iter().enumerate() {
                    names.push(frame.names[*col_idx].clone());
                    for row in 0..frame.data.nrows() {
                        data[(row, col_pos)] = frame.data[(row, *col_idx)];
                    }
                }
                frame.names = names;
                frame.data = data;
            }
        }

        let base = dataset
            .frames
            .get(&base_final)
            .cloned()
            .ok_or_else(|| "Base timeframe data missing".to_string())?;
        Ok((frame, base_final, base))
    });

    let (frame, base_final, base) = result
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

    let dict = PyDict::new(py);
    dict.set_item("timestamps", frame.timestamps)?;
    dict.set_item("feature_names", frame.names)?;
    let features_py = frame.data.into_pyarray(py);
    dict.set_item("features", &features_py)?;
    if arrow_tensor {
        if let Ok(pyarrow) = py.import("pyarrow") {
            if let Ok(tensor_cls) = pyarrow.getattr("Tensor") {
                if let Ok(tensor) = tensor_cls.call_method1("from_numpy", (features_py.as_any(),)) {
                    let _ = dict.set_item("features_arrow_tensor", tensor);
                }
            }
        }
    }
    dict.set_item("base_tf", base_final)?;
    dict.set_item("open", base.open)?;
    dict.set_item("high", base.high)?;
    dict.set_item("low", base.low)?;
    dict.set_item("close", base.close)?;
    if let Some(volume) = base.volume {
        dict.set_item("volume", volume)?;
    }
    if let Some(ts) = base.timestamp {
        dict.set_item("base_timestamps", ts)?;
    }
    Ok(dict.into_any().into())
}

#[pyfunction]
#[pyo3(signature = (
    root,
    symbol,
    base_tf="M1",
    higher_tfs=None,
    knowledge_path=None,
    portfolio_limit=0,
    cache_dir=None,
    cache_ttl_minutes=0,
    cache_enabled=false,
    resample_missing=true
))]
fn load_strategy_signals(
    py: Python,
    root: String,
    symbol: String,
    base_tf: &str,
    higher_tfs: Option<Vec<String>>,
    knowledge_path: Option<String>,
    portfolio_limit: usize,
    cache_dir: Option<String>,
    cache_ttl_minutes: u64,
    cache_enabled: bool,
    resample_missing: bool,
) -> PyResult<Py<PyAny>> {
    let result: Result<(forex_data::FeatureFrame, String, Ohlcv, Vec<String>, Array2<i8>), String> = py.detach(|| {
        let higher = higher_tfs.clone().unwrap_or_default();
        let dataset = if higher.is_empty() {
            load_symbol_dataset(&root, &symbol)
                .map_err(|e| format!("Load dataset failed: {}", e))?
        } else {
            let mut merged = higher.clone();
            if !merged.iter().any(|tf| tf.eq_ignore_ascii_case(&base_tf)) {
                merged.push(base_tf.to_string());
            }
            let refs: Vec<&str> = merged.iter().map(|s| s.as_str()).collect();
            load_symbol_dataset_with_timeframes(&root, &symbol, &refs)
                .map_err(|e| format!("Load dataset failed: {}", e))?
        };

        let dataset = if resample_missing && !higher.is_empty() {
            let refs: Vec<&str> = higher.iter().map(|s| s.as_str()).collect();
            ensure_timeframes_with_resample(&dataset, &base_tf, &refs)
                .map_err(|e| format!("Resample failed: {}", e))?
        } else {
            dataset
        };

        let base_final = resolve_base_tf(&dataset, &base_tf);
        let refs: Vec<&str> = higher.iter().map(|s| s.as_str()).collect();
        let cache = cache_dir.as_ref().map(|dir| FeatureCache::new(dir, cache_ttl_minutes, cache_enabled));
        let frame = prepare_multitimeframe_features(&dataset, &base_final, &refs, cache.as_ref())
            .map_err(|e| format!("Feature computation failed: {}", e))?;

        let base = dataset
            .frames
            .get(&base_final)
            .cloned()
            .ok_or_else(|| "Base timeframe data missing".to_string())?;

        let mut path = if let Some(p) = knowledge_path.clone() {
            PathBuf::from(p)
        } else {
            let dir = cache_dir.clone().unwrap_or_else(|| "cache".to_string());
            let mut p = PathBuf::from(dir);
            if !symbol.is_empty() {
                p.push(format!("talib_knowledge_{}.json", symbol));
            } else {
                p.push("talib_knowledge.json");
            }
            p
        };
        if !path.exists() {
            // Fallback to generic catalog if symbol-specific does not exist.
            if let Some(dir) = cache_dir.clone() {
                let mut fallback = PathBuf::from(dir);
                fallback.push("talib_knowledge.json");
                if fallback.exists() {
                    path = fallback;
                }
            }
        }
        if !path.exists() {
            return Err(format!("Strategy catalog not found at {}", path.display()));
        }

        let mut specs = load_strategy_specs(&path)?;
        if portfolio_limit > 0 && specs.len() > portfolio_limit {
            specs.truncate(portfolio_limit);
        }

        let mut strategy_ids: Vec<String> = Vec::new();
        let mut genes: Vec<forex_search::Gene> = Vec::new();
        for (idx, spec) in specs.into_iter().enumerate() {
            let indicators = spec.indicators.unwrap_or_default();
            if indicators.is_empty() {
                continue;
            }
            let mut indices = Vec::new();
            let mut weights = Vec::new();
            for ind in indicators.iter() {
                if let Some(col_idx) = map_indicator_index(ind, &frame.names) {
                    indices.push(col_idx);
                    let weight = spec
                        .weights
                        .as_ref()
                        .and_then(|w| w.get(ind))
                        .copied()
                        .unwrap_or(1.0);
                    weights.push(weight as f32);
                }
            }
            if indices.is_empty() {
                continue;
            }
            let id = spec
                .strategy_id
                .clone()
                .unwrap_or_else(|| format!("strategy_{idx}"));
            strategy_ids.push(id.clone());
            genes.push(forex_search::Gene {
                indices,
                weights,
                long_threshold: spec.long_threshold.unwrap_or(0.66) as f32,
                short_threshold: spec.short_threshold.unwrap_or(-0.66) as f32,
                fitness: 0.0,
                sharpe_ratio: 0.0,
                win_rate: 0.0,
                max_drawdown: 0.0,
                profit_factor: 0.0,
                expectancy: 0.0,
                trades_count: 0,
                generation: 0,
                strategy_id: id,
                use_ob: spec.use_ob.unwrap_or(false),
                use_fvg: spec.use_fvg.unwrap_or(false),
                use_liq_sweep: spec.use_liq_sweep.unwrap_or(false),
                mtf_confirmation: spec.mtf_confirmation.unwrap_or(false),
                use_premium_discount: spec.use_premium_discount.unwrap_or(false),
                use_inducement: spec.use_inducement.unwrap_or(false),
                tp_pips: spec.tp_pips.unwrap_or(0.0),
                sl_pips: spec.sl_pips.unwrap_or(0.0),
                slice_pass_rate: 0.0,
            });
        }

        if genes.is_empty() {
            return Err("No strategies could be mapped to feature columns".to_string());
        }

        let n_rows = frame.data.nrows();
        let n_cols = genes.len();
        let mut out = Array2::<i8>::zeros((n_rows, n_cols));
        for (col_idx, gene) in genes.iter().enumerate() {
            let sig = forex_search::signals_for_gene(&frame, gene);
            for (row_idx, v) in sig.iter().enumerate() {
                out[(row_idx, col_idx)] = *v;
            }
        }

        let signals = out;
        Ok((frame, base_final, base, strategy_ids, signals))
    });

    let (frame, base_final, base, strategy_ids, signals) = result
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;
    let dict = PyDict::new(py);
    dict.set_item("timestamps", frame.timestamps)?;
    dict.set_item("strategy_ids", strategy_ids)?;
    dict.set_item("signals", signals.into_pyarray(py))?;
    dict.set_item("base_tf", base_final)?;
    dict.set_item("open", base.open)?;
    dict.set_item("high", base.high)?;
    dict.set_item("low", base.low)?;
    dict.set_item("close", base.close)?;
    if let Some(volume) = base.volume {
        dict.set_item("volume", volume)?;
    }
    if let Some(ts) = base.timestamp {
        dict.set_item("base_timestamps", ts)?;
    }
    Ok(dict.into_any().into())
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, timestamps=None, volume=None, population=64, generations=20, max_indicators=12, include_raw=true))]
fn search_evolve_ohlcv(
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
        .map(|g| pythonize(py, g).map(|obj| obj.into()).unwrap_or_else(|_| py.None()))
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
fn search_evolve_gpu_ohlcv(
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
                    let frames = vec![features.clone()];
                    run_gpu_discovery(&frames, &ohlcv, &config)
                        .map_err(|e| format!("GPU search failed: {}", e))?
                }
                #[cfg(not(feature = "gpu"))]
                {
                    let frames = vec![features.clone()];
                    run_gpu_discovery(frames, features.names.clone(), ohlcv.clone(), &config)
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

fn discovery_gene_key(gene: &forex_search::Gene) -> String {
    let sid = gene.strategy_id.trim();
    if !sid.is_empty() {
        return format!("id:{sid}");
    }
    format!(
        "sig:{:?}|{:.6}|{:.6}|{}|{}|{}|{}|{}|{}|{:.2}|{:.2}",
        gene.indices,
        gene.long_threshold,
        gene.short_threshold,
        gene.use_ob as u8,
        gene.use_fvg as u8,
        gene.use_liq_sweep as u8,
        gene.mtf_confirmation as u8,
        gene.use_premium_discount as u8,
        gene.use_inducement as u8,
        gene.tp_pips,
        gene.sl_pips
    )
}

fn rank_dedupe_genes(genes: &[forex_search::Gene]) -> Vec<forex_search::Gene> {
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

fn select_ranked_discovery_genes(
    candidates: &[forex_search::Gene],
    keep_max_dd: f64,
    keep_min_profit: f64,
    keep_min_trades: f64,
    keep_min_count: usize,
    keep_cap: usize,
) -> (Vec<forex_search::Gene>, usize, usize) {
    let ranked_all = rank_dedupe_genes(candidates);
    let ranked_filtered_input: Vec<forex_search::Gene> = candidates
        .iter()
        .filter(|g| {
            g.fitness > keep_min_profit
                && g.max_drawdown <= keep_max_dd
                && (g.trades_count as f64) >= keep_min_trades
        })
        .cloned()
        .collect();
    let ranked_filtered = rank_dedupe_genes(&ranked_filtered_input);

    let mut selected = ranked_filtered.clone();
    if keep_min_count > 0 && selected.len() < keep_min_count {
        let mut seen: HashSet<String> = selected
            .iter()
            .map(discovery_gene_key)
            .collect();
        for gene in &ranked_all {
            let key = discovery_gene_key(gene);
            if seen.insert(key) {
                selected.push(gene.clone());
                if selected.len() >= keep_min_count {
                    break;
                }
            }
        }
    }

    if selected.is_empty() {
        selected = ranked_all.clone();
    }
    if keep_cap > 0 && selected.len() > keep_cap {
        selected.truncate(keep_cap);
    }

    (selected, ranked_filtered.len(), ranked_all.len())
}

#[pyfunction]
#[pyo3(signature = (
    open,
    high,
    low,
    close,
    timestamps=None,
    volume=None,
    population=100,
    generations=5,
    max_indicators=12,
    candidate_count=200,
    portfolio_size=100,
    corr_threshold=0.7,
    min_trades_per_day=1.0,
    include_raw=true,
    keep_max_dd=1.0,
    keep_min_profit=0.0,
    keep_min_trades=0.0,
    keep_min_count=0,
    keep_cap=0
))]
fn search_discovery_ohlcv(
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
    candidate_count: usize,
    portfolio_size: usize,
    corr_threshold: f64,
    min_trades_per_day: f64,
    include_raw: bool,
    keep_max_dd: f64,
    keep_min_profit: f64,
    keep_min_trades: f64,
    keep_min_count: usize,
    keep_cap: usize,
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

    let config = DiscoveryConfig {
        population,
        generations,
        max_indicators,
        candidate_count,
        portfolio_size,
        corr_threshold,
        min_trades_per_day,
    };

    let (features, result) = py
        .detach(|| {
            let features = compute_talib_feature_frame(&ohlcv, include_raw)
                .map_err(|e| format!("Feature computation failed: {}", e))?;
            let result = run_discovery_cycle(&features, &ohlcv, &config)
                .map_err(|e| format!("Discovery failed: {}", e))?;
            Ok::<_, String>((features, result))
        })
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

    let max_dd = keep_max_dd.clamp(0.0, 1.0);
    let min_profit = keep_min_profit;
    let min_trades = keep_min_trades.max(0.0);
    let min_keep = keep_min_count;
    let cap = if keep_cap > 0 {
        keep_cap
    } else {
        portfolio_size.max(1)
    };
    let (selected_portfolio, strict_kept, ranked_total) = select_ranked_discovery_genes(
        &result.portfolio,
        max_dd,
        min_profit,
        min_trades,
        min_keep,
        cap.max(1),
    );

    let portfolio_py: Vec<Py<PyAny>> = selected_portfolio
        .iter()
        .map(|g| pythonize(py, g).map(|obj| obj.into()).unwrap_or_else(|_| py.None()))
        .collect();
    let candidates_py: Vec<Py<PyAny>> = result
        .candidates
        .iter()
        .map(|g| pythonize(py, g).map(|obj| obj.into()).unwrap_or_else(|_| py.None()))
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("portfolio", portfolio_py)?;
    dict.set_item("candidates", candidates_py)?;
    dict.set_item("feature_names", features.names)?;
    dict.set_item("strict_kept", strict_kept)?;
    dict.set_item("ranked_total", ranked_total)?;
    dict.set_item("rust_ranked", true)?;
    Ok(dict.into_any().into())
}

#[pyfunction]
#[pyo3(signature = (
    open,
    high,
    low,
    close,
    pip_size,
    vol_estimator="ensemble",
    vol_window=50,
    ewma_lambda=0.94,
    vol_horizon_bars=5,
    tail_window=100,
    tail_alpha=0.975,
    tail_step=5,
    tail_max_bars=300_000,
    stop_k_vol=1.0,
    stop_k_tail=1.25,
    meta_label_min_dist=0.0,
    regime_adx_trend=25.0,
    regime_adx_range=20.0,
    hurst_window=100,
    hurst_trend=0.55,
    hurst_range=0.45,
    rr_trend=2.5,
    rr_range=1.5,
    rr_neutral=2.0,
    ema_fast_period=20,
    ema_slow_period=50,
    atr_period=14
))]
fn infer_stop_target_pips_ohlcv(
    py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    pip_size: f64,
    vol_estimator: &str,
    vol_window: usize,
    ewma_lambda: f64,
    vol_horizon_bars: usize,
    tail_window: usize,
    tail_alpha: f64,
    tail_step: usize,
    tail_max_bars: usize,
    stop_k_vol: f64,
    stop_k_tail: f64,
    meta_label_min_dist: f64,
    regime_adx_trend: f64,
    regime_adx_range: f64,
    hurst_window: usize,
    hurst_trend: f64,
    hurst_range: f64,
    rr_trend: f64,
    rr_range: f64,
    rr_neutral: f64,
    ema_fast_period: usize,
    ema_slow_period: usize,
    atr_period: usize,
) -> PyResult<Option<(f64, f64, f64)>> {
    if pip_size <= 0.0 || !pip_size.is_finite() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "pip_size must be a positive finite value",
        ));
    }

    let open_vec = vec_from_py_f64(&open);
    let high_vec = vec_from_py_f64(&high);
    let low_vec = vec_from_py_f64(&low);
    let close_vec = vec_from_py_f64(&close);
    let n = close_vec.len();
    if n == 0 || open_vec.len() != n || high_vec.len() != n || low_vec.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "open/high/low/close arrays must have equal non-zero length",
        ));
    }

    let mut settings = StopTargetSettings::default();
    settings.vol_estimator = vol_estimator.to_string();
    settings.vol_window = vol_window.max(2);
    settings.ewma_lambda = ewma_lambda;
    settings.vol_horizon_bars = vol_horizon_bars.max(1);
    settings.tail_window = tail_window.max(3);
    settings.tail_alpha = tail_alpha;
    settings.tail_step = tail_step.max(1);
    settings.tail_max_bars = tail_max_bars.max(10_000);
    settings.stop_k_vol = stop_k_vol.max(0.0);
    settings.stop_k_tail = stop_k_tail.max(0.0);
    settings.meta_label_min_dist = meta_label_min_dist.max(0.0);
    settings.regime_adx_trend = regime_adx_trend;
    settings.regime_adx_range = regime_adx_range;
    settings.hurst_window = hurst_window.max(20);
    settings.hurst_trend = hurst_trend;
    settings.hurst_range = hurst_range;
    settings.rr_trend = rr_trend.max(0.1);
    settings.rr_range = rr_range.max(0.1);
    settings.rr_neutral = rr_neutral.max(0.1);
    settings.ema_fast_period = ema_fast_period.max(2);
    settings.ema_slow_period = ema_slow_period.max(settings.ema_fast_period + 1);
    settings.atr_period = atr_period.max(5);

    let out = py.detach(|| {
        Ok::<Option<(f64, f64, f64)>, String>(infer_stop_target_pips_rs(
            &open_vec,
            &high_vec,
            &low_vec,
            &close_vec,
            &settings,
            pip_size,
        ))
    });
    out.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
}

#[pyfunction]
#[pyo3(signature = (
    close_prices,
    high_prices,
    low_prices,
    signals,
    month_indices,
    day_indices,
    sl_pips,
    tp_pips,
    max_hold_bars=0,
    trailing_enabled=false,
    trailing_atr_multiplier=1.0,
    trailing_be_trigger_r=1.0,
    pip_value=0.0001,
    spread_pips=1.5,
    commission_per_trade=0.0,
    pip_value_per_lot=10.0
))]
fn fast_evaluate_strategy(
    py: Python,
    close_prices: PyReadonlyArray1<f64>,
    high_prices: PyReadonlyArray1<f64>,
    low_prices: PyReadonlyArray1<f64>,
    signals: PyReadonlyArray1<i8>,
    month_indices: PyReadonlyArray1<i64>,
    day_indices: PyReadonlyArray1<i64>,
    sl_pips: f64,
    tp_pips: f64,
    max_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
) -> PyResult<Vec<f64>> {
    let close_vec = vec_from_py_f64(&close_prices);
    let high_vec = vec_from_py_f64(&high_prices);
    let low_vec = vec_from_py_f64(&low_prices);
    let sig_vec = vec_from_py_i8(&signals);
    let month_vec = vec_from_py_i64(&month_indices);
    let day_vec = vec_from_py_i64(&day_indices);

    let n = close_vec.len();
    if high_vec.len() != n || low_vec.len() != n || sig_vec.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "close/high/low/signals arrays must have equal length",
        ));
    }
    if month_vec.len() != n || day_vec.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "month_indices/day_indices length must match close length",
        ));
    }

    let out = py
        .detach(|| {
            Ok::<[f64; 11], String>(forex_search::fast_evaluate_strategy_core(
                &close_vec,
                &high_vec,
                &low_vec,
                &sig_vec,
                &month_vec,
                &day_vec,
                sl_pips,
                tp_pips,
                max_hold_bars,
                trailing_enabled,
                trailing_atr_multiplier,
                trailing_be_trigger_r,
                pip_value,
                spread_pips,
                commission_per_trade,
                pip_value_per_lot,
            ))
        })
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

    Ok(out.to_vec())
}

#[pyfunction]
#[pyo3(signature = (
    close_prices,
    high_prices,
    low_prices,
    signals,
    month_indices,
    day_indices,
    sl_pips,
    tp_pips,
    max_hold_bars=0,
    trailing_enabled=false,
    trailing_atr_multiplier=1.0,
    trailing_be_trigger_r=1.0,
    pip_value=0.0001,
    spread_pips=1.5,
    commission_per_trade=0.0,
    pip_value_per_lot=10.0
))]
fn batch_evaluate_strategies<'py>(
    py: Python<'py>,
    close_prices: PyReadonlyArray1<'py, f64>,
    high_prices: PyReadonlyArray1<'py, f64>,
    low_prices: PyReadonlyArray1<'py, f64>,
    signals: PyReadonlyArray2<'py, i8>,
    month_indices: PyReadonlyArray1<'py, i64>,
    day_indices: PyReadonlyArray1<'py, i64>,
    sl_pips: PyReadonlyArray1<'py, f64>,
    tp_pips: PyReadonlyArray1<'py, f64>,
    max_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_vec = vec_from_py_f64(&close_prices);
    let high_vec = vec_from_py_f64(&high_prices);
    let low_vec = vec_from_py_f64(&low_prices);
    let month_vec = vec_from_py_i64(&month_indices);
    let day_vec = vec_from_py_i64(&day_indices);
    let sl_vec = vec_from_py_f64(&sl_pips);
    let tp_vec = vec_from_py_f64(&tp_pips);
    let signals_mat = signals.as_array().to_owned();

    let n = close_vec.len();
    if high_vec.len() != n || low_vec.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "close/high/low arrays must have equal length",
        ));
    }
    if month_vec.len() != n || day_vec.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "month_indices/day_indices length must match close length",
        ));
    }

    let n_strats = signals_mat.nrows();
    let n_bars = signals_mat.ncols();
    if n_bars != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "signals.shape[1] must match close length",
        ));
    }
    if !(sl_vec.len() == n_strats || sl_vec.len() == 1) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "sl_pips length must be 1 or signals.shape[0]",
        ));
    }
    if !(tp_vec.len() == n_strats || tp_vec.len() == 1) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "tp_pips length must be 1 or signals.shape[0]",
        ));
    }

    let out = py
        .detach(|| {
            let mut metrics = Array2::<f64>::zeros((n_strats, 11));
            for row in 0..n_strats {
                let row_view = signals_mat.row(row);
                let sig_owned: Vec<i8>;
                let sig_row: &[i8] = if let Some(s) = row_view.as_slice() {
                    s
                } else {
                    sig_owned = row_view.iter().copied().collect();
                    &sig_owned
                };
                let sl = if sl_vec.len() == 1 { sl_vec[0] } else { sl_vec[row] };
                let tp = if tp_vec.len() == 1 { tp_vec[0] } else { tp_vec[row] };
                let m = forex_search::fast_evaluate_strategy_core(
                    &close_vec,
                    &high_vec,
                    &low_vec,
                    &sig_row,
                    &month_vec,
                    &day_vec,
                    sl,
                    tp,
                    max_hold_bars,
                    trailing_enabled,
                    trailing_atr_multiplier,
                    trailing_be_trigger_r,
                    pip_value,
                    spread_pips,
                    commission_per_trade,
                    pip_value_per_lot,
                );
                for col in 0..11 {
                    metrics[(row, col)] = m[col];
                }
            }
            Ok::<Array2<f64>, String>(metrics)
        })
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (
    close,
    high,
    low,
    sl_dist,
    tp_dist,
    max_hold,
    base_signal=None
))]
fn triple_barrier_labels(
    py: Python,
    close: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    sl_dist: PyReadonlyArray1<f64>,
    tp_dist: PyReadonlyArray1<f64>,
    max_hold: usize,
    base_signal: Option<PyReadonlyArray1<i8>>,
) -> PyResult<Vec<i8>> {
    let close_vec = vec_from_py_f64(&close);
    let high_vec = vec_from_py_f64(&high);
    let low_vec = vec_from_py_f64(&low);
    let sl_vec = vec_from_py_f64(&sl_dist);
    let tp_vec = vec_from_py_f64(&tp_dist);
    let sig_vec = base_signal
        .as_ref()
        .map(|arr| arr.as_array().iter().copied().collect::<Vec<i8>>());

    let n = close_vec.len();
    if high_vec.len() != n || low_vec.len() != n || sl_vec.len() != n || tp_vec.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "close/high/low/sl_dist/tp_dist arrays must have equal length",
        ));
    }
    if let Some(sig) = sig_vec.as_ref() {
        if sig.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "base_signal length must match close length",
            ));
        }
    }

    let labels = py
        .detach(|| {
            if n == 0 {
                return Ok::<Vec<i8>, String>(Vec::new());
            }
            let hold = max_hold.max(1);
            let mut labels = vec![0i8; n];
            for i in 0..n {
                let entry = close_vec[i];
                if !entry.is_finite() {
                    continue;
                }
                let j_end = (i + hold).min(n.saturating_sub(1));
                if j_end <= i {
                    continue;
                }

                let sd = sl_vec[i].max(0.0);
                let td = tp_vec[i].max(0.0);
                if sd <= 0.0 && td <= 0.0 {
                    continue;
                }

                let sig = sig_vec.as_ref().map(|v| v[i]).unwrap_or(0);
                let mut out = 0i8;
                if sig > 0 {
                    let tp_lvl = entry + td;
                    let sl_lvl = entry - sd;
                    for j in (i + 1)..=j_end {
                        let hit_tp = high_vec[j] >= tp_lvl;
                        let hit_sl = low_vec[j] <= sl_lvl;
                        if hit_tp && hit_sl {
                            out = if close_vec[j] >= entry { 1 } else { -1 };
                            break;
                        }
                        if hit_tp {
                            out = 1;
                            break;
                        }
                        if hit_sl {
                            out = -1;
                            break;
                        }
                    }
                } else if sig < 0 {
                    let tp_lvl = entry - td;
                    let sl_lvl = entry + sd;
                    for j in (i + 1)..=j_end {
                        let hit_tp = low_vec[j] <= tp_lvl;
                        let hit_sl = high_vec[j] >= sl_lvl;
                        if hit_tp && hit_sl {
                            out = if close_vec[j] <= entry { 1 } else { -1 };
                            break;
                        }
                        if hit_tp {
                            out = 1;
                            break;
                        }
                        if hit_sl {
                            out = -1;
                            break;
                        }
                    }
                } else {
                    let up_lvl = entry + td;
                    let dn_lvl = entry - sd;
                    for j in (i + 1)..=j_end {
                        let hit_up = high_vec[j] >= up_lvl;
                        let hit_dn = low_vec[j] <= dn_lvl;
                        if hit_up && hit_dn {
                            out = if close_vec[j] >= entry { 1 } else { -1 };
                            break;
                        }
                        if hit_up {
                            out = 1;
                            break;
                        }
                        if hit_dn {
                            out = -1;
                            break;
                        }
                    }
                }
                labels[i] = out;
            }
            Ok(labels)
        })
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

    Ok(labels)
}

// ============================================================================
// TREE MODELS PYTHON BINDINGS
// ============================================================================

#[cfg(feature = "lightgbm")]
#[pyclass]
struct LightGBMModel {
    model: Mutex<LightGBMExpert>,
}

#[cfg(feature = "lightgbm")]
#[pymethods]
impl LightGBMModel {
    #[new]
    #[pyo3(signature = (idx=1, params=None))]
    fn new(idx: usize, params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let params = params_from_py(params).map_err(|msg| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
        })?;
        Self {
            model: Mutex::new(LightGBMExpert::new(idx, params)),
        }
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i64> = labels_array.iter().copied().collect();
            let labels_series = Series::new("label".into(), labels_vec);

            model
                .fit(&df, &labels_series)
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        })();

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = (|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            model
                .predict_proba(&df)
                .map_err(|e| format!("Prediction failed: {}", e))
        })();

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr: Array2<f32>| arr.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;

        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(feature = "xgboost")]
#[pyclass(unsendable)]
struct XGBoostModel {
    model: Mutex<XGBoostExpert>,
}

#[cfg(feature = "xgboost")]
#[pymethods]
impl XGBoostModel {
    #[new]
    #[pyo3(signature = (idx=1, params=None))]
    fn new(idx: usize, params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let params = params_from_py(params).map_err(|msg| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
        })?;
        Ok(Self {
            model: Mutex::new(XGBoostExpert::new(idx, params)),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i64> = labels_array.iter().copied().collect();
            let labels_series = Series::new("label".into(), labels_vec);

            model
                .fit(&df, &labels_series)
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        })();

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = (|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            model
                .predict_proba(&df)
                .map_err(|e| format!("Prediction failed: {}", e))
        })();

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr: Array2<f32>| arr.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;

        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(feature = "xgboost")]
#[pyclass(unsendable)]
struct XGBoostRFModel {
    model: Mutex<XGBoostRFExpert>,
}

#[cfg(feature = "xgboost")]
#[pymethods]
impl XGBoostRFModel {
    #[new]
    #[pyo3(signature = (idx=1, params=None))]
    fn new(idx: usize, params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let params = params_from_py(params).map_err(|msg| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
        })?;
        Ok(Self {
            model: Mutex::new(XGBoostRFExpert::new(idx, params)),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i64> = labels_array.iter().copied().collect();
            let labels_series = Series::new("label".into(), labels_vec);

            model
                .fit(&df, &labels_series)
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        })();

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = (|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            model
                .predict_proba(&df)
                .map_err(|e| format!("Prediction failed: {}", e))
        })();

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr: Array2<f32>| arr.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;

        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(feature = "xgboost")]
#[pyclass(unsendable)]
struct XGBoostDARTModel {
    model: Mutex<XGBoostDARTExpert>,
}

#[cfg(feature = "xgboost")]
#[pymethods]
impl XGBoostDARTModel {
    #[new]
    #[pyo3(signature = (idx=1, params=None))]
    fn new(idx: usize, params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let params = params_from_py(params).map_err(|msg| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
        })?;
        Ok(Self {
            model: Mutex::new(XGBoostDARTExpert::new(idx, params)),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i64> = labels_array.iter().copied().collect();
            let labels_series = Series::new("label".into(), labels_vec);

            model
                .fit(&df, &labels_series)
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        })();

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = (|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            model
                .predict_proba(&df)
                .map_err(|e| format!("Prediction failed: {}", e))
        })();

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr: Array2<f32>| arr.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;

        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(feature = "catboost")]
#[pyclass]
struct CatBoostModel {
    model: Mutex<CatBoostExpert>,
}

#[cfg(feature = "catboost")]
#[pymethods]
impl CatBoostModel {
    #[new]
    #[pyo3(signature = (idx=1, params=None))]
    fn new(idx: usize, params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let params = params_from_py(params)
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;
        Ok(Self {
            model: Mutex::new(CatBoostExpert::new(idx, params)),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i64> = labels_array.iter().copied().collect();
            let labels_series = Series::new("label".into(), labels_vec);

            model
                .fit(&df, &labels_series)
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        })();

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = (|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            model
                .predict_proba(&df)
                .map_err(|e| format!("Prediction failed: {}", e))
        })();

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr: Array2<f32>| arr.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;

        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(feature = "catboost")]
#[pyclass]
struct CatBoostAltModel {
    model: Mutex<CatBoostAltExpert>,
}

#[cfg(feature = "catboost")]
#[pymethods]
impl CatBoostAltModel {
    #[new]
    #[pyo3(signature = (idx=1, params=None))]
    fn new(idx: usize, params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let params = params_from_py(params)
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;
        Ok(Self {
            model: Mutex::new(CatBoostAltExpert::new(idx, params)),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i64> = labels_array.iter().copied().collect();
            let labels_series = Series::new("label".into(), labels_vec);

            model
                .fit(&df, &labels_series)
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        })();

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = (|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            model
                .predict_proba(&df)
                .map_err(|e| format!("Prediction failed: {}", e))
        })();

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr: Array2<f32>| arr.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;

        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;

        Ok(())
    }
}

#[pymodule]
fn forex_bindings(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {       
    m.add_class::<ForexCore>()?;
    #[cfg(feature = "onnx")]
    m.add_class::<ModelEngine>()?;
    m.add_function(wrap_pyfunction!(search_evolve_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(search_evolve_gpu_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(search_discovery_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(load_symbol_frames, m)?)?;
    m.add_function(wrap_pyfunction!(load_symbol_features, m)?)?;
    m.add_function(wrap_pyfunction!(infer_stop_target_pips_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(fast_evaluate_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(batch_evaluate_strategies, m)?)?;
    m.add_function(wrap_pyfunction!(triple_barrier_labels, m)?)?;
    m.add_function(wrap_pyfunction!(compute_position_size_lots, m)?)?;
    m.add_function(wrap_pyfunction!(pip_size_from_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(compute_order_prices, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_trade_edge, m)?)?;
    m.add_function(wrap_pyfunction!(talib_bulk_signals_ohlcv, m)?)?;

    // Add tree model classes if features enabled
    #[cfg(feature = "lightgbm")]
    m.add_class::<LightGBMModel>()?;
    #[cfg(feature = "xgboost")]
    {
        m.add_class::<XGBoostModel>()?;
        m.add_class::<XGBoostRFModel>()?;
        m.add_class::<XGBoostDARTModel>()?;
    }
    #[cfg(feature = "catboost")]
    {
        m.add_class::<CatBoostModel>()?;
        m.add_class::<CatBoostAltModel>()?;
    }

    Ok(())
}
