use chrono::{Datelike, TimeZone, Utc};
use forex_core::system::HardwareProbe;
use forex_data::{
    compute_talib_feature_frame, ensure_timeframes_with_resample, load_symbol_dataset,
    load_symbol_dataset_with_timeframes, prepare_multitimeframe_features_with_options,
    FeatureBuildOptions, FeatureCache, FeatureProfile, Ohlcv, SymbolDataset,
};
#[cfg(feature = "onnx")]
use forex_models::ONNXInferenceEngine;
use forex_search::{
    evaluate_population_core, evolve_search, infer_stop_target_pips as infer_stop_target_pips_rs,
    run_discovery_cycle, run_gpu_discovery, DiscoveryConfig, GpuDiscoveryConfig,
    StopTargetSettings,
};
use ndarray::{Array2, Ix1, Ix2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pythonize::pythonize;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "onnx")]
use std::sync::Arc;
use std::sync::Mutex;

#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
use forex_models::base::ExpertModel;
#[cfg(feature = "lightgbm")]
use forex_models::tree_models::LightGBMExpert;
#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
use forex_models::tree_models::ParamValue;

#[cfg(feature = "xgboost")]
use forex_models::tree_models::{XGBoostDARTExpert, XGBoostExpert, XGBoostRFExpert};

use forex_models::genetic::GeneticStrategyExpert;
use forex_models::neural_networks::MLPExpert as RustMlpExpert;
#[cfg(feature = "catboost")]
use forex_models::tree_models::{CatBoostAltExpert, CatBoostExpert};

#[pyclass]
struct ForexCore {
    probe: Mutex<HardwareProbe>,
}

#[pyclass(name = "ConformalGate", module = "forex_bindings")]
struct ConformalGate {
    #[pyo3(get, set)]
    alpha: f64,
    #[pyo3(get, set)]
    qhat: f64,
    #[pyo3(get, set)]
    fitted: bool,
    #[pyo3(get, set)]
    n_calib: usize,
}

#[pymethods]
impl ConformalGate {
    #[new]
    #[pyo3(signature = (alpha=0.10))]
    fn new(alpha: f64) -> Self {
        Self {
            alpha,
            qhat: 1.0,
            fitted: false,
            n_calib: 0,
        }
    }

    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        probs: &Bound<'py, PyAny>,
        y_true: &Bound<'py, PyAny>,
    ) -> PyResult<bool> {
        let np = py.import("numpy")?;
        let probs_array_any = np.getattr("asarray")?.call1((probs,))?;
        let probs_array_f64 = probs_array_any.call_method1("astype", ("float64",))?;
        let probs_array: PyReadonlyArray2<'py, f64> = probs_array_f64.extract()?;

        let labels_array_any = np.getattr("asarray")?.call1((y_true,))?;
        let labels_array_i64 = labels_array_any.call_method1("astype", ("int64",))?;
        let labels_array: PyReadonlyArray1<'py, i64> = labels_array_i64.extract()?;

        let p = probs_array.as_array();
        if p.ndim() != 2 || p.shape()[1] < 3 {
            return Ok(false);
        }

        let y_raw = labels_array.as_array();
        let n = usize::min(y_raw.len(), p.shape()[0]);
        if n < 64 {
            return Ok(false);
        }

        let alpha = self.alpha.clamp(1e-6, 0.99);
        let q_level = (((n + 1) as f64) * (1.0 - alpha)).ceil() / (n as f64);
        let q_level = q_level.clamp(0.0, 1.0);

        let mut scores = Vec::with_capacity(n);
        for row in 0..n {
            let y = match y_raw[row] {
                -1 => 2usize,
                value if value < 0 => 0usize,
                value if value > 2 => 2usize,
                value => value as usize,
            };
            let prob = p[[row, y]].clamp(1e-8, 1.0);
            scores.push(1.0 - prob);
        }
        scores.sort_by(|a, b| a.total_cmp(b));
        let idx = ((q_level * (n as f64)).ceil() as isize - 1).clamp(0, (n - 1) as isize) as usize;
        self.qhat = scores[idx].clamp(0.0, 1.0);
        self.fitted = true;
        self.n_calib = n;
        Ok(true)
    }

    fn prediction_set<'py>(
        &self,
        py: Python<'py>,
        probs_row: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<usize>> {
        let np = py.import("numpy")?;
        let row_any = np.getattr("asarray")?.call1((probs_row,))?;
        let row_f64 = row_any.call_method1("astype", ("float64",))?;
        let row_array: PyReadonlyArray1<'py, f64> = row_f64.extract()?;
        let row = row_array.as_array();
        if row.len() < 3 {
            return Ok(vec![0, 1, 2]);
        }

        let mut probs = [0.0_f64; 3];
        for idx in 0..3 {
            probs[idx] = row[idx].clamp(1e-8, 1.0);
        }

        let mut keep: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter_map(|(idx, prob)| {
                if (1.0 - *prob) <= self.qhat {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        if keep.is_empty() {
            let best = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            keep.push(best);
        }
        Ok(keep)
    }

    #[pyo3(signature = (probs_row, min_set_size=3))]
    fn should_abstain<'py>(
        &self,
        py: Python<'py>,
        probs_row: &Bound<'py, PyAny>,
        min_set_size: usize,
    ) -> PyResult<(bool, usize)> {
        if !self.fitted {
            return Ok((false, 1));
        }
        let keep = self.prediction_set(py, probs_row)?;
        let size = keep.len();
        Ok((size >= usize::max(1, min_set_size), size))
    }
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

fn dataframe_from_ndarray(features: &Array2<f64>) -> Result<DataFrame, String> {
    let mut df_data: Vec<Column> = Vec::with_capacity(features.ncols());
    for col_idx in 0..features.ncols() {
        let col_data: Vec<f64> = features.column(col_idx).iter().copied().collect();
        let name = format!("feature_{col_idx}");
        df_data.push(Series::new(name.into(), col_data).into());
    }
    DataFrame::new(df_data).map_err(|e| format!("DataFrame creation failed: {}", e))
}

fn dataframe_from_named_ndarray(
    features: &Array2<f64>,
    column_names: Option<&[String]>,
) -> Result<DataFrame, String> {
    let names = if let Some(names) = column_names {
        if names.len() == features.ncols() {
            names.to_vec()
        } else {
            (0..features.ncols())
                .map(|col_idx| format!("feature_{col_idx}"))
                .collect()
        }
    } else {
        (0..features.ncols())
            .map(|col_idx| format!("feature_{col_idx}"))
            .collect()
    };

    let mut df_data: Vec<Column> = Vec::with_capacity(features.ncols());
    for (col_idx, name) in names.iter().enumerate() {
        let col_data: Vec<f64> = features.column(col_idx).iter().copied().collect();
        df_data.push(Series::new(name.as_str().into(), col_data).into());
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

fn vec_from_py_i32(arr: &PyReadonlyArray1<i32>) -> Vec<i32> {
    arr.as_array().iter().copied().collect()
}

fn vec_from_py_f32(arr: &PyReadonlyArray1<f32>) -> Vec<f32> {
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
    let mut candidates: Vec<String> = Vec::new();
    candidates.push(norm.clone());
    if !norm.starts_with("smc_") {
        candidates.push(format!("smc_{norm}"));
    }
    match norm.as_str() {
        "ob" | "order_block" => candidates.push("smc_ob".to_string()),
        "fvg" | "fair_value_gap" => candidates.push("smc_fvg".to_string()),
        "liq" | "liq_sweep" | "liquidity_sweep" => candidates.push("smc_liq".to_string()),
        "premium" | "premium_discount" => candidates.push("smc_premium".to_string()),
        "inducement" => candidates.push("smc_inducement".to_string()),
        "bos" => candidates.push("smc_bos".to_string()),
        "choch" => candidates.push("smc_choch".to_string()),
        "eqh" => candidates.push("smc_eqh".to_string()),
        "eql" => candidates.push("smc_eql".to_string()),
        "displacement" => candidates.push("smc_displacement".to_string()),
        _ => {}
    }
    for (idx, name) in feature_names.iter().enumerate() {
        let raw = normalize_indicator_name(name.strip_prefix("ta_").unwrap_or(name));
        for cand in candidates.iter() {
            if raw == *cand || raw.starts_with(&format!("{}_", cand)) {
                return Some(idx);
            }
        }
    }
    None
}

fn causal_tanh_zscore_column(data: &Array2<f32>, col_idx: usize, min_periods: usize) -> Vec<f64> {
    let n = data.nrows();
    let mut out = vec![0.0_f64; n];
    if n == 0 {
        return out;
    }
    let needed = min_periods.max(2);
    let mut count: usize = 0;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    for r in 0..n {
        let v = data[(r, col_idx)] as f64;
        if count >= needed {
            let var = m2 / (count.max(1) as f64);
            let std = if var > 0.0 { var.sqrt() } else { 0.0 };
            let z = if std > 1e-12 {
                (v - mean) / std
            } else {
                v - mean
            };
            if z.is_finite() {
                out[r] = z.tanh();
            }
        }
        if !v.is_finite() {
            continue;
        }
        count += 1;
        let delta = v - mean;
        mean += delta / (count as f64);
        let delta2 = v - mean;
        m2 += delta * delta2;
    }
    out
}

fn parse_feature_profile(raw: Option<&str>, default: FeatureProfile) -> FeatureProfile {
    let Some(value) = raw else {
        return default;
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return default;
    }
    FeatureProfile::from_str(trimmed)
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
    use_bos: Option<bool>,
    use_choch: Option<bool>,
    use_eqh: Option<bool>,
    use_eql: Option<bool>,
    use_displacement: Option<bool>,
}

fn load_strategy_specs(path: &PathBuf) -> Result<Vec<StrategySpec>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read strategy catalog {}: {}", path.display(), e))?;
    let value: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("Strategy catalog JSON invalid: {}", e))?;

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

fn norm_symbol(symbol: &str) -> String {
    let alpha: String = symbol
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .map(|ch| ch.to_ascii_uppercase())
        .collect();
    if alpha.len() >= 6 {
        alpha[..6].to_string()
    } else {
        alpha
    }
}

fn split_symbol(symbol: &str) -> Option<(String, String)> {
    let sym = norm_symbol(symbol);
    if sym.len() == 6 && sym.chars().all(|c| c.is_ascii_alphabetic()) {
        Some((sym[..3].to_string(), sym[3..6].to_string()))
    } else {
        None
    }
}

fn symbol_kind(symbol: &str, parts: Option<&(String, String)>) -> &'static str {
    if let Some((base, quote)) = parts {
        if base == "XAU" || base == "XAG" {
            return "metal";
        }
        if base == "BTC" || base == "ETH" || base == "LTC" {
            return "crypto";
        }
        if base.len() == 3 && quote.len() == 3 {
            return "fx";
        }
    }
    let sym = norm_symbol(symbol);
    if sym.contains("BTC") || sym.contains("ETH") || sym.contains("LTC") {
        return "crypto";
    }
    if sym.starts_with("XAU") || sym.starts_with("XAG") {
        return "metal";
    }
    "other"
}

fn pip_size_from_parts(symbol: &str, parts: Option<&(String, String)>) -> f64 {
    match symbol_kind(symbol, parts) {
        "metal" => 0.01,
        "crypto" => 1.0,
        "fx" => {
            if let Some((_base, quote)) = parts {
                if quote == "JPY" {
                    0.01
                } else {
                    0.0001
                }
            } else {
                0.0001
            }
        }
        _ => 0.0001,
    }
}

fn contract_size_from_parts(symbol: &str, parts: Option<&(String, String)>) -> f64 {
    match symbol_kind(symbol, parts) {
        "metal" => {
            if let Some((base, _quote)) = parts {
                if base == "XAU" {
                    100.0
                } else if base == "XAG" {
                    5000.0
                } else {
                    100.0
                }
            } else {
                100.0
            }
        }
        "crypto" => 1.0,
        "fx" => 100_000.0,
        _ => 1.0,
    }
}

fn quote_to_account_rate(
    base: &str,
    quote: &str,
    account: &str,
    price: Option<f64>,
    refs: &HashMap<String, f64>,
) -> Option<f64> {
    let acc = account.to_ascii_uppercase();
    if quote == acc {
        return Some(1.0);
    }

    let px = match price {
        Some(v) if v.is_finite() && v > 0.0 => Some(v),
        _ => None,
    };

    if let Some(v) = px {
        if base == acc {
            return Some(1.0 / v);
        }
    }

    let direct_key = format!("{quote}{acc}");
    if let Some(v) = refs.get(&direct_key) {
        if v.is_finite() && *v > 0.0 {
            return Some(*v);
        }
    }

    let inverse_key = format!("{acc}{quote}");
    if let Some(v) = refs.get(&inverse_key) {
        if v.is_finite() && *v > 0.0 {
            return Some(1.0 / *v);
        }
    }

    if let Some(v) = px {
        let base_to_acc = format!("{base}{acc}");
        if let Some(bv) = refs.get(&base_to_acc) {
            if bv.is_finite() && *bv > 0.0 {
                return Some(*bv / v);
            }
        }
        let acc_to_base = format!("{acc}{base}");
        if let Some(av) = refs.get(&acc_to_base) {
            if av.is_finite() && *av > 0.0 {
                return Some(1.0 / (*av * v));
            }
        }
        return Some(1.0 / v);
    }

    None
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
        let ptv = if pt.is_finite() && pt > 0.0 {
            pt
        } else {
            0.0001
        };
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
#[pyo3(signature = (symbol, price=None, account_currency="USD", reference_prices=None))]
fn infer_pip_metrics(
    symbol: &str,
    price: Option<f64>,
    account_currency: &str,
    reference_prices: Option<&Bound<'_, PyAny>>,
) -> PyResult<(f64, f64)> {
    let parts = split_symbol(symbol);
    let pip_size = pip_size_from_parts(symbol, parts.as_ref());
    let contract_size = contract_size_from_parts(symbol, parts.as_ref());
    let pip_value_quote = pip_size * contract_size;

    let mut refs: HashMap<String, f64> = HashMap::new();
    if let Some(raw) = reference_prices {
        if let Ok(dict) = raw.cast::<PyDict>() {
            for (k, v) in dict.iter() {
                let key = match k.extract::<String>() {
                    Ok(s) => norm_symbol(&s),
                    Err(_) => continue,
                };
                if key.len() != 6 {
                    continue;
                }
                let val = match v.extract::<f64>() {
                    Ok(x) if x.is_finite() && x > 0.0 => x,
                    _ => continue,
                };
                refs.insert(key, val);
            }
        }
    }

    let mut pip_value = pip_value_quote;
    if let Some((base, quote)) = parts {
        let rate = quote_to_account_rate(&base, &quote, account_currency, price, &refs);
        if let Some(r) = rate {
            if r.is_finite() && r > 0.0 {
                pip_value = pip_value_quote * r;
            }
        }
    }

    if !pip_value.is_finite() || pip_value <= 0.0 {
        pip_value = pip_value_quote.max(1e-6);
    }
    Ok((pip_size, pip_value))
}

#[pyfunction]
#[pyo3(signature = (index_ns))]
fn derive_time_index_arrays<'py>(
    py: Python<'py>,
    index_ns: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let ns_vec = vec_from_py_i64(&index_ns);
    let n = ns_vec.len();
    let mut unix_ms: Vec<i64> = Vec::with_capacity(n);
    let mut month_idx: Vec<i64> = Vec::with_capacity(n);
    let mut day_idx: Vec<i64> = Vec::with_capacity(n);

    for (i, ns) in ns_vec.iter().copied().enumerate() {
        let ms = ns / 1_000_000;
        unix_ms.push(ms);
        if let Some(dt) = Utc.timestamp_millis_opt(ms).single() {
            let y = dt.year() as i64;
            let m = dt.month() as i64;
            let d = dt.day() as i64;
            month_idx.push(y * 12 + m);
            day_idx.push(y * 10_000 + m * 100 + d);
        } else {
            let seq = i as i64;
            month_idx.push(seq);
            day_idx.push(seq);
        }
    }

    Ok((
        unix_ms.into_pyarray(py),
        month_idx.into_pyarray(py),
        day_idx.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (index_ns))]
fn count_weekday_trading_days(index_ns: PyReadonlyArray1<'_, i64>) -> PyResult<usize> {
    const NS_PER_DAY: i64 = 86_400_000_000_000;

    let ns_vec = vec_from_py_i64(&index_ns);
    let mut uniq_days: HashSet<i64> = HashSet::with_capacity(ns_vec.len());

    for ns in ns_vec {
        let day_num = ns.div_euclid(NS_PER_DAY);
        let weekday = (day_num + 3).rem_euclid(7);
        if weekday < 5 {
            uniq_days.insert(day_num);
        }
    }

    Ok(uniq_days.len())
}

#[pyfunction]
#[pyo3(signature = (src_idx_ns, src_vals, tgt_idx_ns, fill=0.0))]
fn align_ffill_values_by_ns<'py>(
    py: Python<'py>,
    src_idx_ns: PyReadonlyArray1<'py, i64>,
    src_vals: PyReadonlyArray1<'py, f64>,
    tgt_idx_ns: PyReadonlyArray1<'py, i64>,
    fill: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let src_idx = vec_from_py_i64(&src_idx_ns);
    let src_vals_vec = vec_from_py_f64(&src_vals);
    let tgt_idx = vec_from_py_i64(&tgt_idx_ns);

    let mut out = vec![fill; tgt_idx.len()];
    if src_idx.is_empty() || src_vals_vec.is_empty() || tgt_idx.is_empty() {
        return Ok(out.into_pyarray(py));
    }

    let m = src_idx.len().min(src_vals_vec.len());
    let mut pairs: Vec<(i64, f64)> = Vec::with_capacity(m);
    for i in 0..m {
        pairs.push((src_idx[i], src_vals_vec[i]));
    }
    pairs.sort_by_key(|(ts, _)| *ts);

    let mut sorted_idx: Vec<i64> = Vec::with_capacity(m);
    let mut sorted_vals: Vec<f64> = Vec::with_capacity(m);
    for (ts, v) in pairs {
        sorted_idx.push(ts);
        sorted_vals.push(v);
    }

    for (i, t_ref) in tgt_idx.iter().enumerate() {
        let t = *t_ref;
        let p = sorted_idx.partition_point(|x| *x <= t);
        if p > 0 {
            let v = sorted_vals[p - 1];
            out[i] = if v.is_finite() { v } else { fill };
        }
    }

    for v in out.iter_mut() {
        if !v.is_finite() {
            *v = fill;
        }
    }

    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (src_idx_ns, src_vals, tgt_idx_ns, fill=0.0))]
fn align_exact_values_by_ns<'py>(
    py: Python<'py>,
    src_idx_ns: PyReadonlyArray1<'py, i64>,
    src_vals: PyReadonlyArray1<'py, f64>,
    tgt_idx_ns: PyReadonlyArray1<'py, i64>,
    fill: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let src_idx = vec_from_py_i64(&src_idx_ns);
    let src_vals_vec = vec_from_py_f64(&src_vals);
    let tgt_idx = vec_from_py_i64(&tgt_idx_ns);

    let mut out = vec![fill; tgt_idx.len()];
    if src_idx.is_empty() || src_vals_vec.is_empty() || tgt_idx.is_empty() {
        return Ok(out.into_pyarray(py));
    }

    let m = src_idx.len().min(src_vals_vec.len());
    let mut pairs: Vec<(i64, f64)> = Vec::with_capacity(m);
    for i in 0..m {
        pairs.push((src_idx[i], src_vals_vec[i]));
    }
    pairs.sort_by_key(|(ts, _)| *ts);

    let mut sorted_idx: Vec<i64> = Vec::with_capacity(m);
    let mut sorted_vals: Vec<f64> = Vec::with_capacity(m);
    for (ts, v) in pairs {
        sorted_idx.push(ts);
        sorted_vals.push(v);
    }

    for (i, t_ref) in tgt_idx.iter().enumerate() {
        let t = *t_ref;
        let p = sorted_idx.partition_point(|x| *x < t);
        if p < sorted_idx.len() && sorted_idx[p] == t {
            let v = sorted_vals[p];
            out[i] = if v.is_finite() { v } else { fill };
        }
    }

    for v in out.iter_mut() {
        if !v.is_finite() {
            *v = fill;
        }
    }

    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (src_matrix, src_col_idx, dst_col_idx, dst_width))]
fn align_feature_matrix<'py>(
    py: Python<'py>,
    src_matrix: PyReadonlyArray2<'py, f32>,
    src_col_idx: PyReadonlyArray1<'py, i64>,
    dst_col_idx: PyReadonlyArray1<'py, i64>,
    dst_width: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let src = src_matrix.as_array();
    let rows = src.nrows();
    let src_width = src.ncols();
    let src_idx = vec_from_py_i64(&src_col_idx);
    let dst_idx = vec_from_py_i64(&dst_col_idx);

    if rows == 0 || dst_width == 0 || src_idx.is_empty() || dst_idx.is_empty() {
        return Ok(Array2::<f32>::zeros((rows, dst_width)).into_pyarray(py));
    }

    let n = src_idx.len().min(dst_idx.len());
    let mut out = Array2::<f32>::zeros((rows, dst_width));
    for i in 0..n {
        let s = src_idx[i];
        let d = dst_idx[i];
        if s < 0 || d < 0 {
            continue;
        }
        let su = s as usize;
        let du = d as usize;
        if su >= src_width || du >= dst_width {
            continue;
        }
        let src_col = src.column(su);
        let mut dst_col = out.column_mut(du);
        dst_col.assign(&src_col);
    }

    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (idx_ns))]
fn sorted_index_order<'py>(
    py: Python<'py>,
    idx_ns: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let idx_vec = vec_from_py_i64(&idx_ns);
    let rows = idx_vec.len();
    if rows == 0 {
        return Ok(Vec::<i64>::new().into_pyarray(py));
    }

    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by_key(|&i| idx_vec[i]);
    let out: Vec<i64> = order.into_iter().map(|i| i as i64).collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (scores, absolute=false))]
fn rank_scores_desc<'py>(
    py: Python<'py>,
    scores: PyReadonlyArray1<'py, f64>,
    absolute: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let score_vec = vec_from_py_f64(&scores);
    let rows = score_vec.len();
    if rows == 0 {
        return Ok(Vec::<i64>::new().into_pyarray(py));
    }

    let rank_key = |value: f64| {
        if !value.is_finite() {
            f64::NEG_INFINITY
        } else if absolute {
            value.abs()
        } else {
            value
        }
    };

    let keys: Vec<f64> = score_vec.into_iter().map(rank_key).collect();
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&i, &j| keys[j].total_cmp(&keys[i]));

    let out: Vec<i64> = order.into_iter().map(|i| i as i64).collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (base_idx_ns, event_idx_ns, event_sent, event_conf, lookback_ns))]
fn aggregate_news_features<'py>(
    py: Python<'py>,
    base_idx_ns: PyReadonlyArray1<'py, i64>,
    event_idx_ns: PyReadonlyArray1<'py, i64>,
    event_sent: PyReadonlyArray1<'py, f64>,
    event_conf: PyReadonlyArray1<'py, f64>,
    lookback_ns: i64,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let base_idx = vec_from_py_i64(&base_idx_ns);
    let event_idx = vec_from_py_i64(&event_idx_ns);
    let event_sent_vec = vec_from_py_f64(&event_sent);
    let event_conf_vec = vec_from_py_f64(&event_conf);
    let n = base_idx.len();

    let mut out_sent = vec![0.0_f32; n];
    let mut out_conf = vec![0.0_f32; n];
    let mut out_count = vec![0.0_f32; n];
    let mut out_recency = vec![9999.0_f32; n];

    if n == 0 || event_idx.is_empty() || event_sent_vec.is_empty() || event_conf_vec.is_empty() {
        return Ok((
            out_sent.into_pyarray(py),
            out_conf.into_pyarray(py),
            out_count.into_pyarray(py),
            out_recency.into_pyarray(py),
        ));
    }

    let m = event_idx
        .len()
        .min(event_sent_vec.len())
        .min(event_conf_vec.len());
    if m == 0 {
        return Ok((
            out_sent.into_pyarray(py),
            out_conf.into_pyarray(py),
            out_count.into_pyarray(py),
            out_recency.into_pyarray(py),
        ));
    }

    let mut pairs: Vec<(i64, f64, f64)> = Vec::with_capacity(m);
    for i in 0..m {
        pairs.push((event_idx[i], event_sent_vec[i], event_conf_vec[i]));
    }
    pairs.sort_by_key(|(ts, _, _)| *ts);

    let mut uniq_ns: Vec<i64> = Vec::new();
    let mut sent_mean: Vec<f64> = Vec::new();
    let mut conf_mean: Vec<f64> = Vec::new();

    let mut cur_ts = pairs[0].0;
    let mut sent_sum = 0.0_f64;
    let mut conf_sum = 0.0_f64;
    let mut count = 0_usize;
    for (ts, sent, conf) in pairs {
        if ts != cur_ts && count > 0 {
            uniq_ns.push(cur_ts);
            sent_mean.push(sent_sum / count as f64);
            conf_mean.push(conf_sum / count as f64);
            cur_ts = ts;
            sent_sum = 0.0;
            conf_sum = 0.0;
            count = 0;
        }
        sent_sum += sent;
        conf_sum += conf;
        count += 1;
    }
    if count > 0 {
        uniq_ns.push(cur_ts);
        sent_mean.push(sent_sum / count as f64);
        conf_mean.push(conf_sum / count as f64);
    }

    let lookback = lookback_ns.max(0);
    for (i, base_ts) in base_idx.iter().enumerate() {
        let right = uniq_ns.partition_point(|x| *x <= *base_ts);
        if right > 0 {
            let prev = right - 1;
            out_sent[i] = sent_mean[prev] as f32;
            out_conf[i] = conf_mean[prev] as f32;
            let recency_ns = (*base_ts - uniq_ns[prev]).max(0);
            out_recency[i] = (recency_ns as f64 / 60_000_000_000.0) as f32;
        }
        let left_bound = base_ts.saturating_sub(lookback);
        let left = uniq_ns.partition_point(|x| *x < left_bound);
        out_count[i] = (right.saturating_sub(left)) as f32;
    }

    Ok((
        out_sent.into_pyarray(py),
        out_conf.into_pyarray(py),
        out_count.into_pyarray(py),
        out_recency.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (base_idx_ns, event_idx_ns, event_sent, event_conf, back_ns, fwd_ns))]
fn aggregate_news_activation<'py>(
    py: Python<'py>,
    base_idx_ns: PyReadonlyArray1<'py, i64>,
    event_idx_ns: PyReadonlyArray1<'py, i64>,
    event_sent: PyReadonlyArray1<'py, f64>,
    event_conf: PyReadonlyArray1<'py, f64>,
    back_ns: i64,
    fwd_ns: i64,
) -> PyResult<(
    Bound<'py, PyArray1<i8>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let base_idx = vec_from_py_i64(&base_idx_ns);
    let event_idx = vec_from_py_i64(&event_idx_ns);
    let event_sent_vec = vec_from_py_f64(&event_sent);
    let event_conf_vec = vec_from_py_f64(&event_conf);
    let n = base_idx.len();

    let mut nearby = vec![0_i8; n];
    let mut conf_max = vec![0.0_f32; n];
    let mut sent_max = vec![0.0_f32; n];

    if n == 0 || event_idx.is_empty() || event_sent_vec.is_empty() || event_conf_vec.is_empty() {
        return Ok((
            nearby.into_pyarray(py),
            conf_max.into_pyarray(py),
            sent_max.into_pyarray(py),
        ));
    }

    let m = event_idx
        .len()
        .min(event_sent_vec.len())
        .min(event_conf_vec.len());
    if m == 0 {
        return Ok((
            nearby.into_pyarray(py),
            conf_max.into_pyarray(py),
            sent_max.into_pyarray(py),
        ));
    }

    let mut pairs: Vec<(i64, f64, f64)> = Vec::with_capacity(m);
    for i in 0..m {
        pairs.push((event_idx[i], event_sent_vec[i], event_conf_vec[i]));
    }
    pairs.sort_by_key(|(ts, _, _)| *ts);

    let mut sorted_ns: Vec<i64> = Vec::with_capacity(m);
    let mut sorted_sent: Vec<f64> = Vec::with_capacity(m);
    let mut sorted_conf: Vec<f64> = Vec::with_capacity(m);
    for (ts, sent, conf) in pairs {
        sorted_ns.push(ts);
        sorted_sent.push(sent);
        sorted_conf.push(conf);
    }

    let back = back_ns.max(0);
    let fwd = fwd_ns.max(0);
    for (i, base_ts) in base_idx.iter().enumerate() {
        let left_bound = base_ts.saturating_sub(back);
        let right_bound = base_ts.saturating_add(fwd);
        let left = sorted_ns.partition_point(|x| *x < left_bound);
        let right = sorted_ns.partition_point(|x| *x <= right_bound);
        if right <= left {
            continue;
        }
        nearby[i] = 1;
        let mut max_conf = f64::NEG_INFINITY;
        let mut max_sent = f64::NEG_INFINITY;
        for j in left..right {
            max_conf = max_conf.max(sorted_conf[j]);
            max_sent = max_sent.max(sorted_sent[j]);
        }
        conf_max[i] = if max_conf.is_finite() {
            max_conf as f32
        } else {
            0.0
        };
        sent_max[i] = if max_sent.is_finite() {
            max_sent as f32
        } else {
            0.0
        };
    }

    Ok((
        nearby.into_pyarray(py),
        conf_max.into_pyarray(py),
        sent_max.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (close_prices, adx_values=None, volatility_window=20))]
fn extract_regime_features<'py>(
    py: Python<'py>,
    close_prices: PyReadonlyArray1<'py, f64>,
    adx_values: Option<PyReadonlyArray1<'py, f64>>,
    volatility_window: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let close_vec = vec_from_py_f64(&close_prices);
    let n = close_vec.len();
    if n < 3 {
        return Ok(Array2::<f32>::zeros((0, 3)).into_pyarray(py));
    }

    let mut returns = vec![0.0_f32; n];
    for i in 1..n {
        let prev = close_vec[i - 1];
        let curr = close_vec[i];
        let ratio = if prev.abs() > 1e-12 { curr / prev } else { 1.0 };
        let ret = ratio.max(1e-12).ln() as f32;
        returns[i] = if ret.is_finite() { ret } else { 0.0 };
    }

    let w = volatility_window.max(1);
    let mut volatility = vec![0.0_f32; n];
    if n >= w {
        let mut c1 = vec![0.0_f64; n + 1];
        let mut c2 = vec![0.0_f64; n + 1];
        for i in 0..n {
            let v = returns[i] as f64;
            c1[i + 1] = c1[i] + v;
            c2[i + 1] = c2[i] + v * v;
        }
        for i in (w - 1)..n {
            let start = i + 1 - w;
            let sum_w = c1[i + 1] - c1[start];
            let sq_w = c2[i + 1] - c2[start];
            let mean_w = sum_w / w as f64;
            let var_w = ((sq_w / w as f64) - (mean_w * mean_w)).max(0.0);
            volatility[i] = var_w.sqrt() as f32;
        }
    }

    let mut adx: Vec<f32> = vec![0.0; n];
    if let Some(adx_arr) = adx_values {
        let adx_vec = vec_from_py_f64(&adx_arr);
        if adx_vec.is_empty() {
            for i in 0..n {
                adx[i] = volatility[i] * 100.0;
            }
        } else {
            for i in 0..n {
                let v = adx_vec[i % adx_vec.len()] as f32;
                adx[i] = if v.is_finite() { v } else { 0.0 };
            }
        }
    } else {
        for i in 0..n {
            adx[i] = volatility[i] * 100.0;
        }
    }

    let mut out = Array2::<f32>::zeros((n - 1, 3));
    for i in 0..(n - 1) {
        out[(i, 0)] = if returns[i].is_finite() {
            returns[i]
        } else {
            0.0
        };
        out[(i, 1)] = if volatility[i].is_finite() {
            volatility[i]
        } else {
            0.0
        };
        out[(i, 2)] = if adx[i].is_finite() { adx[i] } else { 0.0 };
    }

    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (labels))]
fn remap_labels_neutral_buy_sell<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let input = vec_from_py_i64(&labels);
    let out: Vec<i64> = input
        .into_iter()
        .map(|value| if value == -1 { 2 } else { value.clamp(0, 2) })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (labels))]
fn remap_labels_sell_neutral_buy<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let input = vec_from_py_i64(&labels);
    let out: Vec<i64> = input
        .into_iter()
        .map(|value| match value {
            -1 => 0,
            0 => 1,
            1 => 2,
            _ => 0,
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (probs, classes=None))]
fn pad_probs_neutral_buy_sell<'py>(
    py: Python<'py>,
    probs: PyReadonlyArrayDyn<'py, f64>,
    classes: Option<Vec<i64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = probs.as_array();
    let mut out = match view.ndim() {
        1 => Array2::<f64>::zeros((view.len(), 3)),
        2 => {
            let rows = view.shape()[0];
            Array2::<f64>::zeros((rows, 3))
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "probs must be 1D or 2D",
            ))
        }
    };

    match view.ndim() {
        1 => {
            let arr = view.into_dimensionality::<Ix1>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("probs must be 1D or 2D")
            })?;
            for r in 0..arr.len() {
                let value = arr[r];
                out[(r, 0)] = 1.0 - value;
                out[(r, 1)] = value;
            }
        }
        2 => {
            let arr = view.into_dimensionality::<Ix2>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("probs must be 1D or 2D")
            })?;
            let rows = arr.nrows();
            let cols = arr.ncols();
            if let Some(class_map) = classes.as_ref() {
                if class_map.len() == cols {
                    for (col, cls_val) in class_map.iter().copied().enumerate() {
                        match cls_val {
                            0 => {
                                for r in 0..rows {
                                    out[(r, 0)] = arr[(r, col)];
                                }
                            }
                            1 => {
                                for r in 0..rows {
                                    out[(r, 1)] = arr[(r, col)];
                                }
                            }
                            -1 | 2 => {
                                for r in 0..rows {
                                    out[(r, 2)] = arr[(r, col)];
                                }
                            }
                            _ => {}
                        }
                    }
                    return Ok(out.into_pyarray(py));
                }
            }
            if cols == 3 {
                for r in 0..rows {
                    out[(r, 0)] = arr[(r, 0)];
                    out[(r, 1)] = arr[(r, 1)];
                    out[(r, 2)] = arr[(r, 2)];
                }
            } else if cols == 2 {
                for r in 0..rows {
                    out[(r, 0)] = arr[(r, 0)];
                    out[(r, 1)] = arr[(r, 1)];
                }
            } else if cols >= 1 {
                for r in 0..rows {
                    let value = arr[(r, 0)];
                    out[(r, 0)] = 1.0 - value;
                    out[(r, 1)] = value;
                }
            }
        }
        _ => {}
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (decision))]
fn margins_to_probs<'py>(
    py: Python<'py>,
    decision: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = decision.as_array();
    match view.ndim() {
        1 => {
            let arr = view.into_dimensionality::<Ix1>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("decision must be 1D or 2D")
            })?;
            let n = arr.len();
            let mut out = Array2::<f64>::zeros((n, 3));
            for (i, value) in arr.iter().copied().enumerate() {
                let clipped = if value.is_nan() {
                    f64::NAN
                } else {
                    value.max(-30.0).min(30.0)
                };
                let p1 = 1.0 / (1.0 + (-clipped).exp());
                out[(i, 0)] = 1.0 - p1;
                out[(i, 1)] = p1;
                out[(i, 2)] = 0.0;
            }
            Ok(out.into_pyarray(py))
        }
        2 => {
            let arr = view.into_dimensionality::<Ix2>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("decision must be 1D or 2D")
            })?;
            let rows = arr.nrows();
            let cols = arr.ncols();
            let mut out = Array2::<f64>::zeros((rows, cols));
            for r in 0..rows {
                let row = arr.row(r);
                let mut row_max = f64::NEG_INFINITY;
                let mut has_nan = false;
                for value in row.iter().copied() {
                    if value.is_nan() {
                        has_nan = true;
                        break;
                    }
                    row_max = row_max.max(value);
                }
                if has_nan {
                    for c in 0..cols {
                        out[(r, c)] = f64::NAN;
                    }
                    continue;
                }
                let mut sum = 0.0_f64;
                for c in 0..cols {
                    let shifted = row[c] - row_max;
                    let clipped = if shifted.is_nan() {
                        f64::NAN
                    } else {
                        shifted.max(-30.0).min(30.0)
                    };
                    let ex = clipped.exp();
                    out[(r, c)] = ex;
                    sum += ex;
                }
                let denom = if sum <= 0.0 { 1.0 } else { sum };
                for c in 0..cols {
                    out[(r, c)] /= denom;
                }
            }
            Ok(out.into_pyarray(py))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "decision must be 1D or 2D",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (probs))]
fn probs_to_signals<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let arr = probs.as_array();
    let rows = arr.nrows();
    let cols = arr.ncols();
    let mut out = vec![0_i8; rows];
    if cols == 0 {
        return Ok(out.into_pyarray(py));
    }

    for r in 0..rows {
        let mut best_idx = 0usize;
        let mut best_val = arr[(r, 0)];
        for c in 1..cols {
            let value = arr[(r, c)];
            if value > best_val {
                best_val = value;
                best_idx = c;
            }
        }
        out[r] = match best_idx {
            1 => 1,
            2 => -1,
            _ => 0,
        };
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (probs, conf_threshold, y_true=None))]
fn threshold_signals_and_accuracy<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray2<'py, f64>,
    conf_threshold: f64,
    y_true: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(Bound<'py, PyArray1<i8>>, f64)> {
    let arr = probs.as_array();
    let rows = arr.nrows();
    let cols = arr.ncols();
    let mut out = vec![0_i8; rows];
    if cols == 0 {
        return Ok((out.into_pyarray(py), 0.0));
    }

    for r in 0..rows {
        let buy = if cols > 1 { arr[(r, 1)] } else { 0.0 };
        let sell = if cols > 2 { arr[(r, 2)] } else { 0.0 };
        if !buy.is_finite() || !sell.is_finite() {
            continue;
        }
        let trade_prob = if buy >= sell { buy } else { sell };
        if trade_prob >= conf_threshold {
            out[r] = if buy >= sell { 1 } else { -1 };
        }
    }

    let accuracy = if let Some(labels) = y_true {
        let label_vec = vec_from_py_i64(&labels);
        let n = rows.min(label_vec.len());
        if n == 0 {
            0.0
        } else {
            let mut correct = 0usize;
            for i in 0..n {
                let expected = match label_vec[i] {
                    1 => 1_i8,
                    -1 | 2 => -1_i8,
                    _ => 0_i8,
                };
                if out[i] == expected {
                    correct += 1;
                }
            }
            correct as f64 / n as f64
        }
    } else {
        0.0
    };

    Ok((out.into_pyarray(py), accuracy))
}

#[pyfunction]
#[pyo3(signature = (
    net_profit,
    sortino,
    drawdown,
    profit_factor,
    trades,
    daily_dd,
    months,
    dd_limit,
    daily_dd_limit,
    min_monthly,
    initial_balance,
    acc,
    prop_weight,
    acc_weight,
    win_rate=None,
    include_win_rate_bonus=false,
    ignore_zero_trade_entries=true
))]
fn aggregate_prop_score_metrics(
    net_profit: PyReadonlyArray1<'_, f64>,
    sortino: PyReadonlyArray1<'_, f64>,
    drawdown: PyReadonlyArray1<'_, f64>,
    profit_factor: PyReadonlyArray1<'_, f64>,
    trades: PyReadonlyArray1<'_, f64>,
    daily_dd: PyReadonlyArray1<'_, f64>,
    months: f64,
    dd_limit: f64,
    daily_dd_limit: f64,
    min_monthly: f64,
    initial_balance: f64,
    acc: f64,
    prop_weight: f64,
    acc_weight: f64,
    win_rate: Option<PyReadonlyArray1<'_, f64>>,
    include_win_rate_bonus: bool,
    ignore_zero_trade_entries: bool,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64)> {
    let net_profit_vec = vec_from_py_f64(&net_profit);
    let sortino_vec = vec_from_py_f64(&sortino);
    let drawdown_vec = vec_from_py_f64(&drawdown);
    let profit_factor_vec = vec_from_py_f64(&profit_factor);
    let trades_vec = vec_from_py_f64(&trades);
    let daily_dd_vec = vec_from_py_f64(&daily_dd);
    let win_rate_vec = win_rate.as_ref().map(vec_from_py_f64);

    let mut n = net_profit_vec.len();
    n = n.min(sortino_vec.len());
    n = n.min(drawdown_vec.len());
    n = n.min(profit_factor_vec.len());
    n = n.min(trades_vec.len());
    n = n.min(daily_dd_vec.len());
    if include_win_rate_bonus {
        if let Some(ref values) = win_rate_vec {
            n = n.min(values.len());
        }
    }

    let penalty_factor = |drawdown_value: f64, daily_dd_value: f64, monthly_ret: f64| -> f64 {
        let mut penalty = 1.0_f64;
        if dd_limit > 0.0 && drawdown_value > dd_limit {
            let excess = (drawdown_value - dd_limit).max(0.0);
            penalty *= (1.0 - (excess / dd_limit)).max(0.1);
        }
        if daily_dd_limit > 0.0 && daily_dd_value > daily_dd_limit {
            let excess = (daily_dd_value - daily_dd_limit).max(0.0);
            penalty *= (1.0 - (excess / daily_dd_limit)).max(0.1);
        }
        if min_monthly > 0.0 && monthly_ret < min_monthly {
            penalty *= (monthly_ret / min_monthly).max(0.1);
        }
        penalty
    };

    let mut total_trades = 0.0_f64;
    let mut aggregation_weight = 0.0_f64;
    let mut weighted_score = 0.0_f64;
    let mut weighted_monthly = 0.0_f64;
    let mut weighted_sortino = 0.0_f64;
    let mut weighted_calmar = 0.0_f64;
    let mut weighted_pf = 0.0_f64;
    let mut max_dd = 0.0_f64;
    let mut max_daily = 0.0_f64;

    for i in 0..n {
        let dd_value = drawdown_vec[i];
        let daily_dd_value = daily_dd_vec[i];
        if dd_value > max_dd {
            max_dd = dd_value;
        }
        if daily_dd_value > max_daily {
            max_daily = daily_dd_value;
        }

        let trade_count = trades_vec[i];
        let weight = if trade_count > 0.0 {
            trade_count
        } else if ignore_zero_trade_entries {
            0.0
        } else {
            1.0
        };
        if weight <= 0.0 {
            continue;
        }

        let monthly_ret = (net_profit_vec[i] / initial_balance) / months;
        let calmar = if dd_value > 1e-9 {
            monthly_ret / dd_value
        } else {
            0.0
        };
        let mut prop_score =
            monthly_ret * 100.0 + 0.6 * sortino_vec[i] + 0.4 * calmar + 0.2 * profit_factor_vec[i]
                - 50.0 * (dd_value - dd_limit).max(0.0);
        if include_win_rate_bonus {
            if let Some(ref values) = win_rate_vec {
                prop_score += 0.1 * (values[i] * 100.0);
            }
        }
        prop_score *= penalty_factor(dd_value, daily_dd_value, monthly_ret);

        if trade_count > 0.0 {
            total_trades += trade_count;
        }
        aggregation_weight += weight;
        weighted_score += weight * prop_score;
        weighted_monthly += weight * monthly_ret;
        weighted_sortino += weight * sortino_vec[i];
        weighted_calmar += weight * calmar;
        weighted_pf += weight * profit_factor_vec[i];
    }

    let denom = aggregation_weight.max(1.0);
    Ok((
        prop_weight * (weighted_score / denom) + acc_weight * acc,
        max_dd,
        max_daily,
        total_trades,
        weighted_monthly / denom,
        weighted_sortino / denom,
        weighted_calmar / denom,
        weighted_pf / denom,
    ))
}

#[pyfunction]
#[pyo3(signature = (labels))]
fn balanced_class_weights<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
    let input = vec_from_py_i64(&labels);
    let n_samples = input.len();
    if n_samples == 0 {
        return Ok((
            Vec::<i64>::new().into_pyarray(py),
            Vec::<f64>::new().into_pyarray(py),
        ));
    }

    let mut counts: HashMap<i64, usize> = HashMap::new();
    for value in input {
        *counts.entry(value).or_insert(0) += 1;
    }

    let mut classes: Vec<i64> = counts.keys().copied().collect();
    classes.sort_unstable();
    let n_classes = classes.len().max(1) as f64;
    let weights: Vec<f64> = classes
        .iter()
        .map(|cls| {
            let count = *counts.get(cls).unwrap_or(&0) as f64;
            if count > 0.0 {
                n_samples as f64 / (n_classes * count)
            } else {
                0.0
            }
        })
        .collect();

    Ok((classes.into_pyarray(py), weights.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (labels))]
fn sample_weights_from_labels<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let input = vec_from_py_i64(&labels);
    let n_samples = input.len();
    if n_samples == 0 {
        return Ok(Vec::<f32>::new().into_pyarray(py));
    }

    let mut counts: HashMap<i64, usize> = HashMap::new();
    for value in input.iter().copied() {
        *counts.entry(value).or_insert(0) += 1;
    }
    let n_classes = counts.len().max(1) as f64;
    let out: Vec<f32> = input
        .into_iter()
        .map(|value| {
            let count = *counts.get(&value).unwrap_or(&0) as f64;
            if count > 0.0 {
                (n_samples as f64 / (n_classes * count)) as f32
            } else {
                0.0
            }
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close_prices, signals))]
fn quick_backtest_metrics<'py>(
    _py: Python<'py>,
    close_prices: PyReadonlyArray1<'py, f64>,
    signals: PyReadonlyArray1<'py, i8>,
) -> PyResult<(f64, f64, f64, i64)> {
    let close_vec = vec_from_py_f64(&close_prices);
    let signal_vec = vec_from_py_i8(&signals);
    let n = close_vec.len().min(signal_vec.len());
    if n <= 1 {
        return Ok((0.0, 0.0, 0.0, 0));
    }

    let mut pnl_sum = 0.0_f64;
    let mut wins = 0usize;
    let mut correct = 0usize;
    let trades = signal_vec.iter().filter(|&&sig| sig != 0).count();
    for i in 0..(n - 1) {
        let sig = signal_vec[i];
        let ret = close_vec[i + 1] - close_vec[i];
        let sign = if ret > 0.0 {
            1_i8
        } else if ret < 0.0 {
            -1_i8
        } else {
            0_i8
        };
        if sig == sign {
            correct += 1;
        }
        if sig != 0 {
            let pnl = if sig == 1 {
                if ret > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            } else if ret < 0.0 {
                1.0
            } else {
                -1.0
            };
            pnl_sum += pnl;
            if pnl > 0.0 {
                wins += 1;
            }
        }
    }

    let steps = (n - 1) as f64;
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
#[pyo3(signature = (x, y, idx_ns))]
fn sort_rows_with_labels_by_index<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    y: PyReadonlyArray1<'py, i64>,
    idx_ns: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let x_arr = x.as_array();
    let y_vec = vec_from_py_i64(&y);
    let idx_vec = vec_from_py_i64(&idx_ns);
    let rows = x_arr.nrows().min(y_vec.len()).min(idx_vec.len());
    let cols = x_arr.ncols();

    if rows == 0 {
        return Ok((
            Array2::<f32>::zeros((0, cols)).into_pyarray(py),
            Vec::<i64>::new().into_pyarray(py),
            Vec::<i64>::new().into_pyarray(py),
        ));
    }

    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by_key(|&i| idx_vec[i]);

    let mut out_x = Array2::<f32>::zeros((rows, cols));
    let mut out_y: Vec<i64> = Vec::with_capacity(rows);
    let mut out_idx: Vec<i64> = Vec::with_capacity(rows);
    for (dst_i, src_i) in order.into_iter().enumerate() {
        out_idx.push(idx_vec[src_i]);
        out_y.push(y_vec[src_i]);
        for c in 0..cols {
            out_x[(dst_i, c)] = x_arr[(src_i, c)];
        }
    }

    Ok((
        out_x.into_pyarray(py),
        out_y.into_pyarray(py),
        out_idx.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, idx_ns))]
fn sort_dedup_rows_by_index<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    y: PyReadonlyArray1<'py, i8>,
    idx_ns: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<i8>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let x_arr = x.as_array();
    let y_vec = vec_from_py_i8(&y);
    let idx_vec = vec_from_py_i64(&idx_ns);
    let rows = x_arr.nrows().min(y_vec.len()).min(idx_vec.len());
    let cols = x_arr.ncols();

    if rows == 0 {
        return Ok((
            Array2::<f32>::zeros((0, cols)).into_pyarray(py),
            Vec::<i8>::new().into_pyarray(py),
            Vec::<i64>::new().into_pyarray(py),
        ));
    }

    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by_key(|&i| idx_vec[i]);

    let mut keep: Vec<usize> = Vec::with_capacity(rows);
    let mut last_idx: Option<i64> = None;
    for src_i in order {
        let ts = idx_vec[src_i];
        if last_idx.map_or(true, |prev| prev != ts) {
            keep.push(src_i);
            last_idx = Some(ts);
        }
    }

    let kept = keep.len();
    let mut out_x = Array2::<f32>::zeros((kept, cols));
    let mut out_y: Vec<i8> = Vec::with_capacity(kept);
    let mut out_idx: Vec<i64> = Vec::with_capacity(kept);
    for (dst_i, src_i) in keep.into_iter().enumerate() {
        out_idx.push(idx_vec[src_i]);
        out_y.push(y_vec[src_i]);
        for c in 0..cols {
            out_x[(dst_i, c)] = x_arr[(src_i, c)];
        }
    }

    Ok((
        out_x.into_pyarray(py),
        out_y.into_pyarray(py),
        out_idx.into_pyarray(py),
    ))
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
    let total_cost_pips =
        (spread_pips.max(0.0) + slippage_pips.max(0.0) + commission_pips.max(0.0)).max(0.0);
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
    include_raw=false,
    causal_min_bars=30
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
    arrow_tensor=false,
    feature_profile="full",
    htf_feature_profile=None,
    max_features=0,
    max_htf_features=0,
    tail_rows=0
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
    feature_profile: &str,
    htf_feature_profile: Option<&str>,
    max_features: usize,
    max_htf_features: usize,
    tail_rows: usize,
) -> PyResult<Py<PyAny>> {
    let base_profile = parse_feature_profile(Some(feature_profile), FeatureProfile::Full);
    let htf_profile = parse_feature_profile(htf_feature_profile, base_profile);
    let options = FeatureBuildOptions {
        base_profile,
        htf_profile,
        max_base_features: max_features,
        max_htf_features,
    };

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
        let dataset = if tail_rows > 0 {
            dataset.tail_rows(tail_rows)
        } else {
            dataset
        };

        let base_final = resolve_base_tf(&dataset, &base_tf);
        let refs: Vec<&str> = higher.iter().map(|s| s.as_str()).collect();
        let cache = if tail_rows > 0 {
            None
        } else {
            cache_dir
                .as_ref()
                .map(|dir| FeatureCache::new(dir, cache_ttl_minutes, cache_enabled))
        };
        let mut frame = prepare_multitimeframe_features_with_options(
            &dataset,
            &base_final,
            &refs,
            cache.as_ref(),
            &options,
        )
        .map_err(|e| format!("Feature computation failed: {}", e))?;

        if !include_raw {
            // Remove raw OHLC/volume columns when requested.
            let mut keep = Vec::new();
            for (idx, name) in frame.names.iter().enumerate() {
                let lower = name.to_ascii_lowercase();
                if lower == "open"
                    || lower == "high"
                    || lower == "low"
                    || lower == "close"
                    || lower == "volume"
                {
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

    let (frame, base_final, base) =
        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

    let dict = PyDict::new(py);
    let index_ns = frame.timestamps.clone();
    let labels = vec![0_i8; frame.data.nrows()];
    let index_ns_py = index_ns.into_pyarray(py);
    let labels_py = labels.into_pyarray(py);
    dict.set_item("timestamps", frame.timestamps)?;
    dict.set_item("index_ns", &index_ns_py)?;
    dict.set_item("feature_names", frame.names)?;
    dict.set_item("labels", &labels_py)?;
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
    resample_missing=true,
    feature_profile="full",
    htf_feature_profile=None,
    max_features=0,
    max_htf_features=0
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
    feature_profile: &str,
    htf_feature_profile: Option<&str>,
    max_features: usize,
    max_htf_features: usize,
) -> PyResult<Py<PyAny>> {
    let base_profile = parse_feature_profile(Some(feature_profile), FeatureProfile::Full);
    let htf_profile = parse_feature_profile(htf_feature_profile, base_profile);
    let options = FeatureBuildOptions {
        base_profile,
        htf_profile,
        max_base_features: max_features,
        max_htf_features,
    };

    let result: Result<
        (
            forex_data::FeatureFrame,
            String,
            Ohlcv,
            Vec<String>,
            Array2<i8>,
        ),
        String,
    > = py.detach(|| {
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
        let cache = cache_dir
            .as_ref()
            .map(|dir| FeatureCache::new(dir, cache_ttl_minutes, cache_enabled));
        let frame = prepare_multitimeframe_features_with_options(
            &dataset,
            &base_final,
            &refs,
            cache.as_ref(),
            &options,
        )
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
                use_bos: spec.use_bos.unwrap_or(false),
                use_choch: spec.use_choch.unwrap_or(false),
                use_eqh: spec.use_eqh.unwrap_or(false),
                use_eql: spec.use_eql.unwrap_or(false),
                use_displacement: spec.use_displacement.unwrap_or(false),
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

    let (frame, base_final, base, strategy_ids, signals) =
        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;
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
        let mut seen: HashSet<String> = selected.iter().map(discovery_gene_key).collect();
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
        .map(|g| {
            pythonize(py, g)
                .map(|obj| obj.into())
                .unwrap_or_else(|_| py.None())
        })
        .collect();
    let candidates_py: Vec<Py<PyAny>> = result
        .candidates
        .iter()
        .map(|g| {
            pythonize(py, g)
                .map(|obj| obj.into())
                .unwrap_or_else(|_| py.None())
        })
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
    atr_period=14,
    stop_target_mode="blend",
    signal=0,
    atr_stop_multiplier=1.5,
    min_risk_reward=2.0,
    structure_lookback_bars=120,
    structure_swing_window=2,
    structure_min_atr_mult=0.8,
    structure_max_atr_mult=4.0
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
    stop_target_mode: &str,
    signal: i8,
    atr_stop_multiplier: f64,
    min_risk_reward: f64,
    structure_lookback_bars: usize,
    structure_swing_window: usize,
    structure_min_atr_mult: f64,
    structure_max_atr_mult: f64,
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
    settings.min_risk_reward = min_risk_reward.max(0.1);
    settings.atr_stop_multiplier = atr_stop_multiplier.max(0.1);
    settings.stop_target_mode = stop_target_mode.to_string();
    settings.structure_lookback_bars = structure_lookback_bars.max(20);
    settings.structure_swing_window = structure_swing_window.max(1);
    settings.structure_min_atr_mult = structure_min_atr_mult.max(0.1);
    settings.structure_max_atr_mult = structure_max_atr_mult.max(settings.structure_min_atr_mult);
    settings.ema_fast_period = ema_fast_period.max(2);
    settings.ema_slow_period = ema_slow_period.max(settings.ema_fast_period + 1);
    settings.atr_period = atr_period.max(5);

    let out = py.detach(|| {
        Ok::<Option<(f64, f64, f64)>, String>(infer_stop_target_pips_rs(
            &open_vec, &high_vec, &low_vec, &close_vec, &settings, pip_size, signal,
        ))
    });
    out.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
}

fn _quantile(mut values: Vec<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 1 {
        return values[0];
    }
    let qv = q.clamp(0.0, 1.0);
    let pos = qv * (n.saturating_sub(1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo >= n {
        return *values.last().unwrap_or(&0.0);
    }
    if hi >= n || lo == hi {
        return values[lo];
    }
    let w = pos - (lo as f64);
    values[lo] + ((values[hi] - values[lo]) * w)
}

fn _mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / (values.len() as f64)
    }
}

fn _sum(values: &[f64]) -> f64 {
    values.iter().sum::<f64>()
}

fn _max(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| if v > acc { v } else { acc })
}

fn _month_key_from_code(code: i64) -> String {
    if code >= 100001 {
        let year = code / 100;
        let month = (code % 100).abs();
        if (1..=12).contains(&month) {
            return format!("{year:04}-{month:02}");
        }
    }
    format!("m_{code}")
}

#[derive(Clone, Copy, Default)]
struct _MonthlyAcc {
    trades: f64,
    wins: f64,
    losses: f64,
    net_profit: f64,
    swap_total: f64,
    hold_hours_total: f64,
    trade_dd_pct_total: f64,
}

#[pyfunction]
#[pyo3(signature = (
    close_prices,
    high_prices,
    low_prices,
    signals,
    month_codes,
    day_codes,
    sl_pips,
    tp_pips,
    max_hold_bars=0,
    trailing_enabled=false,
    trailing_atr_multiplier=1.0,
    trailing_be_trigger_r=1.0,
    pip_value=0.0001,
    spread_pips=1.5,
    commission_per_trade=0.0,
    pip_value_per_lot=10.0,
    history_days=0.0,
    history_months=0.0,
    bar_hours=0.0,
    timestamps_ms=None,
    swap_long_per_day=0.0,
    swap_short_per_day=0.0
))]
fn trade_journal_metrics(
    py: Python,
    close_prices: PyReadonlyArray1<f64>,
    high_prices: PyReadonlyArray1<f64>,
    low_prices: PyReadonlyArray1<f64>,
    signals: PyReadonlyArray1<i8>,
    month_codes: PyReadonlyArray1<i64>,
    day_codes: PyReadonlyArray1<i64>,
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
    history_days: f64,
    history_months: f64,
    bar_hours: f64,
    timestamps_ms: Option<PyReadonlyArray1<i64>>,
    swap_long_per_day: f64,
    swap_short_per_day: f64,
) -> PyResult<Py<PyAny>> {
    let close_vec = vec_from_py_f64(&close_prices);
    let high_vec = vec_from_py_f64(&high_prices);
    let low_vec = vec_from_py_f64(&low_prices);
    let sig_vec = vec_from_py_i8(&signals);
    let month_vec = vec_from_py_i64(&month_codes);
    let day_vec = vec_from_py_i64(&day_codes);
    let ts_vec = timestamps_ms.as_ref().map(vec_from_py_i64);

    let n = close_vec.len();
    if n <= 1 {
        let obj = pythonize(
            py,
            &serde_json::json!({"computed": false, "reason": "insufficient_rows"}),
        )?;
        return Ok(obj.into());
    }
    if high_vec.len() != n || low_vec.len() != n || sig_vec.len() != n {
        let obj = pythonize(
            py,
            &serde_json::json!({"computed": false, "reason": "shape_mismatch"}),
        )?;
        return Ok(obj.into());
    }
    if month_vec.len() != n || day_vec.len() != n {
        let obj = pythonize(
            py,
            &serde_json::json!({"computed": false, "reason": "time_index_mismatch"}),
        )?;
        return Ok(obj.into());
    }
    if let Some(ref ts) = ts_vec {
        if ts.len() != n {
            let obj = pythonize(
                py,
                &serde_json::json!({"computed": false, "reason": "timestamp_mismatch"}),
            )?;
            return Ok(obj.into());
        }
    }

    let pv = pip_value.abs().max(1e-12);
    let cash_per_pip = pip_value_per_lot;
    let bar_h = bar_hours.max(1e-9);
    let sl = sl_pips.max(0.0);
    let tp = tp_pips.max(0.0);
    if sl <= 0.0 || tp <= 0.0 {
        let obj = pythonize(
            py,
            &serde_json::json!({"computed": false, "reason": "invalid_sl_tp"}),
        )?;
        return Ok(obj.into());
    }

    let mut in_position: i8 = 0;
    let mut entry_price = 0.0_f64;
    let mut entry_i: i64 = -1;
    let mut trail_price = 0.0_f64;
    let mut trade_adverse = 0.0_f64;

    let mut equity = 100000.0_f64;
    let mut peak_equity = equity;

    let mut holds_hours: Vec<f64> = Vec::new();
    let mut trade_pnl: Vec<f64> = Vec::new();
    let mut trade_pnl_after_swap: Vec<f64> = Vec::new();
    let mut trade_dd_pct: Vec<f64> = Vec::new();
    let mut eq_dd_after_trade_pct: Vec<f64> = Vec::new();
    let mut swap_costs: Vec<f64> = Vec::new();

    let mut monthly: HashMap<String, _MonthlyAcc> = HashMap::new();
    let mut daily_counts: HashMap<i64, usize> = HashMap::new();

    for i in 1..n {
        if in_position != 0 {
            let current_low = low_vec[i];
            let current_high = high_vec[i];
            let adverse = if in_position == 1 {
                ((entry_price - current_low) / entry_price.max(1e-12)).max(0.0)
            } else {
                ((current_high - entry_price) / entry_price.max(1e-12)).max(0.0)
            };
            if adverse > trade_adverse {
                trade_adverse = adverse;
            }

            let mut pnl = 0.0_f64;
            let mut exit_signal = false;
            if in_position == 1 {
                let mut sl_price = entry_price - (sl * pv);
                let tp_price = entry_price + (tp * pv);
                if trailing_enabled {
                    let mv = current_high - entry_price;
                    if mv >= (trailing_be_trigger_r * sl * pv) {
                        let trail_dist = trailing_atr_multiplier * sl * pv;
                        let candidate = current_high - trail_dist;
                        if trail_price == 0.0 || candidate > trail_price {
                            trail_price = candidate;
                        }
                        if trail_price > sl_price {
                            sl_price = trail_price;
                        }
                    }
                }
                if current_low <= sl_price {
                    pnl = (sl_price - entry_price) / pv * cash_per_pip;
                    exit_signal = true;
                } else if current_high >= tp_price {
                    pnl = (tp_price - entry_price) / pv * cash_per_pip;
                    exit_signal = true;
                }
            } else {
                let mut sl_price = entry_price + (sl * pv);
                let tp_price = entry_price - (tp * pv);
                if trailing_enabled {
                    let mv = entry_price - current_low;
                    if mv >= (trailing_be_trigger_r * sl * pv) {
                        let trail_dist = trailing_atr_multiplier * sl * pv;
                        let candidate = current_low + trail_dist;
                        if trail_price == 0.0 || candidate < trail_price {
                            trail_price = candidate;
                        }
                        if trail_price < sl_price {
                            sl_price = trail_price;
                        }
                    }
                }
                if current_high >= sl_price {
                    pnl = (entry_price - sl_price) / pv * cash_per_pip;
                    exit_signal = true;
                } else if current_low <= tp_price {
                    pnl = (entry_price - tp_price) / pv * cash_per_pip;
                    exit_signal = true;
                }
            }

            if !exit_signal && max_hold_bars > 0 && entry_i >= 0 {
                if (i as i64 - entry_i) as usize >= max_hold_bars {
                    if in_position == 1 {
                        pnl = (close_vec[i] - entry_price) / pv * cash_per_pip;
                    } else {
                        pnl = (entry_price - close_vec[i]) / pv * cash_per_pip;
                    }
                    exit_signal = true;
                }
            }

            if !exit_signal {
                let s = sig_vec[i - 1];
                if in_position == 1 && s == -1 {
                    pnl = (close_vec[i] - entry_price) / pv * cash_per_pip;
                    exit_signal = true;
                } else if in_position == -1 && s == 1 {
                    pnl = (entry_price - close_vec[i]) / pv * cash_per_pip;
                    exit_signal = true;
                }
            }

            if exit_signal {
                let mut hold_h = if entry_i >= 0 {
                    (i as f64 - entry_i as f64) * bar_h
                } else {
                    0.0
                };
                if let Some(ref ts) = ts_vec {
                    if entry_i >= 0 {
                        let ei = entry_i as usize;
                        if i < ts.len() && ei < ts.len() {
                            hold_h = ((ts[i] - ts[ei]) as f64 / 3_600_000.0).max(0.0);
                        }
                    }
                }
                let swap_rate = if in_position == 1 {
                    swap_long_per_day
                } else {
                    swap_short_per_day
                };
                let swap_cost = (hold_h / 24.0).max(0.0) * swap_rate.max(0.0);
                let pnl_net = pnl - commission_per_trade - swap_cost;

                equity += pnl_net;
                if equity > peak_equity {
                    peak_equity = equity;
                }
                let eq_dd = (peak_equity - equity) / peak_equity.max(1e-9);

                holds_hours.push(hold_h);
                trade_pnl.push(pnl - commission_per_trade);
                trade_pnl_after_swap.push(pnl_net);
                trade_dd_pct.push(trade_adverse);
                eq_dd_after_trade_pct.push(eq_dd);
                swap_costs.push(swap_cost);

                let day_key = day_vec[i];
                *daily_counts.entry(day_key).or_insert(0) += 1;
                let month_key = _month_key_from_code(month_vec[i]);
                let m = monthly.entry(month_key).or_default();
                m.trades += 1.0;
                if pnl_net > 0.0 {
                    m.wins += 1.0;
                } else {
                    m.losses += 1.0;
                }
                m.net_profit += pnl_net;
                m.swap_total += swap_cost;
                m.hold_hours_total += hold_h;
                m.trade_dd_pct_total += trade_adverse;

                in_position = 0;
                entry_i = -1;
                trail_price = 0.0;
                trade_adverse = 0.0;
            }
        }

        if in_position == 0 {
            let s = sig_vec[i - 1];
            if s == 1 {
                in_position = 1;
                entry_price = close_vec[i] + (spread_pips * pv);
                entry_i = i as i64;
                trail_price = 0.0;
                trade_adverse = 0.0;
            } else if s == -1 {
                in_position = -1;
                entry_price = close_vec[i] - (spread_pips * pv);
                entry_i = i as i64;
                trail_price = 0.0;
                trade_adverse = 0.0;
            }
        }
    }

    let trades = trade_pnl_after_swap.len();
    let wins = trade_pnl_after_swap.iter().filter(|v| **v > 0.0).count();
    let losses = trade_pnl_after_swap.iter().filter(|v| **v <= 0.0).count();
    let net_after_swap = _sum(&trade_pnl_after_swap);
    let net_no_swap = _sum(&trade_pnl);
    let avg_hold = _mean(&holds_hours);
    let median_hold = _quantile(holds_hours.clone(), 0.5);
    let p90_hold = _quantile(holds_hours.clone(), 0.9);
    let max_hold = _max(&holds_hours);
    let avg_dd_trade = _mean(&trade_dd_pct);
    let max_dd_trade = _max(&trade_dd_pct);
    let avg_eq_dd_trade = _mean(&eq_dd_after_trade_pct);
    let max_eq_dd_trade = _max(&eq_dd_after_trade_pct);

    let mut monthly_keys: Vec<String> = monthly.keys().cloned().collect();
    monthly_keys.sort();
    let mut monthly_out = serde_json::Map::new();
    for key in monthly_keys {
        if let Some(m) = monthly.get(&key) {
            let tr = m.trades;
            let wins_m = m.wins;
            let net_m = m.net_profit;
            monthly_out.insert(
                key,
                serde_json::json!({
                    "trades": tr,
                    "wins": wins_m,
                    "losses": m.losses,
                    "win_rate": if tr > 0.0 { wins_m / tr } else { 0.0 },
                    "net_profit": net_m,
                    "swap_total": m.swap_total,
                    "profit_per_trade": if tr > 0.0 { net_m / tr } else { 0.0 },
                    "avg_holding_hours": if tr > 0.0 { m.hold_hours_total / tr } else { 0.0 },
                    "avg_trade_dd_pct": if tr > 0.0 { m.trade_dd_pct_total / tr } else { 0.0 },
                }),
            );
        }
    }

    let active_days = daily_counts.len();
    let max_trades_day = daily_counts.values().copied().max().unwrap_or(0) as f64;
    let history_days_f = if history_days > 0.0 {
        history_days
    } else {
        (active_days as f64).max(1e-9)
    };
    let history_months_f = if history_months > 0.0 {
        history_months
    } else if history_days_f > 0.0 {
        history_days_f / 30.4375
    } else {
        0.0
    };

    let out = serde_json::json!({
        "computed": true,
        "history_days": history_days_f,
        "history_months": history_months_f,
        "trade_count": trades as f64,
        "wins": wins as f64,
        "losses": losses as f64,
        "win_rate": if trades > 0 { wins as f64 / trades as f64 } else { 0.0 },
        "net_profit": net_after_swap,
        "net_profit_no_swap": net_no_swap,
        "swap_total": _sum(&swap_costs),
        "avg_swap_per_trade": if swap_costs.is_empty() { 0.0 } else { _mean(&swap_costs) },
        "profit_per_trade": if trades > 0 { net_after_swap / trades as f64 } else { 0.0 },
        "avg_holding_hours": avg_hold,
        "median_holding_hours": median_hold,
        "p90_holding_hours": p90_hold,
        "max_holding_hours": max_hold,
        "avg_trades_per_day": if history_days_f > 0.0 { trades as f64 / history_days_f } else { 0.0 },
        "avg_trades_per_month": if history_months_f > 0.0 { trades as f64 / history_months_f } else { 0.0 },
        "avg_trades_active_day": if active_days > 0 { trades as f64 / active_days as f64 } else { 0.0 },
        "max_trades_single_day": max_trades_day,
        "active_days": active_days as f64,
        "avg_trade_dd_pct": avg_dd_trade,
        "max_trade_dd_pct": max_dd_trade,
        "avg_equity_dd_after_trade_pct": avg_eq_dd_trade,
        "max_equity_dd_after_trade_pct": max_eq_dd_trade,
        "monthly": monthly_out,
    });
    let py_obj = pythonize(py, &out)?;
    Ok(py_obj.into())
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
                let sl = if sl_vec.len() == 1 {
                    sl_vec[0]
                } else {
                    sl_vec[row]
                };
                let tp = if tp_vec.len() == 1 {
                    tp_vec[0]
                } else {
                    tp_vec[row]
                };
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
    open,
    high,
    low,
    close,
    indicator_sets,
    weight_sets=None,
    long_thresholds=None,
    short_thresholds=None,
    sl_pips=None,
    tp_pips=None,
    use_ob_flags=None,
    use_fvg_flags=None,
    use_liq_flags=None,
    use_mtf_flags=None,
    use_premium_flags=None,
    use_inducement_flags=None,
    timestamps=None,
    volume=None,
    include_raw=false,
    smc_gate_threshold=0.0,
    smc_weight_ob=1.0,
    smc_weight_fvg=1.0,
    smc_weight_liq=1.0,
    smc_weight_mtf=1.0,
    smc_weight_premium=1.0,
    smc_weight_inducement=1.0,
    max_hold_bars=0,
    trailing_enabled=false,
    trailing_atr_multiplier=1.0,
    trailing_be_trigger_r=1.0,
    pip_value=0.0001,
    spread_pips=1.5,
    commission_per_trade=0.0,
    pip_value_per_lot=10.0,
    causal_min_bars=30,
    use_bos_flags=None,
    use_choch_flags=None,
    use_eqh_flags=None,
    use_eql_flags=None,
    use_displacement_flags=None,
    smc_weight_bos=1.0,
    smc_weight_choch=1.0,
    smc_weight_eqh=1.0,
    smc_weight_eql=1.0,
    smc_weight_displacement=1.0
))]
fn evaluate_population_talib_ohlcv<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    indicator_sets: Vec<Vec<String>>,
    weight_sets: Option<Vec<Vec<f64>>>,
    long_thresholds: Option<Vec<f64>>,
    short_thresholds: Option<Vec<f64>>,
    sl_pips: Option<Vec<f64>>,
    tp_pips: Option<Vec<f64>>,
    use_ob_flags: Option<Vec<i8>>,
    use_fvg_flags: Option<Vec<i8>>,
    use_liq_flags: Option<Vec<i8>>,
    use_mtf_flags: Option<Vec<i8>>,
    use_premium_flags: Option<Vec<i8>>,
    use_inducement_flags: Option<Vec<i8>>,
    timestamps: Option<PyReadonlyArray1<'py, i64>>,
    volume: Option<PyReadonlyArray1<'py, f64>>,
    include_raw: bool,
    smc_gate_threshold: f32,
    smc_weight_ob: f32,
    smc_weight_fvg: f32,
    smc_weight_liq: f32,
    smc_weight_mtf: f32,
    smc_weight_premium: f32,
    smc_weight_inducement: f32,
    max_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
    causal_min_bars: usize,
    use_bos_flags: Option<Vec<i8>>,
    use_choch_flags: Option<Vec<i8>>,
    use_eqh_flags: Option<Vec<i8>>,
    use_eql_flags: Option<Vec<i8>>,
    use_displacement_flags: Option<Vec<i8>>,
    smc_weight_bos: f32,
    smc_weight_choch: f32,
    smc_weight_eqh: f32,
    smc_weight_eql: f32,
    smc_weight_displacement: f32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n_genes = indicator_sets.len();
    if n_genes == 0 {
        return Ok(Array2::<f64>::zeros((0, 11)).into_pyarray(py));
    }
    if let Some(ref w) = weight_sets {
        if w.len() != n_genes {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "weight_sets length must match indicator_sets length",
            ));
        }
    }
    if let Some(ref v) = long_thresholds {
        if v.len() != n_genes {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "long_thresholds length must match indicator_sets length",
            ));
        }
    }
    if let Some(ref v) = short_thresholds {
        if v.len() != n_genes {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "short_thresholds length must match indicator_sets length",
            ));
        }
    }
    if let Some(ref v) = sl_pips {
        if !(v.len() == n_genes || v.len() == 1) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "sl_pips length must be 1 or indicator_sets length",
            ));
        }
    }
    if let Some(ref v) = tp_pips {
        if !(v.len() == n_genes || v.len() == 1) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "tp_pips length must be 1 or indicator_sets length",
            ));
        }
    }

    let ohlcv = build_ohlcv(
        &open,
        &high,
        &low,
        &close,
        timestamps.as_ref(),
        volume.as_ref(),
    )
    .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;

    let close_vec = vec_from_py_f64(&close);
    let high_vec = vec_from_py_f64(&high);
    let low_vec = vec_from_py_f64(&low);
    let open_vec = vec_from_py_f64(&open);
    let _vol_vec = volume.as_ref().map(vec_from_py_f64);

    let frame = compute_talib_feature_frame(&ohlcv, include_raw).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Feature computation failed: {}",
            e
        ))
    })?;

    let n_samples = frame.data.nrows();
    if n_samples == 0 {
        return Ok(Array2::<f64>::zeros((n_genes, 11)).into_pyarray(py));
    }

    // Map indicator name to column index.
    let mut idx_map: HashMap<String, usize> = HashMap::with_capacity(frame.names.len() * 2);
    for (idx, name) in frame.names.iter().enumerate() {
        let norm = normalize_indicator_name(name.strip_prefix("ta_").unwrap_or(name));
        if !norm.is_empty() {
            idx_map.entry(norm).or_insert(idx);
        }
    }

    // Build indicator matrix (rows = indicators) with causal tanh z-score per column.
    // Collect unique indicators first.
    let mut needed: HashSet<String> = HashSet::new();
    for inds in indicator_sets.iter() {
        for ind in inds {
            let norm = normalize_indicator_name(ind);
            if !norm.is_empty() {
                needed.insert(norm);
            }
        }
    }
    if needed.is_empty() {
        return Ok(Array2::<f64>::zeros((n_genes, 11)).into_pyarray(py));
    }
    let mut indicator_list: Vec<String> = needed.into_iter().collect();
    indicator_list.sort();
    let mut ind_to_row: HashMap<String, usize> = HashMap::with_capacity(indicator_list.len());
    for (i, name) in indicator_list.iter().enumerate() {
        ind_to_row.insert(name.clone(), i);
    }

    let mut indicators = Array2::<f32>::zeros((indicator_list.len(), n_samples));
    for (row_idx, name) in indicator_list.iter().enumerate() {
        let col_idx = idx_map
            .get(name)
            .copied()
            .or_else(|| map_indicator_index(name, &frame.names));
        if let Some(ci) = col_idx {
            let score = causal_tanh_zscore_column(&frame.data, ci, causal_min_bars);
            for r in 0..n_samples {
                indicators[(row_idx, r)] = score[r] as f32;
            }
        }
    }

    // Build gene arrays
    let mut offsets: Vec<i32> = Vec::with_capacity(n_genes + 1);
    let mut indices: Vec<i32> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    let mut long_thr_vec: Vec<f32> = Vec::with_capacity(n_genes);
    let mut short_thr_vec: Vec<f32> = Vec::with_capacity(n_genes);
    let mut sl_vec: Vec<f64> = Vec::with_capacity(n_genes);
    let mut tp_vec: Vec<f64> = Vec::with_capacity(n_genes);
    let mut use_ob_vec: Vec<i8> = Vec::with_capacity(n_genes);
    let mut use_fvg_vec: Vec<i8> = Vec::with_capacity(n_genes);
    let mut use_liq_vec: Vec<i8> = Vec::with_capacity(n_genes);
    let mut use_mtf_vec: Vec<i8> = Vec::with_capacity(n_genes);
    let mut use_premium_vec: Vec<i8> = Vec::with_capacity(n_genes);
    let mut use_inducement_vec: Vec<i8> = Vec::with_capacity(n_genes);
    let mut use_bos_vec: Vec<i8> = Vec::with_capacity(n_genes);
    let mut use_choch_vec: Vec<i8> = Vec::with_capacity(n_genes);
    let mut use_eqh_vec: Vec<i8> = Vec::with_capacity(n_genes);
    let mut use_eql_vec: Vec<i8> = Vec::with_capacity(n_genes);
    let mut use_displacement_vec: Vec<i8> = Vec::with_capacity(n_genes);
    offsets.push(0);

    for g in 0..n_genes {
        let inds = &indicator_sets[g];
        let ws = weight_sets.as_ref().and_then(|w| w.get(g));
        for (k, ind) in inds.iter().enumerate() {
            let norm = normalize_indicator_name(ind);
            if norm.is_empty() {
                continue;
            }
            if let Some(row_idx) = ind_to_row.get(&norm) {
                indices.push(*row_idx as i32);
                let w = ws.and_then(|arr| arr.get(k)).copied().unwrap_or(1.0);
                weights.push(w as f32);
            }
        }
        offsets.push(indices.len() as i32);
        let lt = long_thresholds
            .as_ref()
            .and_then(|v| v.get(g))
            .copied()
            .unwrap_or(0.66);
        let st = short_thresholds
            .as_ref()
            .and_then(|v| v.get(g))
            .copied()
            .unwrap_or(-0.66);
        long_thr_vec.push(lt as f32);
        short_thr_vec.push(st as f32);

        let sl = sl_pips
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0.0);
        let tp = tp_pips
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0.0);
        sl_vec.push(sl);
        tp_vec.push(tp);

        let use_ob_val = use_ob_flags
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0);
        let use_fvg_val = use_fvg_flags
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0);
        let use_liq_val = use_liq_flags
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0);
        let use_mtf_val = use_mtf_flags
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0);
        let use_premium_val = use_premium_flags
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0);
        let use_inducement_val = use_inducement_flags
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0);
        let use_bos_val = use_bos_flags
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0);
        let use_choch_val = use_choch_flags
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0);
        let use_eqh_val = use_eqh_flags
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0);
        let use_eql_val = use_eql_flags
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0);
        let use_displacement_val = use_displacement_flags
            .as_ref()
            .map(|v| if v.len() == 1 { v[0] } else { v[g] })
            .unwrap_or(0);

        use_ob_vec.push(use_ob_val);
        use_fvg_vec.push(use_fvg_val);
        use_liq_vec.push(use_liq_val);
        use_mtf_vec.push(use_mtf_val);
        use_premium_vec.push(use_premium_val);
        use_inducement_vec.push(use_inducement_val);
        use_bos_vec.push(use_bos_val);
        use_choch_vec.push(use_choch_val);
        use_eqh_vec.push(use_eqh_val);
        use_eql_vec.push(use_eql_val);
        use_displacement_vec.push(use_displacement_val);
    }

    // Infer SL/TP if missing (<=0).
    let needs_sl_tp = sl_vec
        .iter()
        .zip(tp_vec.iter())
        .any(|(sl, tp)| *sl <= 0.0 || *tp <= 0.0);
    if needs_sl_tp {
        let default = infer_stop_target_pips_rs(
            &open_vec,
            &high_vec,
            &low_vec,
            &close_vec,
            &StopTargetSettings::default(),
            pip_value,
            0,
        )
        .unwrap_or((20.0, 40.0, 2.0));
        for (sl, tp) in sl_vec.iter_mut().zip(tp_vec.iter_mut()) {
            if !sl.is_finite() || *sl <= 0.0 {
                *sl = default.0;
            }
            if !tp.is_finite() || *tp <= 0.0 {
                *tp = default.1;
            }
        }
    }

    // Month/day indices
    let (month_vec, day_vec) = if let Some(ts) = &ohlcv.timestamp {
        let mut months = Vec::with_capacity(ts.len());
        let mut days = Vec::with_capacity(ts.len());
        for t in ts {
            let dt = Utc.timestamp_millis_opt(*t).single();
            if let Some(dt) = dt {
                months.push((dt.year() as i64) * 12 + dt.month() as i64);
                days.push((dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64);
            } else {
                months.push(0);
                days.push(0);
            }
        }
        (months, days)
    } else {
        let n = close.as_array().len();
        let mut seq: Vec<i64> = Vec::with_capacity(n);
        for i in 0..n {
            seq.push(i as i64);
        }
        (seq.clone(), seq)
    };

    // SMC arrays derived from OHLC, with feature-column overrides when present.
    let (
        ob_vec_full,
        fvg_vec_full,
        liq_vec_full,
        trend_vec_full,
        premium_vec_full,
        inducement_vec_full,
        bos_vec_full,
        choch_vec_full,
        eqh_vec_full,
        eql_vec_full,
        displacement_vec_full,
    ) = build_smc_arrays(&frame, &ohlcv);
    let ob_arr = ob_vec_full;
    let fvg_arr = fvg_vec_full;
    let liq_arr = liq_vec_full;
    let trend_arr = trend_vec_full;
    let premium_arr = premium_vec_full;
    let inducement_arr = inducement_vec_full;
    let bos_arr = bos_vec_full;
    let choch_arr = choch_vec_full;
    let eqh_arr = eqh_vec_full;
    let eql_arr = eql_vec_full;
    let displacement_arr = displacement_vec_full;

    let metrics = py.detach(|| {
        evaluate_population_core(
            &close_vec,
            &high_vec,
            &low_vec,
            indicators.view(),
            &offsets,
            &indices,
            &weights,
            &long_thr_vec,
            &short_thr_vec,
            &month_vec,
            &day_vec,
            &sl_vec,
            &tp_vec,
            &ob_arr,
            &fvg_arr,
            &liq_arr,
            &trend_arr,
            &premium_arr,
            &inducement_arr,
            &bos_arr,
            &choch_arr,
            &eqh_arr,
            &eql_arr,
            &displacement_arr,
            &use_ob_vec,
            &use_fvg_vec,
            &use_liq_vec,
            &use_mtf_vec,
            &use_premium_vec,
            &use_inducement_vec,
            &use_bos_vec,
            &use_choch_vec,
            &use_eqh_vec,
            &use_eql_vec,
            &use_displacement_vec,
            smc_gate_threshold,
            smc_weight_ob,
            smc_weight_fvg,
            smc_weight_liq,
            smc_weight_mtf,
            smc_weight_premium,
            smc_weight_inducement,
            smc_weight_bos,
            smc_weight_choch,
            smc_weight_eqh,
            smc_weight_eql,
            smc_weight_displacement,
            max_hold_bars,
            trailing_enabled,
            trailing_atr_multiplier,
            trailing_be_trigger_r,
            pip_value,
            spread_pips,
            commission_per_trade,
            pip_value_per_lot,
        )
        .map(|rows| {
            let mut out = Array2::<f64>::zeros((rows.len(), 11));
            for (r, row) in rows.iter().enumerate() {
                for c in 0..11 {
                    out[(r, c)] = row[c];
                }
            }
            out
        })
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    })?;

    Ok(metrics.into_pyarray(py))
}

#[derive(Debug, Clone, Copy, Default)]
struct SmcColumns {
    ob: Option<usize>,
    fvg: Option<usize>,
    liq: Option<usize>,
    trend: Option<usize>,
    premium: Option<usize>,
    inducement: Option<usize>,
    bos: Option<usize>,
    choch: Option<usize>,
    eqh: Option<usize>,
    eql: Option<usize>,
    displacement: Option<usize>,
}

fn normalize_feature_name(name: &str) -> String {
    name.to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "_")
}

fn find_feature_column(names: &[String], aliases: &[&str]) -> Option<usize> {
    let normalized_aliases: Vec<String> =
        aliases.iter().map(|a| normalize_feature_name(a)).collect();
    for (idx, raw) in names.iter().enumerate() {
        let norm = normalize_feature_name(raw);
        if normalized_aliases
            .iter()
            .any(|a| norm == *a || norm.contains(a))
        {
            return Some(idx);
        }
    }
    None
}

fn detect_smc_columns(names: &[String]) -> SmcColumns {
    SmcColumns {
        ob: find_feature_column(names, &["smc_ob", "order_block", "ob"]),
        fvg: find_feature_column(names, &["smc_fvg", "fair_value_gap", "fvg"]),
        liq: find_feature_column(names, &["smc_liq", "liquidity_sweep", "liq_sweep", "liq"]),
        trend: find_feature_column(names, &["smc_trend", "trend", "market_trend"]),
        premium: find_feature_column(names, &["smc_premium", "premium_discount", "premium"]),
        inducement: find_feature_column(names, &["smc_inducement", "inducement"]),
        bos: find_feature_column(names, &["smc_bos", "bos", "break_of_structure"]),
        choch: find_feature_column(names, &["smc_choch", "choch", "change_of_character"]),
        eqh: find_feature_column(names, &["smc_eqh", "eqh", "equal_highs"]),
        eql: find_feature_column(names, &["smc_eql", "eql", "equal_lows"]),
        displacement: find_feature_column(
            names,
            &["smc_displacement", "displacement", "impulse_displacement"],
        ),
    }
}

fn quantize_dir(value: f32) -> i8 {
    if value > 1e-9 {
        1
    } else if value < -1e-9 {
        -1
    } else {
        0
    }
}

fn quantize_binary(value: f32) -> i8 {
    if value > 1e-9 {
        1
    } else {
        0
    }
}

fn derive_smc_arrays(
    ohlcv: &Ohlcv,
) -> (
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
) {
    let n = ohlcv.close.len();
    let mut ob = vec![0_i8; n];
    let mut fvg = vec![0_i8; n];
    let mut liq = vec![0_i8; n];
    let mut trend = vec![0_i8; n];
    let mut premium = vec![0_i8; n];
    let mut inducement = vec![0_i8; n];
    let mut bos = vec![0_i8; n];
    let mut choch = vec![0_i8; n];
    let mut eqh = vec![0_i8; n];
    let mut eql = vec![0_i8; n];
    let mut displacement = vec![0_i8; n];
    if n == 0 {
        return (
            ob,
            fvg,
            liq,
            trend,
            premium,
            inducement,
            bos,
            choch,
            eqh,
            eql,
            displacement,
        );
    }

    let lookback = 12usize;
    let eq_lookback = 20usize;
    let displacement_lookback = 20usize;
    for i in 0..n {
        if i >= lookback {
            let d = ohlcv.close[i] - ohlcv.close[i - lookback];
            trend[i] = if d > 0.0 {
                1
            } else if d < 0.0 {
                -1
            } else {
                0
            };
        } else if i > 0 {
            let d = ohlcv.close[i] - ohlcv.close[i - 1];
            trend[i] = if d > 0.0 {
                1
            } else if d < 0.0 {
                -1
            } else {
                0
            };
        }

        let mid = (ohlcv.high[i] + ohlcv.low[i]) * 0.5;
        premium[i] = if ohlcv.close[i] <= mid { 1 } else { -1 };

        if i >= 1 {
            let bull = ohlcv.close[i] > ohlcv.open[i]
                && ohlcv.close[i - 1] < ohlcv.open[i - 1]
                && ohlcv.close[i] >= ohlcv.high[i - 1];
            let bear = ohlcv.close[i] < ohlcv.open[i]
                && ohlcv.close[i - 1] > ohlcv.open[i - 1]
                && ohlcv.close[i] <= ohlcv.low[i - 1];
            ob[i] = if bull {
                1
            } else if bear {
                -1
            } else {
                0
            };

            let body = (ohlcv.close[i] - ohlcv.open[i]).abs();
            let upper = ohlcv.high[i] - ohlcv.open[i].max(ohlcv.close[i]);
            let lower = ohlcv.open[i].min(ohlcv.close[i]) - ohlcv.low[i];
            if body > 1e-12 && ((upper / body) > 2.0 || (lower / body) > 2.0) {
                inducement[i] = 1;
            }
        }

        if i >= 2 {
            if ohlcv.low[i] > ohlcv.high[i - 2] {
                fvg[i] = 1;
            } else if ohlcv.high[i] < ohlcv.low[i - 2] {
                fvg[i] = -1;
            }
        }

        if i >= 3 {
            let prev_low = ohlcv.low[(i - 3)..i]
                .iter()
                .fold(f64::INFINITY, |a, b| a.min(*b));
            let prev_high = ohlcv.high[(i - 3)..i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            if ohlcv.low[i] < prev_low && ohlcv.close[i] > prev_low {
                liq[i] = 1;
            } else if ohlcv.high[i] > prev_high && ohlcv.close[i] < prev_high {
                liq[i] = -1;
            }
        }

        if i >= lookback {
            let prev_low = ohlcv.low[(i - lookback)..i]
                .iter()
                .fold(f64::INFINITY, |a, b| a.min(*b));
            let prev_high = ohlcv.high[(i - lookback)..i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            if ohlcv.close[i] > prev_high {
                bos[i] = 1;
            } else if ohlcv.close[i] < prev_low {
                bos[i] = -1;
            }
        }

        if i >= 1 && trend[i] != 0 && trend[i - 1] != 0 && trend[i] != trend[i - 1] {
            choch[i] = trend[i];
        }

        if i >= eq_lookback {
            let lb = i - eq_lookback;
            let mut range_sum = 0.0;
            for j in lb..=i {
                range_sum += (ohlcv.high[j] - ohlcv.low[j]).abs();
            }
            let avg_range = range_sum / ((eq_lookback as f64) + 1.0);
            let tol = (avg_range * 0.1).max(1e-9);
            for j in lb..i {
                if (ohlcv.high[i] - ohlcv.high[j]).abs() <= tol {
                    eqh[i] = -1;
                    break;
                }
            }
            for j in lb..i {
                if (ohlcv.low[i] - ohlcv.low[j]).abs() <= tol {
                    eql[i] = 1;
                    break;
                }
            }
        }

        if i >= displacement_lookback {
            let body = (ohlcv.close[i] - ohlcv.open[i]).abs();
            let mut avg_body = 0.0;
            for j in (i - displacement_lookback)..i {
                avg_body += (ohlcv.close[j] - ohlcv.open[j]).abs();
            }
            avg_body /= displacement_lookback as f64;
            if avg_body > 1e-12 && body >= (1.8 * avg_body) {
                displacement[i] = if ohlcv.close[i] > ohlcv.open[i] {
                    1
                } else if ohlcv.close[i] < ohlcv.open[i] {
                    -1
                } else {
                    0
                };
            }
        }
    }

    (
        ob,
        fvg,
        liq,
        trend,
        premium,
        inducement,
        bos,
        choch,
        eqh,
        eql,
        displacement,
    )
}

fn build_smc_arrays(
    frame: &forex_data::FeatureFrame,
    ohlcv: &Ohlcv,
) -> (
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
) {
    let n = frame.data.nrows();
    let cols = detect_smc_columns(&frame.names);
    let (
        mut ob,
        mut fvg,
        mut liq,
        mut trend,
        mut premium,
        mut inducement,
        mut bos,
        mut choch,
        mut eqh,
        mut eql,
        mut displacement,
    ) = derive_smc_arrays(ohlcv);

    let apply_dir_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for i in 0..n {
                    target[i] = quantize_dir(frame.data[(i, col)]);
                }
            }
        }
    };
    let apply_binary_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for i in 0..n {
                    target[i] = quantize_binary(frame.data[(i, col)]);
                }
            }
        }
    };
    let apply_eqh_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for i in 0..n {
                    let v = frame.data[(i, col)];
                    let q = quantize_dir(v);
                    if q != 0 {
                        target[i] = q;
                    } else if quantize_binary(v) != 0 {
                        target[i] = -1;
                    } else {
                        target[i] = 0;
                    }
                }
            }
        }
    };
    let apply_eql_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for i in 0..n {
                    let v = frame.data[(i, col)];
                    let q = quantize_dir(v);
                    if q != 0 {
                        target[i] = q;
                    } else if quantize_binary(v) != 0 {
                        target[i] = 1;
                    } else {
                        target[i] = 0;
                    }
                }
            }
        }
    };
    let apply_dir_fill_zeros = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for i in 0..n {
                    if target[i] == 0 {
                        target[i] = quantize_dir(frame.data[(i, col)]);
                    }
                }
            }
        }
    };
    let apply_eq_levels = |target: &mut Vec<i8>, eqh_col: Option<usize>, eql_col: Option<usize>| {
        if let Some(col) = eqh_col {
            if col < frame.data.ncols() {
                for i in 0..n {
                    if quantize_binary(frame.data[(i, col)]) != 0 {
                        target[i] = -1;
                    }
                }
            }
        }
        if let Some(col) = eql_col {
            if col < frame.data.ncols() {
                for i in 0..n {
                    if quantize_binary(frame.data[(i, col)]) != 0 {
                        target[i] = 1;
                    }
                }
            }
        }
    };

    apply_dir_col(&mut ob, cols.ob);
    apply_dir_col(&mut fvg, cols.fvg);
    apply_dir_col(&mut liq, cols.liq);
    apply_dir_col(&mut trend, cols.trend);
    apply_dir_col(&mut premium, cols.premium);
    apply_binary_col(&mut inducement, cols.inducement);
    apply_dir_col(&mut bos, cols.bos);
    apply_dir_col(&mut choch, cols.choch);
    apply_eqh_col(&mut eqh, cols.eqh);
    apply_eql_col(&mut eql, cols.eql);
    apply_dir_col(&mut displacement, cols.displacement);
    apply_dir_fill_zeros(&mut ob, cols.bos);
    apply_dir_fill_zeros(&mut ob, cols.choch);
    apply_eq_levels(&mut liq, cols.eqh, cols.eql);
    apply_dir_fill_zeros(&mut trend, cols.bos);
    apply_dir_fill_zeros(&mut trend, cols.choch);
    apply_dir_fill_zeros(&mut trend, cols.displacement);
    if let Some(col) = cols.displacement {
        if col < frame.data.ncols() {
            for i in 0..n {
                if quantize_dir(frame.data[(i, col)]) != 0 {
                    inducement[i] = 1;
                }
            }
        }
    }
    for i in 0..n {
        if displacement[i] != 0 {
            inducement[i] = 1;
        }
    }

    (
        ob,
        fvg,
        liq,
        trend,
        premium,
        inducement,
        bos,
        choch,
        eqh,
        eql,
        displacement,
    )
}

#[pyfunction(name = "evaluate_population_core")]
#[pyo3(signature = (
    close_prices,
    high_prices,
    low_prices,
    indicators,
    gene_offsets,
    gene_indices,
    gene_weights,
    long_thr,
    short_thr,
    month_indices,
    day_indices,
    sl_pips,
    tp_pips,
    ob_arr,
    fvg_arr,
    liq_arr,
    trend_arr,
    premium_arr,
    inducement_arr,
    use_ob_arr,
    use_fvg_arr,
    use_liq_arr,
    use_mtf_arr,
    use_premium_arr,
    use_inducement_arr,
    smc_gate_threshold=0.0,
    smc_weight_ob=1.0,
    smc_weight_fvg=1.0,
    smc_weight_liq=1.0,
    smc_weight_mtf=1.0,
    smc_weight_premium=1.0,
    smc_weight_inducement=1.0,
    max_hold_bars=0,
    trailing_enabled=false,
    trailing_atr_multiplier=1.0,
    trailing_be_trigger_r=1.0,
    pip_value=0.0001,
    spread_pips=1.5,
    commission_per_trade=0.0,
    pip_value_per_lot=10.0,
    bos_arr=None,
    choch_arr=None,
    eqh_arr=None,
    eql_arr=None,
    displacement_arr=None,
    use_bos_arr=None,
    use_choch_arr=None,
    use_eqh_arr=None,
    use_eql_arr=None,
    use_displacement_arr=None,
    smc_weight_bos=1.0,
    smc_weight_choch=1.0,
    smc_weight_eqh=1.0,
    smc_weight_eql=1.0,
    smc_weight_displacement=1.0
))]
fn evaluate_population_core_py<'py>(
    py: Python<'py>,
    close_prices: PyReadonlyArray1<'py, f64>,
    high_prices: PyReadonlyArray1<'py, f64>,
    low_prices: PyReadonlyArray1<'py, f64>,
    indicators: PyReadonlyArray2<'py, f32>,
    gene_offsets: PyReadonlyArray1<'py, i32>,
    gene_indices: PyReadonlyArray1<'py, i32>,
    gene_weights: PyReadonlyArray1<'py, f32>,
    long_thr: PyReadonlyArray1<'py, f32>,
    short_thr: PyReadonlyArray1<'py, f32>,
    month_indices: PyReadonlyArray1<'py, i64>,
    day_indices: PyReadonlyArray1<'py, i64>,
    sl_pips: PyReadonlyArray1<'py, f64>,
    tp_pips: PyReadonlyArray1<'py, f64>,
    ob_arr: PyReadonlyArray1<'py, i8>,
    fvg_arr: PyReadonlyArray1<'py, i8>,
    liq_arr: PyReadonlyArray1<'py, i8>,
    trend_arr: PyReadonlyArray1<'py, i8>,
    premium_arr: PyReadonlyArray1<'py, i8>,
    inducement_arr: PyReadonlyArray1<'py, i8>,
    use_ob_arr: PyReadonlyArray1<'py, i8>,
    use_fvg_arr: PyReadonlyArray1<'py, i8>,
    use_liq_arr: PyReadonlyArray1<'py, i8>,
    use_mtf_arr: PyReadonlyArray1<'py, i8>,
    use_premium_arr: PyReadonlyArray1<'py, i8>,
    use_inducement_arr: PyReadonlyArray1<'py, i8>,
    smc_gate_threshold: f32,
    smc_weight_ob: f32,
    smc_weight_fvg: f32,
    smc_weight_liq: f32,
    smc_weight_mtf: f32,
    smc_weight_premium: f32,
    smc_weight_inducement: f32,
    max_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
    bos_arr: Option<PyReadonlyArray1<'py, i8>>,
    choch_arr: Option<PyReadonlyArray1<'py, i8>>,
    eqh_arr: Option<PyReadonlyArray1<'py, i8>>,
    eql_arr: Option<PyReadonlyArray1<'py, i8>>,
    displacement_arr: Option<PyReadonlyArray1<'py, i8>>,
    use_bos_arr: Option<PyReadonlyArray1<'py, i8>>,
    use_choch_arr: Option<PyReadonlyArray1<'py, i8>>,
    use_eqh_arr: Option<PyReadonlyArray1<'py, i8>>,
    use_eql_arr: Option<PyReadonlyArray1<'py, i8>>,
    use_displacement_arr: Option<PyReadonlyArray1<'py, i8>>,
    smc_weight_bos: f32,
    smc_weight_choch: f32,
    smc_weight_eqh: f32,
    smc_weight_eql: f32,
    smc_weight_displacement: f32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_vec = vec_from_py_f64(&close_prices);
    let high_vec = vec_from_py_f64(&high_prices);
    let low_vec = vec_from_py_f64(&low_prices);
    let month_vec = vec_from_py_i64(&month_indices);
    let day_vec = vec_from_py_i64(&day_indices);
    let sl_vec = vec_from_py_f64(&sl_pips);
    let tp_vec = vec_from_py_f64(&tp_pips);
    let offsets = vec_from_py_i32(&gene_offsets);
    let indices = vec_from_py_i32(&gene_indices);
    let weights = vec_from_py_f32(&gene_weights);
    let long_vec = vec_from_py_f32(&long_thr);
    let short_vec = vec_from_py_f32(&short_thr);
    let ob_vec = vec_from_py_i8(&ob_arr);
    let fvg_vec = vec_from_py_i8(&fvg_arr);
    let liq_vec = vec_from_py_i8(&liq_arr);
    let trend_vec = vec_from_py_i8(&trend_arr);
    let premium_vec = vec_from_py_i8(&premium_arr);
    let inducement_vec = vec_from_py_i8(&inducement_arr);
    let use_ob_vec = vec_from_py_i8(&use_ob_arr);
    let use_fvg_vec = vec_from_py_i8(&use_fvg_arr);
    let use_liq_vec = vec_from_py_i8(&use_liq_arr);
    let use_mtf_vec = vec_from_py_i8(&use_mtf_arr);
    let use_premium_vec = vec_from_py_i8(&use_premium_arr);
    let use_inducement_vec = vec_from_py_i8(&use_inducement_arr);
    let n_samples = close_vec.len();
    let n_genes = long_vec.len();
    let bos_vec = bos_arr
        .as_ref()
        .map(vec_from_py_i8)
        .unwrap_or_else(|| vec![0_i8; n_samples]);
    let choch_vec = choch_arr
        .as_ref()
        .map(vec_from_py_i8)
        .unwrap_or_else(|| vec![0_i8; n_samples]);
    let eqh_vec = eqh_arr
        .as_ref()
        .map(vec_from_py_i8)
        .unwrap_or_else(|| vec![0_i8; n_samples]);
    let eql_vec = eql_arr
        .as_ref()
        .map(vec_from_py_i8)
        .unwrap_or_else(|| vec![0_i8; n_samples]);
    let displacement_vec = displacement_arr
        .as_ref()
        .map(vec_from_py_i8)
        .unwrap_or_else(|| vec![0_i8; n_samples]);
    let use_bos_vec = use_bos_arr
        .as_ref()
        .map(vec_from_py_i8)
        .unwrap_or_else(|| vec![0_i8; n_genes]);
    let use_choch_vec = use_choch_arr
        .as_ref()
        .map(vec_from_py_i8)
        .unwrap_or_else(|| vec![0_i8; n_genes]);
    let use_eqh_vec = use_eqh_arr
        .as_ref()
        .map(vec_from_py_i8)
        .unwrap_or_else(|| vec![0_i8; n_genes]);
    let use_eql_vec = use_eql_arr
        .as_ref()
        .map(vec_from_py_i8)
        .unwrap_or_else(|| vec![0_i8; n_genes]);
    let use_displacement_vec = use_displacement_arr
        .as_ref()
        .map(vec_from_py_i8)
        .unwrap_or_else(|| vec![0_i8; n_genes]);
    let indicators_arr = indicators.as_array().to_owned();

    let result: Result<Array2<f64>, PyErr> = py.detach(|| {
        evaluate_population_core(
            &close_vec,
            &high_vec,
            &low_vec,
            indicators_arr.view(),
            &offsets,
            &indices,
            &weights,
            &long_vec,
            &short_vec,
            &month_vec,
            &day_vec,
            &sl_vec,
            &tp_vec,
            &ob_vec,
            &fvg_vec,
            &liq_vec,
            &trend_vec,
            &premium_vec,
            &inducement_vec,
            &bos_vec,
            &choch_vec,
            &eqh_vec,
            &eql_vec,
            &displacement_vec,
            &use_ob_vec,
            &use_fvg_vec,
            &use_liq_vec,
            &use_mtf_vec,
            &use_premium_vec,
            &use_inducement_vec,
            &use_bos_vec,
            &use_choch_vec,
            &use_eqh_vec,
            &use_eql_vec,
            &use_displacement_vec,
            smc_gate_threshold,
            smc_weight_ob,
            smc_weight_fvg,
            smc_weight_liq,
            smc_weight_mtf,
            smc_weight_premium,
            smc_weight_inducement,
            smc_weight_bos,
            smc_weight_choch,
            smc_weight_eqh,
            smc_weight_eql,
            smc_weight_displacement,
            max_hold_bars,
            trailing_enabled,
            trailing_atr_multiplier,
            trailing_be_trigger_r,
            pip_value,
            spread_pips,
            commission_per_trade,
            pip_value_per_lot,
        )
        .map(|vec_rows| {
            let mut out = Array2::<f64>::zeros((vec_rows.len(), 11));
            for (row_idx, row) in vec_rows.iter().enumerate() {
                for col in 0..11 {
                    out[(row_idx, col)] = row[col];
                }
            }
            out
        })
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    });

    result.map(|arr| arr.into_pyarray(py))
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
#[pyclass(unsendable)]
struct LightGBMModel {
    model: Mutex<LightGBMExpert>,
}

#[cfg(feature = "lightgbm")]
#[pymethods]
impl LightGBMModel {
    #[new]
    #[pyo3(signature = (idx=1, params=None))]
    fn new(idx: usize, params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let params = params_from_py(params)
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;
        Ok(Self {
            model: Mutex::new(LightGBMExpert::new(idx, params)),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i32> = labels_array.iter().copied().collect();
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
        let params = params_from_py(params)
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;
        Ok(Self {
            model: Mutex::new(XGBoostExpert::new(idx, params)),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i32> = labels_array.iter().copied().collect();
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
        let params = params_from_py(params)
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;
        Ok(Self {
            model: Mutex::new(XGBoostRFExpert::new(idx, params)),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i32> = labels_array.iter().copied().collect();
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
        let params = params_from_py(params)
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;
        Ok(Self {
            model: Mutex::new(XGBoostDARTExpert::new(idx, params)),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i32> = labels_array.iter().copied().collect();
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
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i32> = labels_array.iter().copied().collect();
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
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i32> = labels_array.iter().copied().collect();
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

#[pyclass(unsendable)]
struct GeneticModel {
    model: Mutex<GeneticStrategyExpert>,
}

#[pymethods]
impl GeneticModel {
    #[new]
    #[pyo3(signature = (idx=1, population_size=50, generations=10, max_indicators=0))]
    fn new(
        idx: usize,
        population_size: usize,
        generations: usize,
        max_indicators: usize,
    ) -> PyResult<Self> {
        let _ = idx;
        let model = GeneticStrategyExpert::new(population_size, generations, max_indicators)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Genetic init failed: {}",
                    e
                ))
            })?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    #[pyo3(signature = (
        features,
        labels,
        feature_names=None,
        metadata=None,
        metadata_columns=None,
        metadata_symbol=None,
    ))]
    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
        feature_names: Option<Vec<String>>,
        metadata: Option<PyReadonlyArray2<'py, f64>>,
        metadata_columns: Option<Vec<String>>,
        metadata_symbol: Option<String>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_vec: Vec<i32> = labels.as_array().iter().copied().collect();
        let metadata_array = metadata.map(|arr| arr.as_array().to_owned());

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())?;
            let labels_series = Series::new("label".into(), labels_vec);
            let metadata_df = match metadata_array.as_ref() {
                Some(arr) => Some(dataframe_from_named_ndarray(
                    arr,
                    metadata_columns.as_deref(),
                )?),
                None => None,
            };

            model
                .fit(
                    &df,
                    &labels_series,
                    metadata_df.as_ref(),
                    metadata_symbol.as_deref(),
                )
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        })();

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    #[pyo3(signature = (
        features,
        feature_names=None,
        metadata=None,
        metadata_columns=None,
        metadata_symbol=None,
    ))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        feature_names: Option<Vec<String>>,
        metadata: Option<PyReadonlyArray2<'py, f64>>,
        metadata_columns: Option<Vec<String>>,
        metadata_symbol: Option<String>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();
        let metadata_array = metadata.map(|arr| arr.as_array().to_owned());

        let result: Result<Array2<f32>, String> = (|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())?;
            let metadata_df = match metadata_array.as_ref() {
                Some(arr) => Some(dataframe_from_named_ndarray(
                    arr,
                    metadata_columns.as_deref(),
                )?),
                None => None,
            };

            model
                .predict_proba(&df, metadata_df.as_ref(), metadata_symbol.as_deref())
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

#[pyclass(unsendable)]
struct MLPModel {
    model: Mutex<RustMlpExpert>,
}

#[pymethods]
impl MLPModel {
    #[new]
    #[pyo3(signature = (
        idx=1,
        hidden_dim=256,
        n_layers=3,
        dropout=0.1,
        lr=1e-3,
        max_time_sec=36000,
        device="cpu",
        batch_size=4096,
    ))]
    fn new(
        idx: usize,
        hidden_dim: i64,
        n_layers: i64,
        dropout: f64,
        lr: f64,
        max_time_sec: u64,
        device: &str,
        batch_size: i64,
    ) -> PyResult<Self> {
        let _ = idx;
        let model = RustMlpExpert::new(
            hidden_dim,
            n_layers,
            dropout,
            lr,
            max_time_sec,
            device,
            batch_size,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("MLP init failed: {}", e))
        })?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_vec: Vec<i32> = labels.as_array().iter().copied().collect();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;
            model
                .fit(&features_array, &labels_vec)
                .map_err(|e| format!("Training failed: {}", e))?;
            Ok(())
        })();

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = (|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;
            model
                .predict_proba(&features_array)
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

// ============================================================================
// BURN NEURAL NETWORK MODELS — Pure Rust, no Python, no GIL
// ============================================================================

#[cfg(feature = "burn-backend")]
mod burn_bindings {
    use super::*;
    use forex_models::burn_models::*;

    /// Helper macro to create PyO3 wrappers for each Burn model
    macro_rules! burn_model_wrapper {
        (
            $py_name:ident,
            $config_type:ident,
            $model_type:ident,
            $default_hidden:expr,
            $default_layers:expr
        ) => {
            #[pyclass(unsendable, module = "forex_bindings")]
            pub struct $py_name {
                model: Option<$model_type<TrainBackend>>,
                input_dim: usize,
                hidden_dim: usize,
                n_classes: usize,
                lr: f64,
                batch_size: usize,
                max_epochs: usize,
                patience: usize,
            }

            #[pymethods]
            impl $py_name {
                #[new]
                #[pyo3(signature = (
                    input_dim=96,
                    hidden_dim=$default_hidden,
                    n_classes=3,
                    lr=1e-3,
                    batch_size=64,
                    max_epochs=100,
                    patience=8,
                ))]
                fn new(
                    input_dim: usize,
                    hidden_dim: usize,
                    n_classes: usize,
                    lr: f64,
                    batch_size: usize,
                    max_epochs: usize,
                    patience: usize,
                ) -> Self {
                    Self {
                        model: None,
                        input_dim,
                        hidden_dim,
                        n_classes,
                        lr,
                        batch_size,
                        max_epochs,
                        patience,
                    }
                }

                fn fit<'py>(
                    &mut self,
                    _py: Python<'py>,
                    features: PyReadonlyArray2<'py, f32>,
                    labels: PyReadonlyArray1<'py, i32>,
                ) -> PyResult<f64> {
                    let x = features.as_array().to_owned();
                    let y: Vec<i32> = labels.as_array().iter().copied().collect();

                    // Auto-detect input_dim from features
                    self.input_dim = x.ncols();

                    let device = <TrainBackend as burn::tensor::backend::Backend>::Device::default();
                    let config = $config_type::new(self.input_dim)
                        .with_hidden_dim(self.hidden_dim)
                        .with_n_classes(self.n_classes);
                    let model = config.init::<TrainBackend>(&device);

                    let train_config = TrainConfig {
                        lr: self.lr,
                        batch_size: self.batch_size,
                        max_epochs: self.max_epochs,
                        patience: self.patience,
                        n_classes: self.n_classes,
                    };

                    let (trained, best_loss) = train_model::<TrainBackend, _>(
                        model, &x, &y, &train_config,
                    );
                    self.model = Some(trained);
                    Ok(best_loss as f64)
                }

                fn predict_proba<'py>(
                    &self,
                    py: Python<'py>,
                    features: PyReadonlyArray2<'py, f32>,
                ) -> PyResult<Bound<'py, PyArray2<f32>>> {
                    let x = features.as_array().to_owned();
                    let model = self.model.as_ref().ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            "Model not trained yet. Call fit() first.",
                        )
                    })?;
                    let probs = predict_proba::<TrainBackend, _>(model, &x, self.batch_size);
                    Ok(probs.into_pyarray(py))
                }
            }
        };
    }

    burn_model_wrapper!(BurnMLPModel, BurnMLPConfig, BurnMLP, 256, 3);
    burn_model_wrapper!(BurnNBeatsModel, BurnNBeatsConfig, BurnNBeats, 64, 3);
    burn_model_wrapper!(BurnTiDEModel, BurnTiDEConfig, BurnTiDE, 128, 2);
    burn_model_wrapper!(BurnKANModel, BurnKANConfig, BurnKAN, 32, 2);
    burn_model_wrapper!(BurnTransformerModel, BurnTransformerConfig, BurnTransformer, 128, 4);

    // TabNet needs special handling (n_steps instead of n_layers)
    #[pyclass(unsendable, name = "BurnTabNetModel", module = "forex_bindings")]
    pub struct BurnTabNetModel {
        model: Option<BurnTabNet<TrainBackend>>,
        input_dim: usize,
        hidden_dim: usize,
        n_classes: usize,
        lr: f64,
        batch_size: usize,
        max_epochs: usize,
        patience: usize,
    }

    #[pymethods]
    impl BurnTabNetModel {
        #[new]
        #[pyo3(signature = (input_dim=96, hidden_dim=64, n_classes=3, lr=2e-3, batch_size=64, max_epochs=100, patience=8))]
        fn new(
            input_dim: usize, hidden_dim: usize, n_classes: usize,
            lr: f64, batch_size: usize, max_epochs: usize, patience: usize,
        ) -> Self {
            Self { model: None, input_dim, hidden_dim, n_classes, lr, batch_size, max_epochs, patience }
        }

        fn fit<'py>(
            &mut self, _py: Python<'py>,
            features: PyReadonlyArray2<'py, f32>,
            labels: PyReadonlyArray1<'py, i32>,
        ) -> PyResult<f64> {
            let x = features.as_array().to_owned();
            let y: Vec<i32> = labels.as_array().iter().copied().collect();
            self.input_dim = x.ncols();

            let device = <TrainBackend as burn::tensor::backend::Backend>::Device::default();
            let config = BurnTabNetConfig::new(self.input_dim)
                .with_hidden_dim(self.hidden_dim)
                .with_n_classes(self.n_classes);
            let model = config.init::<TrainBackend>(&device);

            let train_config = TrainConfig {
                lr: self.lr, batch_size: self.batch_size,
                max_epochs: self.max_epochs, patience: self.patience, n_classes: self.n_classes,
            };
            let (trained, best_loss) = train_model::<TrainBackend, _>(model, &x, &y, &train_config);
            self.model = Some(trained);
            Ok(best_loss as f64)
        }

        fn predict_proba<'py>(
            &self, py: Python<'py>,
            features: PyReadonlyArray2<'py, f32>,
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let x = features.as_array().to_owned();
            let model = self.model.as_ref().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not trained. Call fit() first.")
            })?;
            let probs = predict_proba::<TrainBackend, _>(model, &x, self.batch_size);
            Ok(probs.into_pyarray(py))
        }
    }

    /// Register all Burn model classes in the pymodule
    pub fn register_burn_models(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<BurnMLPModel>()?;
        m.add_class::<BurnNBeatsModel>()?;
        m.add_class::<BurnTiDEModel>()?;
        m.add_class::<BurnTabNetModel>()?;
        m.add_class::<BurnKANModel>()?;
        m.add_class::<BurnTransformerModel>()?;
        Ok(())
    }
}

use forex_core::domain::consistency::{ConsistencyTracker as CoreConsistencyTracker, TradeEvent};
use forex_core::domain::meta_controller::{MetaController as CoreMetaController, PropMetaState};

#[pyclass(name = "ConsistencyTracker")]
pub struct ConsistencyTracker {
    inner: CoreConsistencyTracker,
}

#[pymethods]
impl ConsistencyTracker {
    #[new]
    #[pyo3(signature = (cache_dir, lookback_days=30))]
    fn new(cache_dir: &Bound<'_, PyAny>, lookback_days: i64) -> Self {
        let _ = cache_dir;
        // We ignore cache_dir as the Rust implementation handles logic purely in-memory
        Self {
            inner: CoreConsistencyTracker::new(lookback_days),
        }
    }

    fn update(&mut self, trade_event: &Bound<'_, PyDict>) -> PyResult<()> {
        let entry_time: String = trade_event.get_item("entry_time")?.unwrap().extract()?;
        let pnl: f64 = trade_event.get_item("pnl")?.map(|x| x.extract().unwrap_or(0.0)).unwrap_or(0.0);
        let risk_pct: f64 = trade_event.get_item("risk_pct")?.map(|x| x.extract().unwrap_or(0.0)).unwrap_or(0.0);
        let size: f64 = trade_event.get_item("size")?.map(|x| x.extract().unwrap_or(0.0)).unwrap_or(0.0);
        let hold_minutes: f64 = trade_event.get_item("hold_minutes")?.map(|x| x.extract().unwrap_or(0.0)).unwrap_or(0.0);
        
        let win: Option<i32> = match trade_event.get_item("win")? {
            Some(v) => {
                if let Ok(b) = v.extract::<bool>() {
                    Some(if b { 1 } else { 0 })
                } else if let Ok(i) = v.extract::<i32>() {
                    Some(i)
                } else {
                    None
                }
            },
            None => None,
        };

        let event = TradeEvent { entry_time, pnl, risk_pct, size, hold_minutes, win };
        self.inner.update(&event);
        Ok(())
    }

    fn get_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let metrics = self.inner.get_metrics();
        
        // We return an object that quacks like the Python ConsistencyMetrics dataclass
        let dict = PyDict::new(py);
        dict.set_item("score", metrics.score)?;
        dict.set_item("daily_profit_consistency", metrics.daily_profit_consistency)?;
        dict.set_item("daily_trade_consistency", metrics.daily_trade_consistency)?;
        dict.set_item("daily_risk_consistency", metrics.daily_risk_consistency)?;
        dict.set_item("weekly_profit_consistency", metrics.weekly_profit_consistency)?;
        dict.set_item("weekly_drawdown_consistency", metrics.weekly_drawdown_consistency)?;
        dict.set_item("trade_size_consistency", metrics.trade_size_consistency)?;
        dict.set_item("hold_time_consistency", metrics.hold_time_consistency)?;
        dict.set_item("win_rate_rolling", metrics.win_rate_rolling)?;
        dict.set_item("grade", metrics.grade)?;
        
        // Return a mock dataclass object in Python
        let dataclass_module = PyModule::import(py, "forex_bot.execution.consistency")?;
        let class = dataclass_module.getattr("ConsistencyMetrics")?;
        let inst = class.call((), Some(&dict))?;
        Ok(inst)
    }
}

#[pyclass(name = "MetaController")]
pub struct MetaController {
    inner: CoreMetaController,
}

#[pymethods]
impl MetaController {
    #[new]
    #[pyo3(signature = (max_daily_dd=None, safety_buffer=None, base_risk_per_trade=None, base_confidence=None, settings=None, silent=None))]
    fn new(
        max_daily_dd: Option<f64>,
        safety_buffer: Option<f64>,
        base_risk_per_trade: Option<f64>,
        base_confidence: Option<f64>,
        settings: Option<&Bound<'_, PyAny>>,
        silent: Option<bool>,
    ) -> PyResult<Self> {
        let mut k_steepness = 200.0;
        let mut final_base_confidence = base_confidence.unwrap_or(0.55);

        if let Some(s) = settings {
            if let Ok(dyn_cfg) = s.getattr("dynamic") {
                if let Ok(risk_params) = dyn_cfg.call_method0("get") { // Might be dict
                    if let Ok(k) = risk_params.call_method1("get", ("risk_curve_steepness", 200.0)) {
                        k_steepness = k.extract().unwrap_or(200.0);
                    }
                    if let Ok(c) = risk_params.call_method1("get", ("confidence_threshold",)) {
                        if let Ok(c_val) = c.extract::<f64>() {
                            final_base_confidence = c_val;
                        }
                    }
                }
            }
        }

        Ok(Self {
            inner: CoreMetaController::new(max_daily_dd, safety_buffer, base_risk_per_trade, Some(final_base_confidence), silent, Some(k_steepness)),
        })
    }

    fn get_risk_parameters(&mut self, state: &Bound<'_, PyAny>) -> PyResult<(f64, f64, bool)> {
        let mut m_regime = "Normal".to_string();
        if let Ok(regime) = state.getattr("market_regime") {
            if let Ok(r) = regime.extract::<String>() {
                m_regime = r;
            }
        }
        
        let s = PropMetaState {
            daily_dd_pct: state.getattr("daily_dd_pct")?.extract()?,
            volatility_regime: state.getattr("volatility_regime")?.extract()?,
            recent_win_rate: state.getattr("recent_win_rate")?.extract()?,
            consecutive_losses: state.getattr("consecutive_losses")?.extract()?,
            model_confidence: state.getattr("model_confidence")?.extract()?,
            hour_of_day: state.getattr("hour_of_day")?.extract()?,
            market_regime: m_regime,
        };
        Ok(self.inner.get_risk_parameters(&s))
    }
}

#[pymodule]
fn forex_bindings(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ConsistencyTracker>()?;
    m.add_class::<MetaController>()?;
    m.add_class::<ForexCore>()?;
    m.add_class::<ConformalGate>()?;
    #[cfg(feature = "onnx")]
    m.add_class::<ModelEngine>()?;
    m.add_function(wrap_pyfunction!(search_evolve_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(search_evolve_gpu_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(search_discovery_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(load_symbol_frames, m)?)?;
    m.add_function(wrap_pyfunction!(load_symbol_features, m)?)?;
    m.add_function(wrap_pyfunction!(load_strategy_signals, m)?)?;
    m.add_function(wrap_pyfunction!(infer_stop_target_pips_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(fast_evaluate_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(batch_evaluate_strategies, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_population_talib_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_population_core_py, m)?)?;
    m.add_function(wrap_pyfunction!(trade_journal_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(triple_barrier_labels, m)?)?;
    m.add_function(wrap_pyfunction!(compute_position_size_lots, m)?)?;
    m.add_function(wrap_pyfunction!(pip_size_from_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(infer_pip_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(derive_time_index_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(count_weekday_trading_days, m)?)?;
    m.add_function(wrap_pyfunction!(align_ffill_values_by_ns, m)?)?;
    m.add_function(wrap_pyfunction!(align_exact_values_by_ns, m)?)?;
    m.add_function(wrap_pyfunction!(align_feature_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(sorted_index_order, m)?)?;
    m.add_function(wrap_pyfunction!(rank_scores_desc, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_news_features, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_news_activation, m)?)?;
    m.add_function(wrap_pyfunction!(extract_regime_features, m)?)?;
    m.add_function(wrap_pyfunction!(remap_labels_neutral_buy_sell, m)?)?;
    m.add_function(wrap_pyfunction!(remap_labels_sell_neutral_buy, m)?)?;
    m.add_function(wrap_pyfunction!(pad_probs_neutral_buy_sell, m)?)?;
    m.add_function(wrap_pyfunction!(margins_to_probs, m)?)?;
    m.add_function(wrap_pyfunction!(probs_to_signals, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_signals_and_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_prop_score_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(balanced_class_weights, m)?)?;
    m.add_function(wrap_pyfunction!(sample_weights_from_labels, m)?)?;
    m.add_function(wrap_pyfunction!(quick_backtest_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(sort_rows_with_labels_by_index, m)?)?;
    m.add_function(wrap_pyfunction!(sort_dedup_rows_by_index, m)?)?;
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
    m.add_class::<MLPModel>()?;
    m.add_class::<GeneticModel>()?;

    // Burn deep learning models (pure Rust)
    #[cfg(feature = "burn-backend")]
    burn_bindings::register_burn_models(m)?;

    Ok(())
}

