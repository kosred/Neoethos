use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use polars::prelude::*;
use pyo3::prelude::*;
use std::collections::HashMap;
use forex_data::{FeatureProfile, Ohlcv};

pub fn dataframe_from_ndarray(features: &Array2<f64>) -> Result<DataFrame, String> {
    let mut df_data: Vec<Column> = Vec::with_capacity(features.ncols());
    for col_idx in 0..features.ncols() {
        let col_data: Vec<f64> = features.column(col_idx).iter().copied().collect();
        let name = format!("feature_{col_idx}");
        df_data.push(Series::new(name.into(), col_data).into());
    }
    DataFrame::new(df_data).map_err(|e| format!("DataFrame creation failed: {}", e))
}

pub fn dataframe_from_named_ndarray(
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

pub fn vec_from_py_f64(arr: &PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

pub fn vec_from_py_i64(arr: &PyReadonlyArray1<i64>) -> Vec<i64> {
    arr.as_array().iter().copied().collect()
}

pub fn vec_from_py_i8(arr: &PyReadonlyArray1<i8>) -> Vec<i8> {
    arr.as_array().iter().copied().collect()
}

pub fn vec_from_py_i32(arr: &PyReadonlyArray1<i32>) -> Vec<i32> {
    arr.as_array().iter().copied().collect()
}

pub fn vec_from_py_f32(arr: &PyReadonlyArray1<f32>) -> Vec<f32> {
    arr.as_array().iter().copied().collect()
}

pub fn build_ohlcv(
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

pub fn resolve_base_tf(dataset: &forex_data::SymbolDataset, preferred: &str) -> String {
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

pub fn normalize_indicator_name(input: &str) -> String {
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

pub fn map_indicator_index(indicator: &str, feature_names: &[String]) -> Option<usize> {
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

pub fn parse_feature_profile(raw: Option<&str>, default: FeatureProfile) -> FeatureProfile {
    let Some(value) = raw else {
        return default;
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return default;
    }
    forex_data::FeatureProfile::from_str(trimmed)
}

pub fn norm_symbol(symbol: &str) -> String {
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

pub fn split_symbol(symbol: &str) -> Option<(String, String)> {
    let sym = norm_symbol(symbol);
    if sym.len() == 6 && sym.chars().all(|c| c.is_ascii_alphabetic()) {
        Some((sym[..3].to_string(), sym[3..6].to_string()))
    } else {
        None
    }
}

pub fn symbol_kind(symbol: &str, parts: Option<&(String, String)>) -> &'static str {
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

pub fn pip_size_from_parts(symbol: &str, parts: Option<&(String, String)>) -> f64 {
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

pub fn contract_size_from_parts(symbol: &str, parts: Option<&(String, String)>) -> f64 {
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

pub fn quote_to_account_rate(
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
#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
#[derive(Debug, Clone)]
pub enum ParamValue {
    Bool(bool),
    Int(i32),
    Float(f64),
    String(String),
}

#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
pub fn param_value_from_py(value: &Bound<'_, PyAny>) -> Option<ParamValue> {
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
pub fn params_from_py(
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
            .map_err(|_| "params keys must be strings".to_string())? ;
        if let Some(val) = param_value_from_py(&v) {
            map.insert(key, val);
        }
    }
    Ok(Some(map))
}

#[derive(Debug, serde::Deserialize)]
pub struct StrategySpec {
    pub indicators: Option<Vec<String>>,
    pub weights: Option<std::collections::HashMap<String, f64>>,
    pub long_threshold: Option<f64>,
    pub short_threshold: Option<f64>,
    pub strategy_id: Option<String>,
    pub tp_pips: Option<f64>,
    pub sl_pips: Option<f64>,
    pub use_ob: Option<bool>,
    pub use_fvg: Option<bool>,
    pub use_liq_sweep: Option<bool>,
    pub mtf_confirmation: Option<bool>,
    pub use_premium_discount: Option<bool>,
    pub use_inducement: Option<bool>,
    pub use_bos: Option<bool>,
    pub use_choch: Option<bool>,
    pub use_eqh: Option<bool>,
    pub use_eql: Option<bool>,
    pub use_displacement: Option<bool>,
}

pub fn load_strategy_specs(path: &std::path::PathBuf) -> Result<Vec<StrategySpec>, String> {
    let content = std::fs::read_to_string(path)
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

#[pyfunction]
pub fn rank_scores_desc(scores: Vec<f64>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    indices
}

#[pyfunction]
pub fn probs_to_signals(probs: PyReadonlyArray2<f32>, threshold: f32) -> Vec<i8> {
    let arr = probs.as_array();
    let rows = arr.nrows();
    let mut out = vec![0i8; rows];
    for r in 0..rows {
        let p_buy = arr[(r, 1)];
        let p_sell = arr[(r, 2)];
        if p_buy > threshold && p_buy > p_sell {
            out[r] = 1;
        } else if p_sell > threshold && p_sell > p_buy {
            out[r] = -1;
        }
    }
    out
}

#[pyfunction]
pub fn threshold_signals_and_accuracy(
    probs: PyReadonlyArray2<f32>,
    labels: PyReadonlyArray1<i64>,
    threshold: f32,
) -> (Vec<i8>, f64) {
    let p = probs.as_array();
    let l = labels.as_array();
    let n = p.nrows();
    let mut signals = vec![0i8; n];
    let mut correct = 0usize;
    let mut total = 0usize;

    for i in 0..n {
        let p1 = p[(i, 1)];
        let p2 = p[(i, 2)];
        let sig = if p1 > threshold && p1 > p2 {
            1i8
        } else if p2 > threshold && p2 > p1 {
            -1i8
        } else {
            0i8
        };
        signals[i] = sig;
        if sig != 0 {
            total += 1;
            let target = match l[i] {
                -1 => -1i8,
                1 => 1i8,
                _ => 0i8,
            };
            if sig == target {
                correct += 1;
            }
        }
    }
    let acc = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
    (signals, acc)
}

#[pyfunction]
pub fn balanced_class_weights(labels: PyReadonlyArray1<i64>) -> Vec<f64> {
    let l = labels.as_array();
    let mut counts = [0usize; 3];
    for &val in l.iter() {
        match val {
            -1 => counts[2] += 1,
            0 => counts[0] += 1,
            1 => counts[1] += 1,
            _ => {}
        }
    }
    let n = l.len() as f64;
    let mut weights = vec![1.0; 3];
    for i in 0..3 {
        if counts[i] > 0 {
            weights[i] = n / (3.0 * counts[i] as f64);
        }
    }
    weights
}

#[pyfunction]
pub fn sample_weights_from_labels(labels: PyReadonlyArray1<i64>, weights: Vec<f64>) -> Vec<f64> {
    let l = labels.as_array();
    l.iter().map(|&val| {
        let idx = match val {
            -1 => 2,
            0 => 0,
            1 => 1,
            _ => 0,
        };
        weights.get(idx).copied().unwrap_or(1.0)
    }).collect()
}

#[pyfunction]
pub fn sort_rows_with_labels_by_index<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    features: PyReadonlyArray2<'py, f32>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Vec<i64>, Bound<'py, PyArray2<f32>>, Vec<i64>)> {
    let idx = index.as_array();
    let mut p: Vec<usize> = (0..idx.len()).collect();
    p.sort_by_key(|&i| idx[i]);
    
    let mut sorted_idx = Vec::with_capacity(idx.len());
    let mut sorted_l = Vec::with_capacity(idx.len());
    let f = features.as_array();
    let mut sorted_f = Array2::<f32>::zeros((f.nrows(), f.ncols()));
    let l = labels.as_array();
    
    for (new_row, &old_row) in p.iter().enumerate() {
        if new_row >= f.nrows() { break; }
        sorted_idx.push(idx[old_row]);
        sorted_l.push(l[old_row]);
        for col in 0..f.ncols() {
            sorted_f[(new_row, col)] = f[(old_row, col)];
        }
    }
    Ok((sorted_idx, sorted_f.into_pyarray(py), sorted_l))
}

#[pyfunction]
pub fn sort_dedup_rows_by_index<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    features: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Vec<i64>, Bound<'py, PyArray2<f32>>)> {
    let idx = index.as_array();
    let mut p: Vec<usize> = (0..idx.len()).collect();
    p.sort_by_key(|&i| idx[i]);
    
    let mut sorted_idx = Vec::new();
    let f = features.as_array();
    let mut rows = Vec::new();
    
    let mut last = None;
    for &old_row in &p {
        let current = idx[old_row];
        if Some(current) != last {
            sorted_idx.push(current);
            rows.push(old_row);
            last = Some(current);
        }
    }
    
    let mut sorted_f = Array2::<f32>::zeros((rows.len(), f.ncols()));
    for (new_row, &old_row) in rows.iter().enumerate() {
        for col in 0..f.ncols() {
            sorted_f[(new_row, col)] = f[(old_row, col)];
        }
    }
    Ok((sorted_idx, sorted_f.into_pyarray(py)))
}
