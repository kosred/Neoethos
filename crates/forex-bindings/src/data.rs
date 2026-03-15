use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::HashSet;
use std::path::PathBuf;
use ndarray::Array2;
use chrono::{Datelike, TimeZone, Utc};
use forex_data::{
    ensure_timeframes_with_resample, load_symbol_dataset,
    load_symbol_dataset_with_timeframes, prepare_multitimeframe_features_with_options,
    FeatureBuildOptions, FeatureCache, FeatureProfile, Ohlcv,
};
use crate::utils::{
    parse_feature_profile, resolve_base_tf, vec_from_py_i64, vec_from_py_f64,
    map_indicator_index,
};

#[pyfunction]
#[pyo3(signature = (index_ns))]
pub fn derive_time_index_arrays<'py>(
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
pub fn count_weekday_trading_days(index_ns: PyReadonlyArray1<'_, i64>) -> PyResult<usize> {
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
pub fn align_ffill_values_by_ns<'py>(
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
pub fn align_exact_values_by_ns<'py>(
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
pub fn align_feature_matrix<'py>(
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
pub fn sorted_index_order<'py>(
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
#[pyo3(signature = (base_idx_ns, event_idx_ns, event_sent, event_conf, lookback_ns))]
pub fn aggregate_news_features<'py>(
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
pub fn aggregate_news_activation<'py>(
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
#[pyo3(signature = (root, symbol, base_tf="M5", higher_tfs=None, conhecimento=None))]
pub fn load_symbol_frames(
    py: Python,
    root: String,
    symbol: String,
    base_tf: &str,
    higher_tfs: Option<Vec<String>>,
    conhecimento: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let _ = conhecimento;
    let result: Result<forex_data::SymbolDataset, String> = py.detach(|| {
        let higher = higher_tfs.unwrap_or_default();
        if higher.is_empty() {
            load_symbol_dataset(&root, &symbol).map_err(|e| format!("Load dataset failed: {}", e))
        } else {
            let mut merged = higher.clone();
            if !merged.iter().any(|tf| tf.eq_ignore_ascii_case(&base_tf)) {
                merged.push(base_tf.to_string());
            }
            let refs: Vec<&str> = merged.iter().map(|s| s.as_str()).collect();
            load_symbol_dataset_with_timeframes(&root, &symbol, &refs)
                .map_err(|e| format!("Load dataset failed: {}", e))
        }
    });

    let dataset = result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;
    let dict = PyDict::new(py);
    for (tf, ohlcv) in dataset.frames {
        let tf_dict = PyDict::new(py);
        tf_dict.set_item("open", ohlcv.open)?;
        tf_dict.set_item("high", ohlcv.high)?;
        tf_dict.set_item("low", ohlcv.low)?;
        tf_dict.set_item("close", ohlcv.close)?;
        if let Some(vol) = ohlcv.volume {
            tf_dict.set_item("volume", vol)?;
        }
        if let Some(ts) = ohlcv.timestamp {
            tf_dict.set_item("timestamp", ts)?;
        }
        dict.set_item(tf, tf_dict)?;
    }
    Ok(dict.into_any().into())
}

#[pyfunction]
#[pyo3(signature = (
    root,
    symbol,
    base_tf="M1",
    higher_tfs=None,
    cache_dir=None,
    cache_ttl_minutes=0,
    cache_enabled=false,
    resample_missing=true,
    feature_profile="full",
    htf_feature_profile=None,
    max_features=0,
    max_htf_features=0
))]
pub fn load_symbol_features(
    py: Python,
    root: String,
    symbol: String,
    base_tf: &str,
    higher_tfs: Option<Vec<String>>,
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

    let result: Result<(forex_data::FeatureFrame, String, Ohlcv), String> = py.detach(|| {
        let higher = higher_tfs.unwrap_or_default();
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

        Ok((frame, base_final, base))
    });

    let (frame, base_final, base) =
        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;
    let dict = PyDict::new(py);
    dict.set_item("data", frame.data.into_pyarray(py))?;
    dict.set_item("names", frame.names)?;
    dict.set_item("timestamps", frame.timestamps)?;
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
pub fn load_strategy_signals(
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

    let result: Result<(forex_data::FeatureFrame, String, Ohlcv, Vec<String>, Array2<i8>), String> =
        py.detach(|| {
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

            let specs_res = crate::utils::load_strategy_specs(&path);
            let mut specs = match specs_res {
                Ok(s) => s,
                Err(e) => return Err(e),
            };
            if portfolio_limit > 0 && specs.len() > portfolio_limit {
                specs.truncate(portfolio_limit);
            }

            let mut strategy_ids: Vec<String> = Vec::new();
            let mut genes: Vec<forex_search::Gene> = Vec::new();
            for (idx, spec) in specs.into_iter().enumerate() {
                let indicators = spec.indicators.clone().unwrap_or_default();
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
                    ..Default::default()
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
                    if row_idx < n_rows {
                        out[(row_idx, col_idx)] = *v;
                    }
                }
            }

            Ok((frame, base_final, base, strategy_ids, out))
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
