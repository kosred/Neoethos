use crate::utils::parse_feature_profile;
use chrono::{Datelike, TimeZone, Utc, Weekday};
use forex_data::{
    FeatureBuildOptions, FeatureCache, FeatureFrame, FeatureProfile, align_features_by_ns,
    compute_hpc_feature_frame, ensure_timeframes_with_resample, load_symbol_dataset,
    load_symbol_dataset_with_timeframes, prepare_multitimeframe_features_with_options,
};
use ndarray::{Array2, Axis, s};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::{HashMap, HashSet};
use std::fs;

#[pyfunction]
#[pyo3(signature = (root, symbol, base_tf=String::from("M1"), higher_tfs=None, cache_dir=None, cache_ttl_minutes=60, cache_enabled=true, resample_missing=true, feature_profile="standard", htf_feature_profile=None, max_features=0, max_htf_features=0))]
#[allow(clippy::too_many_arguments)]
pub fn load_symbol_features(
    py: Python,
    root: String,
    symbol: String,
    base_tf: String,
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
    let base_prof = parse_feature_profile(Some(feature_profile), FeatureProfile::Standard);
    let htf_prof = parse_feature_profile(htf_feature_profile, base_prof);
    let higher = higher_tfs.unwrap_or_default();

    let options = FeatureBuildOptions {
        profile: base_prof,
        include_smc: true,
        include_hpc_ta: true,
        include_regime: true,
        include_quant: true,
        prefix_base_features: false,
        higher_tfs: higher.clone(),
    };

    let dataset = load_symbol_dataset(&root, &symbol).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load dataset failed: {e}"))
    })?;

    let dataset = if resample_missing && !higher.is_empty() {
        let refs: Vec<&str> = higher.iter().map(|s| s.as_str()).collect();
        ensure_timeframes_with_resample(&dataset, &base_tf, &refs).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Resample failed: {e}"))
        })?
    } else {
        dataset
    };

    let cache = cache_dir
        .as_ref()
        .map(|dir| FeatureCache::new(dir, cache_ttl_minutes, cache_enabled));

    let frame = prepare_features_with_optional_htf_profile(
        &dataset,
        &base_tf,
        &options,
        htf_prof,
        cache.as_ref(),
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Feature computation failed: {e}"
        ))
    })?;
    let frame = limit_feature_frame(frame, &higher, max_features, max_htf_features);

    let base = dataset.frames.get(&base_tf).cloned().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Base timeframe data missing")
    })?;

    let dict = PyDict::new(py);
    dict.set_item("data", frame.data.into_pyarray(py))?;
    dict.set_item("names", frame.names)?;
    dict.set_item("timestamps", frame.timestamps)?;
    dict.set_item("base_tf", base_tf)?;
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
    Ok(dict.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (root, symbol, timeframes=None))]
pub fn load_symbol_frames(
    py: Python,
    root: String,
    symbol: String,
    timeframes: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    let dataset = if let Some(tfs) = timeframes {
        let refs: Vec<&str> = tfs.iter().map(|s| s.as_str()).collect();
        load_symbol_dataset_with_timeframes(&root, &symbol, &refs)
    } else {
        load_symbol_dataset(&root, &symbol)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let dict = PyDict::new(py);
    for (tf, ohlcv) in dataset.frames {
        let frame_dict = PyDict::new(py);
        frame_dict.set_item("open", ohlcv.open)?;
        frame_dict.set_item("high", ohlcv.high)?;
        frame_dict.set_item("low", ohlcv.low)?;
        frame_dict.set_item("close", ohlcv.close)?;
        if let Some(ts) = ohlcv.timestamp {
            frame_dict.set_item("timestamp", ts)?;
        }
        if let Some(vol) = ohlcv.volume {
            frame_dict.set_item("volume", vol)?;
        }
        dict.set_item(tf, frame_dict)?;
    }
    Ok(dict.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (path=None))]
pub fn load_strategy_signals(py: Python, path: Option<String>) -> PyResult<Py<PyAny>> {
    let path = path.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "load_strategy_signals requires a JSON file path",
        )
    })?;
    let content = fs::read_to_string(&path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to read strategy signal file {path}: {e}"
        ))
    })?;
    let value: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid strategy signal JSON {path}: {e}"
        ))
    })?;
    pythonize::pythonize(py, &value)
        .map(|obj| obj.unbind())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (timestamps))]
pub fn derive_time_index_arrays(
    py: Python,
    timestamps: PyReadonlyArray1<i64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (months, days) = forex_search::month_day_indices(timestamps.as_slice()?);
    Ok((
        months.into_pyarray(py).into_any().unbind(),
        days.into_pyarray(py).into_any().unbind(),
    ))
}

#[pyfunction]
#[pyo3(signature = (timestamps))]
pub fn count_weekday_trading_days(timestamps: PyReadonlyArray1<i64>) -> PyResult<usize> {
    let mut days = HashSet::new();
    for &ts in timestamps.as_slice()? {
        if let Some(dt) = Utc.timestamp_millis_opt(ts).single()
            && !matches!(dt.weekday(), Weekday::Sat | Weekday::Sun)
        {
            days.insert(dt.date_naive());
        }
    }
    Ok(days.len())
}

#[pyfunction]
#[pyo3(signature = (base_ns, feature_ns, values))]
pub fn align_ffill_values_by_ns(
    py: Python,
    base_ns: PyReadonlyArray1<i64>,
    feature_ns: PyReadonlyArray1<i64>,
    values: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyAny>> {
    let out = align_scalar_values_by_ns(
        base_ns.as_slice()?,
        feature_ns.as_slice()?,
        values.as_slice()?,
        true,
    )?;
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (base_ns, feature_ns, values))]
pub fn align_exact_values_by_ns(
    py: Python,
    base_ns: PyReadonlyArray1<i64>,
    feature_ns: PyReadonlyArray1<i64>,
    values: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyAny>> {
    let out = align_scalar_values_by_ns(
        base_ns.as_slice()?,
        feature_ns.as_slice()?,
        values.as_slice()?,
        false,
    )?;
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (base_ns, feature_ns, feature_data, ffill=true))]
pub fn align_feature_matrix(
    py: Python,
    base_ns: PyReadonlyArray1<i64>,
    feature_ns: PyReadonlyArray1<i64>,
    feature_data: PyReadonlyArray2<f32>,
    ffill: bool,
) -> PyResult<Py<PyAny>> {
    let base = base_ns.as_slice()?;
    let feature = feature_ns.as_slice()?;
    let data = feature_data.as_array().to_owned();
    if feature.len() != data.nrows() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "feature_ns length must match feature_data rows",
        ));
    }
    let out = align_features_by_ns(base, feature, &data, ffill);
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (timestamps))]
pub fn sorted_index_order(timestamps: PyReadonlyArray1<i64>) -> PyResult<Vec<usize>> {
    let mut order = timestamps
        .as_slice()?
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    order.sort_by_key(|(idx, ts)| (*ts, *idx));
    Ok(order.into_iter().map(|(idx, _)| idx).collect())
}

#[pyfunction]
#[pyo3(signature = (base_ns, event_ns, event_scores, half_life_ms=3_600_000.0, lookback_ms=86_400_000))]
pub fn aggregate_news_features(
    py: Python,
    base_ns: PyReadonlyArray1<i64>,
    event_ns: PyReadonlyArray1<i64>,
    event_scores: PyReadonlyArray1<f32>,
    half_life_ms: f64,
    lookback_ms: i64,
) -> PyResult<Py<PyAny>> {
    let out = aggregate_news_feature_values(
        base_ns.as_slice()?,
        event_ns.as_slice()?,
        event_scores.as_slice()?,
        half_life_ms,
        lookback_ms,
    )?;
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (base_ns, event_ns, event_scores, half_life_ms=3_600_000.0, lookback_ms=86_400_000, threshold=0.0))]
pub fn aggregate_news_activation(
    base_ns: PyReadonlyArray1<i64>,
    event_ns: PyReadonlyArray1<i64>,
    event_scores: PyReadonlyArray1<f32>,
    half_life_ms: f64,
    lookback_ms: i64,
    threshold: f32,
) -> PyResult<Vec<i8>> {
    let values = aggregate_news_feature_values(
        base_ns.as_slice()?,
        event_ns.as_slice()?,
        event_scores.as_slice()?,
        half_life_ms,
        lookback_ms,
    )?;
    let threshold = threshold.abs();
    Ok(values
        .into_iter()
        .map(|value| {
            if value > threshold {
                1
            } else if value < -threshold {
                -1
            } else {
                0
            }
        })
        .collect())
}

fn prepare_features_with_optional_htf_profile(
    dataset: &forex_data::SymbolDataset,
    base_tf: &str,
    options: &FeatureBuildOptions,
    htf_profile: FeatureProfile,
    cache: Option<&FeatureCache>,
) -> anyhow::Result<FeatureFrame> {
    if htf_profile == options.profile {
        return prepare_multitimeframe_features_with_options(dataset, base_tf, options, cache);
    }

    let base_ohlcv = dataset
        .frames
        .get(base_tf)
        .ok_or_else(|| anyhow::anyhow!("base tf missing"))?;
    let base_ns = base_ohlcv
        .timestamp
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("base has no timestamps"))?;
    let base_feats = compute_hpc_feature_frame(base_ohlcv, options.profile)?;

    let mut all_names = base_feats
        .names
        .iter()
        .map(|name| {
            if options.prefix_base_features {
                format!("{base_tf}_{name}")
            } else {
                name.clone()
            }
        })
        .collect::<Vec<_>>();
    let mut all_data_parts = vec![base_feats.data];

    for h_tf in &options.higher_tfs {
        if let Some(h_ohlcv) = dataset.frames.get(h_tf) {
            let h_feats = compute_hpc_feature_frame(h_ohlcv, htf_profile)?;
            let h_ns = h_ohlcv
                .timestamp
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("higher tf {h_tf} has no timestamps"))?;
            let aligned = align_features_by_ns(base_ns, h_ns, &h_feats.data, true);
            all_names.extend(h_feats.names.iter().map(|name| format!("{h_tf}_{name}")));
            all_data_parts.push(aligned);
        }
    }

    let total_cols = all_data_parts.iter().map(|part| part.ncols()).sum();
    let mut merged = Array2::zeros((base_ns.len(), total_cols));
    let mut curr_col = 0;
    for part in all_data_parts {
        let ncols = part.ncols();
        merged
            .slice_mut(s![.., curr_col..curr_col + ncols])
            .assign(&part);
        curr_col += ncols;
    }

    Ok(FeatureFrame {
        timestamps: base_ns.clone(),
        names: all_names,
        data: merged,
    })
}

fn limit_feature_frame(
    frame: FeatureFrame,
    higher_tfs: &[String],
    max_base_features: usize,
    max_htf_features: usize,
) -> FeatureFrame {
    if max_base_features == 0 && max_htf_features == 0 {
        return frame;
    }

    let mut selected = Vec::new();
    let mut selected_names = Vec::new();
    let mut base_count = 0usize;
    let mut htf_counts = HashMap::<String, usize>::new();

    for (idx, name) in frame.names.iter().enumerate() {
        let htf = higher_tfs
            .iter()
            .find(|tf| name.starts_with(&format!("{tf}_")));
        if let Some(tf) = htf {
            let count = htf_counts.entry(tf.clone()).or_default();
            if max_htf_features == 0 || *count < max_htf_features {
                selected.push(idx);
                selected_names.push(name.clone());
                *count += 1;
            }
        } else if max_base_features == 0 || base_count < max_base_features {
            selected.push(idx);
            selected_names.push(name.clone());
            base_count += 1;
        }
    }

    let data = if selected.is_empty() {
        Array2::zeros((frame.data.nrows(), 0))
    } else {
        frame.data.select(Axis(1), &selected)
    };

    FeatureFrame {
        timestamps: frame.timestamps,
        names: selected_names,
        data,
    }
}

fn align_scalar_values_by_ns(
    base_ns: &[i64],
    feature_ns: &[i64],
    values: &[f64],
    ffill: bool,
) -> PyResult<Vec<f64>> {
    if feature_ns.len() != values.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "feature_ns length must match values length",
        ));
    }

    let mut out = vec![f64::NAN; base_ns.len()];
    let mut feature_idx = 0usize;
    for (row, &ts) in base_ns.iter().enumerate() {
        while feature_idx < feature_ns.len() && feature_ns[feature_idx] <= ts {
            feature_idx += 1;
        }
        let best_idx = if feature_idx > 0 {
            let prev = feature_idx - 1;
            if feature_ns[prev] == ts || ffill {
                Some(prev)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(idx) = best_idx {
            out[row] = values[idx];
        }
    }
    Ok(out)
}

fn aggregate_news_feature_values(
    base_ns: &[i64],
    event_ns: &[i64],
    event_scores: &[f32],
    half_life_ms: f64,
    lookback_ms: i64,
) -> PyResult<Vec<f32>> {
    if event_ns.len() != event_scores.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "event_ns length must match event_scores length",
        ));
    }

    let mut events = event_ns
        .iter()
        .copied()
        .zip(event_scores.iter().copied())
        .collect::<Vec<_>>();
    events.sort_by_key(|(ts, _)| *ts);

    let mut base_order = base_ns.iter().copied().enumerate().collect::<Vec<_>>();
    base_order.sort_by_key(|(idx, ts)| (*ts, *idx));

    let mut out = vec![0.0_f32; base_ns.len()];
    let mut start = 0usize;
    let mut end = 0usize;
    let use_decay = half_life_ms.is_finite() && half_life_ms > 0.0;

    for (base_idx, ts) in base_order {
        while end < events.len() && events[end].0 <= ts {
            end += 1;
        }
        if lookback_ms > 0 {
            let min_ts = ts.saturating_sub(lookback_ms);
            while start < end && events[start].0 < min_ts {
                start += 1;
            }
        }

        let mut score = 0.0_f64;
        for &(event_ts, event_score) in &events[start..end] {
            let age_ms = ts.saturating_sub(event_ts) as f64;
            let decay = if use_decay {
                (-age_ms / half_life_ms).exp()
            } else {
                1.0
            };
            score += event_score as f64 * decay;
        }
        out[base_idx] = score as f32;
    }

    Ok(out)
}
