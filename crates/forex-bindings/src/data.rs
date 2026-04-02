use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use forex_data::{
    load_symbol_dataset, load_symbol_dataset_with_timeframes, prepare_multitimeframe_features_with_options,
    FeatureBuildOptions, FeatureCache, FeatureProfile, ensure_timeframes_with_resample,
};
use crate::utils::parse_feature_profile;

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
    let _ = max_features;
    let _ = max_htf_features;
    let base_prof = parse_feature_profile(Some(feature_profile), FeatureProfile::Standard);
    let _htf_prof = parse_feature_profile(htf_feature_profile, base_prof);
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

    let dataset = load_symbol_dataset(&root, &symbol)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load dataset failed: {}", e)))?;

    let dataset = if resample_missing && !higher.is_empty() {
        let refs: Vec<&str> = higher.iter().map(|s| s.as_str()).collect();
        ensure_timeframes_with_resample(&dataset, &base_tf, &refs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Resample failed: {}", e)))?
    } else {
        dataset
    };

    let cache = cache_dir
        .as_ref()
        .map(|dir| FeatureCache::new(dir, cache_ttl_minutes, cache_enabled));
    
    let frame = prepare_multitimeframe_features_with_options(
        &dataset,
        &base_tf,
        &options,
        cache.as_ref(),
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Feature computation failed: {}", e)))?;

    let base = dataset
        .frames
        .get(&base_tf)
        .cloned()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Base timeframe data missing"))?;

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
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
pub fn load_strategy_signals(py: Python) -> PyResult<Py<PyAny>> {
    Ok(PyDict::new(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (timestamps))]
pub fn derive_time_index_arrays(
    py: Python,
    timestamps: PyReadonlyArray1<i64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (months, days) = forex_search::month_day_indices(timestamps.as_slice()?);
    Ok((months.into_pyarray(py).into_any().unbind(), days.into_pyarray(py).into_any().unbind()))
}

#[pyfunction]
pub fn count_weekday_trading_days(_py: Python) -> PyResult<usize> {
    Ok(0)
}

#[pyfunction]
pub fn align_ffill_values_by_ns(py: Python) -> PyResult<Py<PyAny>> {
    Ok(py.None())
}

#[pyfunction]
pub fn align_exact_values_by_ns(py: Python) -> PyResult<Py<PyAny>> {
    Ok(py.None())
}

#[pyfunction]
pub fn align_feature_matrix(py: Python) -> PyResult<Py<PyAny>> {
    Ok(py.None())
}

#[pyfunction]
pub fn sorted_index_order(py: Python) -> PyResult<Py<PyAny>> {
    Ok(py.None())
}

#[pyfunction]
pub fn aggregate_news_features(py: Python) -> PyResult<Py<PyAny>> {
    Ok(py.None())
}

#[pyfunction]
pub fn aggregate_news_activation(py: Python) -> PyResult<Py<PyAny>> {
    Ok(py.None())
}
