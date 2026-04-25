use crate::utils::{vec_from_py_f64, vec_from_py_i64};
use ndarray::{Array2, Ix1, Ix2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;

type PyArray1Pair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

pub fn causal_tanh_zscore_column(
    data: &Array2<f32>,
    col_idx: usize,
    min_periods: usize,
) -> Vec<f64> {
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

#[pyfunction(name = "causal_tanh_zscore")]
#[pyo3(signature = (values, min_periods=30))]
pub fn causal_tanh_zscore_py<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    min_periods: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = values.as_array();
    let n = arr.len();
    let mut out = vec![0.0_f64; n];
    if n == 0 {
        return Ok(PyArray1::from_vec(py, out));
    }

    let mut count: usize = 0;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    for i in 0..n {
        let v = arr[i];
        if count >= min_periods {
            let var = m2 / (count as f64);
            let std = if var > 0.0 { var.sqrt() } else { 0.0 };
            let z = if std > 1e-12 {
                (v - mean) / std
            } else {
                v - mean
            };
            out[i] = z.tanh();
        }
        if v.is_finite() {
            count += 1;
            let delta = v - mean;
            mean += delta / (count as f64);
            let delta2 = v - mean;
            m2 += delta * delta2;
        }
    }
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction(name = "detect_divergence")]
#[pyo3(signature = (price, indicator, window=20))]
pub fn detect_divergence_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    indicator: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = price.as_array();
    let ind = indicator.as_array();
    let res = forex_data::detect_divergence(
        p.as_slice().unwrap_or(&[]),
        ind.as_slice().unwrap_or(&[]),
        window,
    );
    Ok(PyArray1::from_vec(py, res))
}

#[pyfunction(name = "vortex_indicator")]
#[pyo3(signature = (high, low, close, period=14))]
pub fn vortex_indicator_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<PyArray1Pair<'py>> {
    let (vp, vm) = forex_data::vortex_indicator(
        high.as_array().as_slice().unwrap_or(&[]),
        low.as_array().as_slice().unwrap_or(&[]),
        close.as_array().as_slice().unwrap_or(&[]),
        period,
    );
    Ok((PyArray1::from_vec(py, vp), PyArray1::from_vec(py, vm)))
}

#[pyfunction(name = "fisher_transform")]
#[pyo3(signature = (price, period=10))]
pub fn fisher_transform_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let res = forex_data::fisher_transform(price.as_array().as_slice().unwrap_or(&[]), period);
    Ok(PyArray1::from_vec(py, res))
}

#[pyfunction]
#[pyo3(signature = (close_prices, adx_values=None, volatility_window=20))]
pub fn extract_regime_features<'py>(
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
pub fn remap_labels_neutral_buy_sell<'py>(
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
pub fn remap_labels_sell_neutral_buy<'py>(
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
pub fn pad_probs_neutral_buy_sell<'py>(
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
            ));
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
pub fn margins_to_probs<'py>(
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
                let p1 = 1.0 / (1.0 + (-value).exp());
                out[(i, 0)] = 1.0 - p1;
                out[(i, 1)] = p1;
            }
            Ok(out.into_pyarray(py))
        }
        2 => {
            let arr = view.into_dimensionality::<Ix2>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("decision must be 1D or 2D")
            })?;
            let rows = arr.nrows();
            let cols = arr.ncols();
            let mut out = Array2::<f64>::zeros((rows, 3));
            if cols >= 2 {
                for r in 0..rows {
                    let mut sum_exp = 0.0_f64;
                    for c in 0..cols.min(3) {
                        sum_exp += arr[(r, c)].exp();
                    }
                    if sum_exp > 0.0 {
                        for c in 0..cols.min(3) {
                            out[(r, c)] = arr[(r, c)].exp() / sum_exp;
                        }
                    }
                }
            } else {
                for r in 0..rows {
                    let value = arr[(r, 0)];
                    let p1 = 1.0 / (1.0 + (-value).exp());
                    out[(r, 0)] = 1.0 - p1;
                    out[(r, 1)] = p1;
                }
            }
            Ok(out.into_pyarray(py))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "decision must be 1D or 2D",
        )),
    }
}
