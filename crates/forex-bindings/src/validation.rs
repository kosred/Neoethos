use crate::utils::{vec_from_py_f64, vec_from_py_i8, vec_from_py_i64};
use forex_search::eval::BacktestSettings;
use forex_search::validation::{
    CombinatorialPurgedCV, WalkforwardBacktestInput, embargoed_walkforward_backtest,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

#[pyfunction]
#[pyo3(signature = (
    close,
    high,
    low,
    signals,
    months,
    days,
    train_ratio=0.7,
    n_splits=5,
    embargo_bars=120,
    settings=None,
    max_daily_loss_pct=0.05,
    max_daily_profit_pct=0.50,
    min_trading_days=3,
    max_trades_per_day=15
))]
#[allow(clippy::too_many_arguments)]
pub fn embargoed_walkforward_backtest_py(
    py: Python,
    close: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    signals: PyReadonlyArray1<i8>,
    months: PyReadonlyArray1<i64>,
    days: PyReadonlyArray1<i64>,
    train_ratio: f64,
    n_splits: usize,
    embargo_bars: usize,
    settings: Option<Py<PyAny>>,
    max_daily_loss_pct: f64,
    max_daily_profit_pct: f64,
    min_trading_days: usize,
    max_trades_per_day: usize,
) -> PyResult<Py<PyAny>> {
    let cl = vec_from_py_f64(&close);
    let hi = vec_from_py_f64(&high);
    let lo = vec_from_py_f64(&low);
    let sig = vec_from_py_i8(&signals);
    let m = vec_from_py_i64(&months);
    let d = vec_from_py_i64(&days);

    let b_settings = backtest_settings_from_py(py, settings)?;

    let res = embargoed_walkforward_backtest(WalkforwardBacktestInput {
        close: &cl,
        high: &hi,
        low: &lo,
        signals: &sig,
        months: &m,
        days: &d,
        train_ratio,
        n_splits,
        embargo_bars,
        settings: &b_settings,
        max_daily_loss_pct,
        max_daily_profit_pct,
        min_trading_days,
        max_trades_per_day,
    })
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    pythonize::pythonize(py, &res)
        .map(|b| b.unbind())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn backtest_settings_from_py(
    py: Python,
    settings: Option<Py<PyAny>>,
) -> PyResult<BacktestSettings> {
    let mut out = BacktestSettings::default();
    let Some(settings) = settings else {
        return Ok(out);
    };
    let obj = settings.bind(py);
    if obj.is_none() {
        return Ok(out);
    }

    set_f64(obj, "sl_pips", &mut out.sl_pips)?;
    set_f64(obj, "stop_loss_pips", &mut out.sl_pips)?;
    set_f64(obj, "tp_pips", &mut out.tp_pips)?;
    set_f64(obj, "take_profit_pips", &mut out.tp_pips)?;
    set_usize(obj, "max_hold_bars", &mut out.max_hold_bars)?;
    set_bool(obj, "trailing_enabled", &mut out.trailing_enabled)?;
    set_f64(
        obj,
        "trailing_atr_multiplier",
        &mut out.trailing_atr_multiplier,
    )?;
    set_f64(obj, "trailing_be_trigger_r", &mut out.trailing_be_trigger_r)?;
    set_f64(obj, "pip_value", &mut out.pip_value)?;
    set_f64(obj, "spread_pips", &mut out.spread_pips)?;
    set_f64(obj, "commission_per_trade", &mut out.commission_per_trade)?;
    set_f64(obj, "commission_per_lot", &mut out.commission_per_trade)?;
    set_f64(obj, "pip_value_per_lot", &mut out.pip_value_per_lot)?;
    Ok(out)
}

fn get_setting_attr<'py>(
    obj: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        if let Some(value) = dict.get_item(key)? {
            return Ok(Some(value));
        }
    }
    match obj.getattr(key) {
        Ok(value) => Ok(Some(value)),
        Err(_) => Ok(None),
    }
}

fn set_f64(obj: &Bound<'_, PyAny>, key: &str, slot: &mut f64) -> PyResult<()> {
    if let Some(value) = get_setting_attr(obj, key)? {
        *slot = value.extract::<f64>()?;
    }
    Ok(())
}

fn set_usize(obj: &Bound<'_, PyAny>, key: &str, slot: &mut usize) -> PyResult<()> {
    if let Some(value) = get_setting_attr(obj, key)? {
        *slot = value.extract::<usize>()?;
    }
    Ok(())
}

fn set_bool(obj: &Bound<'_, PyAny>, key: &str, slot: &mut bool) -> PyResult<()> {
    if let Some(value) = get_setting_attr(obj, key)? {
        *slot = value.extract::<bool>()?;
    }
    Ok(())
}

#[pyclass(name = "CombinatorialPurgedCV")]
pub struct PyCombinatorialPurgedCV {
    inner: CombinatorialPurgedCV,
}

#[pymethods]
impl PyCombinatorialPurgedCV {
    #[new]
    #[pyo3(signature = (n_splits=5, n_test_groups=2, embargo_pct=0.01, purge_pct=0.02))]
    pub fn new(n_splits: usize, n_test_groups: usize, embargo_pct: f64, purge_pct: f64) -> Self {
        Self {
            inner: CombinatorialPurgedCV::new(n_splits, n_test_groups, embargo_pct, purge_pct),
        }
    }

    pub fn split(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        Ok(self.inner.split(n_samples))
    }
}
