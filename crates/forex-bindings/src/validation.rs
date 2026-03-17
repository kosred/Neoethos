use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::utils::{vec_from_py_f64, vec_from_py_i8, vec_from_py_i64};
use forex_search::validation::{embargoed_walkforward_backtest, CombinatorialPurgedCV};
use forex_search::eval::BacktestSettings;

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
    settings: Option<Py<PyAny>>, // Changed from _settings to match macro
    max_daily_loss_pct: f64,
    max_daily_profit_pct: f64,
    min_trading_days: usize,
    max_trades_per_day: usize,
) -> PyResult<Py<PyAny>> {
    let _ = settings;
    let cl = vec_from_py_f64(&close);
    let hi = vec_from_py_f64(&high);
    let lo = vec_from_py_f64(&low);
    let sig = vec_from_py_i8(&signals);
    let m = vec_from_py_i64(&months);
    let d = vec_from_py_i64(&days);

    let b_settings = BacktestSettings::default();

    let res = embargoed_walkforward_backtest(
        &cl, &hi, &lo, &sig, &m, &d,
        train_ratio, n_splits, embargo_bars,
        &b_settings,
        max_daily_loss_pct, max_daily_profit_pct,
        min_trading_days, max_trades_per_day
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    pythonize::pythonize(py, &res).map(|b| b.unbind()).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
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
