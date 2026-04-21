use crate::utils::{dataframe_from_ndarray, params_from_py};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use polars::prelude::{NamedFrom, Series};
use pyo3::prelude::*;
use std::sync::Mutex;

use forex_models::MLPExpert;
use forex_models::NeatExpert;
use forex_models::anomaly::IsolationForestExpert;
use forex_models::base::ExpertModel;
use forex_models::forecasting::SwarmForecaster;
use forex_models::genetic::GeneticStrategyExpert;
use forex_models::statistical::{BayesianLogitExpert, ElasticNetExpert};
use forex_models::tree_models::XGBoostExpert;

fn fit_expert_model<M: ExpertModel>(
    inner: &Mutex<M>,
    features: PyReadonlyArray2<'_, f64>,
    labels: PyReadonlyArray1<'_, i32>,
) -> PyResult<()> {
    let x = features.as_array().to_owned();
    let y = labels.as_array().iter().copied().collect::<Vec<_>>();
    if x.nrows() != y.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "feature rows ({}) must match label rows ({})",
            x.nrows(),
            y.len()
        )));
    }
    let df = dataframe_from_ndarray(&x).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let labels = Series::new("label".into(), y);
    let mut guard = inner
        .lock()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("model mutex poisoned"))?;
    guard
        .fit(&df, &labels)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn predict_expert_model<'py, M: ExpertModel>(
    py: Python<'py>,
    inner: &Mutex<M>,
    features: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x = features.as_array().to_owned();
    let df = dataframe_from_ndarray(&x).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let guard = inner
        .lock()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("model mutex poisoned"))?;
    let probabilities = guard
        .predict_proba(&df)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(probabilities.into_pyarray(py))
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct XGBoostModel {
    pub inner: Mutex<XGBoostExpert>,
}
#[pymethods]
impl XGBoostModel {
    #[new]
    #[pyo3(signature = (params=None))]
    fn new(params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let rs_params =
            params_from_py(params).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self {
            inner: Mutex::new(XGBoostExpert::new(0, rs_params)),
        })
    }

    fn fit(
        &self,
        features: PyReadonlyArray2<'_, f64>,
        labels: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<()> {
        fit_expert_model(&self.inner, features, labels)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        predict_expert_model(py, &self.inner, features)
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct ElasticNetModel {
    pub inner: Mutex<ElasticNetExpert>,
}
#[pymethods]
impl ElasticNetModel {
    #[new]
    #[pyo3(signature = (alpha=1.0, l1_ratio=0.5))]
    fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self {
            inner: Mutex::new(ElasticNetExpert::new(alpha, l1_ratio)),
        }
    }

    fn fit(
        &self,
        features: PyReadonlyArray2<'_, f64>,
        labels: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<()> {
        fit_expert_model(&self.inner, features, labels)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        predict_expert_model(py, &self.inner, features)
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct BayesianLogitModel {
    pub inner: Mutex<BayesianLogitExpert>,
}
#[pymethods]
impl BayesianLogitModel {
    #[new]
    fn new() -> Self {
        Self {
            inner: Mutex::new(BayesianLogitExpert::new()),
        }
    }

    fn fit(
        &self,
        features: PyReadonlyArray2<'_, f64>,
        labels: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<()> {
        fit_expert_model(&self.inner, features, labels)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        predict_expert_model(py, &self.inner, features)
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct IsolationForestModel {
    pub inner: Mutex<IsolationForestExpert>,
}
#[pymethods]
impl IsolationForestModel {
    #[new]
    #[pyo3(signature = (n_trees=100, sample_size=256))]
    fn new(n_trees: usize, sample_size: usize) -> Self {
        Self {
            inner: Mutex::new(IsolationForestExpert::new(n_trees, sample_size)),
        }
    }

    fn fit(
        &self,
        features: PyReadonlyArray2<'_, f64>,
        labels: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<()> {
        fit_expert_model(&self.inner, features, labels)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        predict_expert_model(py, &self.inner, features)
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct SwarmForecasterModel {
    pub inner: Mutex<SwarmForecaster>,
}
#[pymethods]
impl SwarmForecasterModel {
    #[new]
    fn new(mem_limit: f64) -> Self {
        Self {
            inner: Mutex::new(SwarmForecaster::new(mem_limit)),
        }
    }

    fn fit(
        &self,
        features: PyReadonlyArray2<'_, f64>,
        labels: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<()> {
        fit_expert_model(&self.inner, features, labels)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        predict_expert_model(py, &self.inner, features)
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct MLPModel {
    pub inner: Mutex<MLPExpert>,
}
#[pymethods]
impl MLPModel {
    #[new]
    fn new() -> Self {
        Self {
            inner: Mutex::new(MLPExpert::default()),
        }
    }

    fn fit(
        &self,
        features: PyReadonlyArray2<'_, f64>,
        labels: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<()> {
        fit_expert_model(&self.inner, features, labels)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        predict_expert_model(py, &self.inner, features)
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct GeneticModel {
    pub inner: Mutex<GeneticStrategyExpert>,
}
#[pymethods]
impl GeneticModel {
    #[new]
    #[pyo3(signature = (pop=50, gens=10, max_ind=8))]
    fn new(pop: usize, gens: usize, max_ind: usize) -> PyResult<Self> {
        let inner = GeneticStrategyExpert::new(pop, gens, max_ind)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    fn fit(
        &self,
        features: PyReadonlyArray2<'_, f64>,
        labels: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<()> {
        fit_expert_model(&self.inner, features, labels)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        predict_expert_model(py, &self.inner, features)
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct NeatModel {
    pub inner: Mutex<NeatExpert>,
}
#[pymethods]
impl NeatModel {
    #[new]
    #[pyo3(signature = (input_dim=1, population=96, generations=48))]
    fn new(input_dim: usize, population: usize, generations: usize) -> Self {
        Self {
            inner: Mutex::new(NeatExpert::with_config(input_dim, population, generations)),
        }
    }

    fn fit(
        &self,
        features: PyReadonlyArray2<'_, f64>,
        labels: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<()> {
        fit_expert_model(&self.inner, features, labels)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        predict_expert_model(py, &self.inner, features)
    }
}

pub fn register_models(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<XGBoostModel>()?;
    m.add_class::<ElasticNetModel>()?;
    m.add_class::<BayesianLogitModel>()?;
    m.add_class::<IsolationForestModel>()?;
    m.add_class::<SwarmForecasterModel>()?;
    m.add_class::<MLPModel>()?;
    m.add_class::<GeneticModel>()?;
    m.add_class::<NeatModel>()?;
    Ok(())
}
