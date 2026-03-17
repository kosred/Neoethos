use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::Mutex;
use crate::utils::{dataframe_from_named_ndarray, params_from_py};
use polars::prelude::*;

use forex_models::tree_models::{XGBoostExpert};
use forex_search::Gene;
use forex_models::genetic::GeneticStrategyExpert;
use forex_models::forecasting::SwarmForecaster;
use forex_models::statistical::{ElasticNetExpert, BayesianLogitExpert};
use forex_models::anomaly::IsolationForestExpert;

#[pyclass(unsendable, module = "forex_bindings")]
pub struct XGBoostModel { pub inner: Mutex<XGBoostExpert> }
#[pymethods]
impl XGBoostModel {
    #[new]
    #[pyo3(signature = (params=None))]
    fn new(params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let rs_params = params_from_py(params).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(Self { inner: Mutex::new(XGBoostExpert::new(0, rs_params)) })
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct ElasticNetModel { pub inner: Mutex<ElasticNetExpert> }
#[pymethods]
impl ElasticNetModel {
    #[new]
    #[pyo3(signature = (alpha=1.0, l1_ratio=0.5))]
    fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self { inner: Mutex::new(ElasticNetExpert::new(alpha, l1_ratio)) }
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct BayesianLogitModel { pub inner: Mutex<BayesianLogitExpert> }
#[pymethods]
impl BayesianLogitModel {
    #[new]
    fn new() -> Self {
        Self { inner: Mutex::new(BayesianLogitExpert::new()) }
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct IsolationForestModel { pub inner: Mutex<IsolationForestExpert> }
#[pymethods]
impl IsolationForestModel {
    #[new]
    #[pyo3(signature = (n_trees=100, sample_size=256))]
    fn new(n_trees: usize, sample_size: usize) -> Self {
        Self { inner: Mutex::new(IsolationForestExpert::new(n_trees, sample_size)) }
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct SwarmForecasterModel { pub inner: Mutex<SwarmForecaster> }
#[pymethods]
impl SwarmForecasterModel {
    #[new]
    fn new(mem_limit: f64) -> Self {
        Self { inner: Mutex::new(SwarmForecaster::new(mem_limit)) }
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct MLPModel { pub inner: Mutex<Gene> } // Placeholder for MLP using Gene
#[pymethods]
impl MLPModel {
    #[new]
    fn new() -> Self {
        Self { inner: Mutex::new(Gene::default()) }
    }
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct GeneticModel { pub inner: Mutex<GeneticStrategyExpert> }
#[pymethods]
impl GeneticModel {
    #[new]
    #[pyo3(signature = (pop=50, gens=10, max_ind=8))]
    fn new(pop: usize, gens: usize, max_ind: usize) -> Self {
        Self { inner: Mutex::new(GeneticStrategyExpert::new(pop, gens, max_ind).unwrap()) }
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
    Ok(())
}
