use crate::utils::params_from_py;
use pyo3::prelude::*;
use std::sync::Mutex;

use forex_models::MLPExpert;
use forex_models::NeatExpert;
use forex_models::anomaly::IsolationForestExpert;
use forex_models::forecasting::SwarmForecaster;
use forex_models::genetic::GeneticStrategyExpert;
use forex_models::statistical::{BayesianLogitExpert, ElasticNetExpert};
use forex_models::tree_models::XGBoostExpert;

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
}

#[pyclass(unsendable, module = "forex_bindings")]
pub struct GeneticModel {
    pub inner: Mutex<GeneticStrategyExpert>,
}
#[pymethods]
impl GeneticModel {
    #[new]
    #[pyo3(signature = (pop=50, gens=10, max_ind=8))]
    fn new(pop: usize, gens: usize, max_ind: usize) -> Self {
        Self {
            inner: Mutex::new(GeneticStrategyExpert::new(pop, gens, max_ind).unwrap()),
        }
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
