use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::Mutex;
use crate::utils::dataframe_from_named_ndarray;
#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
use crate::utils::params_from_py;
use polars::prelude::*;

#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
use forex_models::base::ExpertModel;
#[cfg(feature = "lightgbm")]
use forex_models::tree_models::LightGBMExpert;
#[cfg(feature = "xgboost")]
use forex_models::tree_models::{XGBoostDARTExpert, XGBoostExpert, XGBoostRFExpert};
#[cfg(feature = "catboost")]
use forex_models::tree_models::{CatBoostAltExpert, CatBoostExpert};

use forex_models::genetic::GeneticStrategyExpert;
use forex_models::neural_networks::MLPExpert as RustMlpExpert;

#[cfg(feature = "lightgbm")]
#[pyclass(unsendable)]
pub struct LightGBMModel {
    model: Mutex<LightGBMExpert>,
}

#[cfg(feature = "lightgbm")]
#[pymethods]
impl LightGBMModel {
    #[new]
    #[pyo3(signature = (params=None))]
    fn new(params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let rs_params = params_from_py(params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
        })?;
        let model = LightGBMExpert::new(rs_params);
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_vec: Vec<i32> = labels.as_array().iter().copied().collect();
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let labels_series = polars::prelude::Series::new("label".into(), labels_vec);
        model.fit(&df, &labels_series, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e))
        })?;
        Ok(())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let probs = model.predict_proba(&df, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Prediction failed: {}", e))
        })?;
        Ok(probs.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;
        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;
        Ok(())
    }
}

#[cfg(feature = "xgboost")]
#[pyclass(unsendable)]
pub struct XGBoostModel {
    model: Mutex<XGBoostExpert>,
}

#[cfg(feature = "xgboost")]
#[pymethods]
impl XGBoostModel {
    #[new]
    #[pyo3(signature = (params=None))]
    fn new(params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let rs_params = params_from_py(params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
        })?;
        let model = XGBoostExpert::new(rs_params);
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_vec: Vec<i32> = labels.as_array().iter().copied().collect();
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let labels_series = polars::prelude::Series::new("label".into(), labels_vec);
        model.fit(&df, &labels_series, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e))
        })?;
        Ok(())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let probs = model.predict_proba(&df, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Prediction failed: {}", e))
        })?;
        Ok(probs.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;
        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;
        Ok(())
    }
}

#[cfg(feature = "xgboost")]
#[pyclass(unsendable)]
pub struct XGBoostRFModel {
    model: Mutex<XGBoostRFExpert>,
}

#[cfg(feature = "xgboost")]
#[pymethods]
impl XGBoostRFModel {
    #[new]
    #[pyo3(signature = (params=None))]
    fn new(params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let rs_params = params_from_py(params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
        })?;
        let model = XGBoostRFExpert::new(rs_params);
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_vec: Vec<i32> = labels.as_array().iter().copied().collect();
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let labels_series = polars::prelude::Series::new("label".into(), labels_vec);
        model.fit(&df, &labels_series, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e))
        })?;
        Ok(())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let probs = model.predict_proba(&df, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Prediction failed: {}", e))
        })?;
        Ok(probs.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;
        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;
        Ok(())
    }
}

#[cfg(feature = "xgboost")]
#[pyclass(unsendable)]
pub struct XGBoostDARTModel {
    model: Mutex<XGBoostDARTExpert>,
}

#[cfg(feature = "xgboost")]
#[pymethods]
impl XGBoostDARTModel {
    #[new]
    #[pyo3(signature = (params=None))]
    fn new(params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let rs_params = params_from_py(params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
        })?;
        let model = XGBoostDARTExpert::new(rs_params);
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_vec: Vec<i32> = labels.as_array().iter().copied().collect();
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let labels_series = polars::prelude::Series::new("label".into(), labels_vec);
        model.fit(&df, &labels_series, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e))
        })?;
        Ok(())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let probs = model.predict_proba(&df, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Prediction failed: {}", e))
        })?;
        Ok(probs.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;
        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;
        Ok(())
    }
}

#[cfg(feature = "catboost")]
#[pyclass(unsendable)]
pub struct CatBoostModel {
    model: Mutex<CatBoostExpert>,
}

#[cfg(feature = "catboost")]
#[pymethods]
impl CatBoostModel {
    #[new]
    #[pyo3(signature = (params=None))]
    fn new(params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let rs_params = params_from_py(params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
        })?;
        let model = CatBoostExpert::new(rs_params);
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_vec: Vec<i32> = labels.as_array().iter().copied().collect();
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let labels_series = polars::prelude::Series::new("label".into(), labels_vec);
        model.fit(&df, &labels_series, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e))
        })?;
        Ok(())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let probs = model.predict_proba(&df, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Prediction failed: {}", e))
        })?;
        Ok(probs.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;
        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;
        Ok(())
    }
}

#[cfg(feature = "catboost")]
#[pyclass(unsendable)]
pub struct CatBoostAltModel {
    model: Mutex<CatBoostAltExpert>,
}

#[cfg(feature = "catboost")]
#[pymethods]
impl CatBoostAltModel {
    #[new]
    #[pyo3(signature = (params=None))]
    fn new(params: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let rs_params = params_from_py(params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
        })?;
        let model = CatBoostAltExpert::new(rs_params);
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_vec: Vec<i32> = labels.as_array().iter().copied().collect();
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let labels_series = polars::prelude::Series::new("label".into(), labels_vec);
        model.fit(&df, &labels_series, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e))
        })?;
        Ok(())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let probs = model.predict_proba(&df, None, None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Prediction failed: {}", e))
        })?;
        Ok(probs.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;
        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;
        Ok(())
    }
}

#[pyclass(unsendable)]
pub struct GeneticModel {
    model: Mutex<GeneticStrategyExpert>,
}

#[pymethods]
impl GeneticModel {
    #[new]
    #[pyo3(signature = (idx=1, population_size=50, generations=10, max_indicators=0))]
    fn new(
        idx: usize,
        population_size: usize,
        generations: usize,
        max_indicators: usize,
    ) -> PyResult<Self> {
        let _ = idx;
        let model = GeneticStrategyExpert::new(population_size, generations, max_indicators)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Genetic init failed: {}",
                    e
                ))
            })?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    #[pyo3(signature = (
        features,
        labels,
        feature_names=None,
        metadata=None,
        metadata_columns=None,
        metadata_symbol=None,
    ))]
    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i32>,
        feature_names: Option<Vec<String>>,
        metadata: Option<PyReadonlyArray2<'py, f64>>,
        metadata_columns: Option<Vec<String>>,
        metadata_symbol: Option<String>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_vec: Vec<i32> = labels.as_array().iter().copied().collect();
        let metadata_array = metadata.map(|arr| arr.as_array().to_owned());

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())?;
            let labels_series = Series::new("label".into(), labels_vec);
            let metadata_df = match metadata_array.as_ref() {
                Some(arr) => Some(dataframe_from_named_ndarray(
                    arr,
                    metadata_columns.as_deref(),
                )?),
                None => None,
            };

            model
                .fit(
                    &df,
                    &labels_series,
                    metadata_df.as_ref(),
                    metadata_symbol.as_deref(),
                )
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        })();

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    #[pyo3(signature = (
        features,
        feature_names=None,
        metadata=None,
        metadata_columns=None,
        metadata_symbol=None,
    ))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        feature_names: Option<Vec<String>>,
        metadata: Option<PyReadonlyArray2<'py, f64>>,
        metadata_columns: Option<Vec<String>>,
        metadata_symbol: Option<String>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();
        let metadata_array = metadata.map(|arr| arr.as_array().to_owned());

        let result: Result<Array2<f32>, String> = (|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_named_ndarray(&features_array, feature_names.as_deref())?;
            let metadata_df = match metadata_array.as_ref() {
                Some(arr) => Some(dataframe_from_named_ndarray(
                    arr,
                    metadata_columns.as_deref(),
                )?),
                None => None,
            };

            model
                .predict_proba(&df, metadata_df.as_ref(), metadata_symbol.as_deref())
                .map_err(|e| format!("Prediction failed: {}", e))
        })();

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr: Array2<f32>| arr.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;

        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;

        Ok(())
    }
}

#[pyclass(unsendable)]
pub struct MLPModel {
    model: Mutex<RustMlpExpert>,
}

#[pymethods]
impl MLPModel {
    #[new]
    #[pyo3(signature = (
        idx=1,
        hidden_dim=256,
        n_layers=3,
        dropout=0.1,
        lr=1e-3,
        max_time_sec=36000,
        device="cpu",
        batch_size=4096,
    ))]
    fn new(
        idx: usize,
        hidden_dim: i64,
        n_layers: i64,
        dropout: f64,
        lr: f64,
        max_time_sec: u64,
        device: &str,
        batch_size: i64,
    ) -> PyResult<Self> {
        let _ = idx;
        let model = RustMlpExpert::new(
            hidden_dim,
            n_layers,
            dropout,
            lr,
            max_time_sec,
            device,
            batch_size,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("MLP init failed: {}", e))
        })?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    fn fit<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_vec: Vec<i32> = labels.as_array().iter().copied().collect();

        let result: Result<(), String> = (|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;
            model
                .fit(&features_array, &labels_vec)
                .map_err(|e| format!("Training failed: {}", e))?;
            Ok(())
        })();

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = (|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;
            model
                .predict_proba(&features_array)
                .map_err(|e| format!("Prediction failed: {}", e))
        })();

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr: Array2<f32>| arr.into_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;

        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(feature = "burn-backend")]
pub mod burn_bindings {
    use super::*;
    use forex_models::burn_models::*;

    macro_rules! burn_model_wrapper {
        (
            $py_name:ident,
            $config_type:ident,
            $model_type:ident,
            $default_hidden:expr,
            $default_layers:expr
        ) => {
            #[pyclass(unsendable, module = "forex_bindings")]
            pub struct $py_name {
                model: Option<$model_type<TrainBackend>>,
                input_dim: usize,
                hidden_dim: usize,
                n_classes: usize,
                lr: f64,
                batch_size: usize,
                max_epochs: usize,
                patience: usize,
            }

            #[pymethods]
            impl $py_name {
                #[new]
                #[pyo3(signature = (
                    input_dim=96,
                    hidden_dim=$default_hidden,
                    n_classes=3,
                    lr=1e-3,
                    batch_size=64,
                    max_epochs=100,
                    patience=8,
                ))]
                fn new(
                    input_dim: usize,
                    hidden_dim: usize,
                    n_classes: usize,
                    lr: f64,
                    batch_size: usize,
                    max_epochs: usize,
                    patience: usize,
                ) -> Self {
                    Self {
                        model: None,
                        input_dim,
                        hidden_dim,
                        n_classes,
                        lr,
                        batch_size,
                        max_epochs,
                        patience,
                    }
                }

                fn fit<'py>(
                    &mut self,
                    _py: Python<'py>,
                    features: PyReadonlyArray2<'py, f32>,
                    labels: PyReadonlyArray1<'py, i32>,
                ) -> PyResult<f64> {
                    let x = features.as_array().to_owned();
                    let y: Vec<i32> = labels.as_array().iter().copied().collect();

                    self.input_dim = x.ncols();

                    let device = <TrainBackend as burn::tensor::backend::Backend>::Device::default();
                    let config = $config_type::new(self.input_dim)
                        .with_hidden_dim(self.hidden_dim)
                        .with_n_classes(self.n_classes);
                    let model = config.init::<TrainBackend>(&device);

                    let train_config = TrainConfig {
                        lr: self.lr,
                        batch_size: self.batch_size,
                        max_epochs: self.max_epochs,
                        patience: self.patience,
                        n_classes: self.n_classes,
                    };

                    let (trained, best_loss) = train_model::<TrainBackend, _>(
                        model, &x, &y, &train_config,
                    );
                    self.model = Some(trained);
                    Ok(best_loss as f64)
                }

                fn predict_proba<'py>(
                    &self,
                    py: Python<'py>,
                    features: PyReadonlyArray2<'py, f32>,
                ) -> PyResult<Bound<'py, PyArray2<f32>>> {
                    let x = features.as_array().to_owned();
                    let model = self.model.as_ref().ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            "Model not trained yet. Call fit() first.",
                        )
                    })?;
                    let probs = predict_proba::<TrainBackend, _>(model, &x, self.batch_size);
                    Ok(probs.into_pyarray(py))
                }
            }
        };
    }

    burn_model_wrapper!(BurnMLPModel, BurnMLPConfig, BurnMLP, 256, 3);
    burn_model_wrapper!(BurnNBeatsModel, BurnNBeatsConfig, BurnNBeats, 64, 3);
    burn_model_wrapper!(BurnTiDEModel, BurnTiDEConfig, BurnTiDE, 128, 2);
    burn_model_wrapper!(BurnKANModel, BurnKANConfig, BurnKAN, 32, 2);
    burn_model_wrapper!(BurnTransformerModel, BurnTransformerConfig, BurnTransformer, 128, 4);

    #[pyclass(unsendable, name = "BurnTabNetModel", module = "forex_bindings")]
    pub struct BurnTabNetModel {
        model: Option<BurnTabNet<TrainBackend>>,
        input_dim: usize,
        hidden_dim: usize,
        n_classes: usize,
        lr: f64,
        batch_size: usize,
        max_epochs: usize,
        patience: usize,
    }

    #[pymethods]
    impl BurnTabNetModel {
        #[new]
        #[pyo3(signature = (input_dim=96, hidden_dim=64, n_classes=3, lr=2e-3, batch_size=64, max_epochs=100, patience=8))]
        fn new(
            input_dim: usize, hidden_dim: usize, n_classes: usize,
            lr: f64, batch_size: usize, max_epochs: usize, patience: usize,
        ) -> Self {
            Self { model: None, input_dim, hidden_dim, n_classes, lr, batch_size, max_epochs, patience }
        }

        fn fit<'py>(
            &mut self, _py: Python<'py>,
            features: PyReadonlyArray2<'py, f32>,
            labels: PyReadonlyArray1<'py, i32>,
        ) -> PyResult<f64> {
            let x = features.as_array().to_owned();
            let y: Vec<i32> = labels.as_array().iter().copied().collect();
            self.input_dim = x.ncols();

            let device = <TrainBackend as burn::tensor::backend::Backend>::Device::default();
            let config = BurnTabNetConfig::new(self.input_dim)
                .with_hidden_dim(self.hidden_dim)
                .with_n_classes(self.n_classes);
            let model = config.init::<TrainBackend>(&device);

            let train_config = TrainConfig {
                lr: self.lr, batch_size: self.batch_size,
                max_epochs: self.max_epochs, patience: self.patience, n_classes: self.n_classes,
            };
            let (trained, best_loss) = train_model::<TrainBackend, _>(model, &x, &y, &train_config);
            self.model = Some(trained);
            Ok(best_loss as f64)
        }

        fn predict_proba<'py>(
            &self, py: Python<'py>,
            features: PyReadonlyArray2<'py, f32>,
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let x = features.as_array().to_owned();
            let model = self.model.as_ref().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not trained. Call fit() first.")
            })?;
            let probs = predict_proba::<TrainBackend, _>(model, &x, self.batch_size);
            Ok(probs.into_pyarray(py))
        }
    }

    pub fn register_burn_models(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<BurnMLPModel>()?;
        m.add_class::<BurnNBeatsModel>()?;
        m.add_class::<BurnTiDEModel>()?;
        m.add_class::<BurnTabNetModel>()?;
        m.add_class::<BurnKANModel>()?;
        m.add_class::<BurnTransformerModel>()?;
        Ok(())
    }
}
