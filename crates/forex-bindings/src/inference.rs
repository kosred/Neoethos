#[cfg(feature = "onnx")]
use forex_models::ONNXInferenceEngine;
#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use numpy::{PyArray2, PyReadonlyArray2};
#[cfg(feature = "onnx")]
use pyo3::prelude::*;
#[cfg(feature = "onnx")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "onnx")]
#[pyclass]
pub struct ModelEngine {
    pub engine: Arc<Mutex<ONNXInferenceEngine>>,
}

#[cfg(feature = "onnx")]
#[pymethods]
impl ModelEngine {
    #[new]
    pub fn new() -> PyResult<Self> {
        let engine = ONNXInferenceEngine::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to init engine: {}",
                e
            ))
        })?;
        Ok(ModelEngine {
            engine: Arc::new(Mutex::new(engine)),
        })
    }

    pub fn load_models(&self, py: Python, path: &str) -> PyResult<()> {
        let path = path.to_string();
        let result: Result<(), String> = py.detach(|| {
            let mut engine = self
                .engine
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;
            engine
                .load_models(&path)
                .map_err(|e| format!("Failed to load models: {}", e))?;
            Ok(())
        });
        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    pub fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        model_name: &str,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();
        let model_name = model_name.to_string();
        let prediction: Array2<f32> = py
            .detach(|| {
                let engine = self
                    .engine
                    .lock()
                    .map_err(|e| format!("Lock poisoned: {}", e))?;
                engine
                    .predict_proba(&model_name, &features_array)
                    .map_err(|e| format!("Prediction failed: {}", e))
            })
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))?;

        use numpy::IntoPyArray;
        Ok(prediction.into_pyarray(py))
    }
}
