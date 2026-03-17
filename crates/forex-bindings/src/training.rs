use pyo3::prelude::*;
use std::path::PathBuf;
use forex_models::TrainingOrchestrator as CoreOrchestrator;
use anyhow;

#[pyclass(name = "TrainingOrchestrator", module = "forex_bindings")]
pub struct TrainingOrchestrator {
    inner: CoreOrchestrator,
}

#[pymethods]
impl TrainingOrchestrator {
    #[new]
    pub fn new(config_path: &str, models_dir: &str) -> PyResult<Self> {
        // Use Settings::from_yaml
        let settings = forex_core::Settings::from_yaml(config_path)
            .map_err(|e: anyhow::Error| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        Ok(Self {
            inner: CoreOrchestrator::new(settings, PathBuf::from(models_dir)),
        })
    }

    pub fn train_symbol(&self, symbol: &str, base_tf: &str) -> PyResult<()> {
        self.inner.train_symbol(symbol, base_tf)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}
