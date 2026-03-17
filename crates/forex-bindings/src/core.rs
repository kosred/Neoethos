use forex_core::system::HardwareProbe;
use pyo3::prelude::*;
use pythonize::pythonize;
use std::sync::Mutex;

#[pyclass(module = "forex_bindings")]
pub struct ForexCore {
    pub probe: Mutex<HardwareProbe>,
}

#[pymethods]
impl ForexCore {
    #[new]
    pub fn new() -> Self {
        ForexCore {
            probe: Mutex::new(HardwareProbe::new()),
        }
    }

    pub fn detect_hardware(&self, py: Python) -> PyResult<Py<PyAny>> {
        let mut probe = self.probe.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let profile = probe.detect();
        let py_profile: Py<PyAny> = pythonize(py, &profile)?.clone().unbind();
        Ok(py_profile)
    }

    pub fn get_available_gpus(&self) -> Vec<String> {
        let mut probe = self.probe.lock().unwrap();
        let profile = probe.detect();
        (0..profile.num_gpus).map(|i| format!("cuda:{}", i)).collect()
    }
}
