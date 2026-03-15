use forex_core::system::HardwareProbe;
use pyo3::prelude::*;
use pythonize::pythonize;
use std::sync::Mutex;

#[pyclass]
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
        let py_profile: Py<PyAny> = pythonize(py, &profile)?.into();
        Ok(py_profile)
    }
}
