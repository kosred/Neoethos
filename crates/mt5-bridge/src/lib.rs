use anyhow::{Result};
use pyo3::prelude::*;
use tracing::{info, error, warn};

fn format_last_error(err: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok((code, description)) = err.extract::<(i32, String)>() {
        return Ok(format!("code={} description={}", code, description));
    }

    if let Ok(description) = err.extract::<String>() {
        return Ok(description);
    }

    Ok(err.str()?.to_string_lossy().into_owned())
}

pub struct MT5Engine {
    connected: bool,
}

impl MT5Engine {
    pub fn new() -> Result<Self> {
        // Python::initialize() is the new way in 0.26+
        Python::initialize();
        Ok(Self { connected: false })
    }

    pub fn initialize(&mut self) -> Result<bool> {
        let res = Python::attach(|py| -> PyResult<bool> {
            let mt5 = match py.import("MetaTrader5") {
                Ok(m) => m,
                Err(e) => {
                    warn!("MetaTrader5 module not found: {}", e);
                    return Ok(false);
                }
            };
            
            let init_result: bool = mt5.getattr("initialize")?.call0()?.extract()?;
            
            if !init_result {
                let err_obj = mt5.getattr("last_error")?.call0()?;
                let err = format_last_error(err_obj.as_any())?;
                error!("MT5 Initialization failed. Last error: {}", err);
                return Ok(false);
            }

            info!("MT5 successfully initialized from Pure Rust.");
            Ok(true)
        }).map_err(|e| anyhow::anyhow!("PyError: {}", e))?;
        
        self.connected = res;
        Ok(res)
    }

    pub fn terminal_info(&self) -> Result<String> {
        if !self.connected { anyhow::bail!("MT5 is not connected."); }

        Python::attach(|py| -> PyResult<String> {
            let mt5 = py.import("MetaTrader5")?;
            let info = mt5.getattr("terminal_info")?.call0()?;
            if info.is_none() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err("Failed to get terminal info."));
            }
            Ok(info.to_string())
        }).map_err(|e| anyhow::anyhow!("PyError: {}", e))
    }

    pub fn shutdown(&mut self) {
        if self.connected {
            let _ = Python::attach(|py| -> PyResult<()> {
                if let Ok(mt5) = py.import("MetaTrader5") {
                    let _ = mt5.getattr("shutdown")?.call0();
                    info!("MT5 Connection Shutdown.");
                }
                Ok(())
            });
            self.connected = false;
        }
    }
}

impl Drop for MT5Engine {
    fn drop(&mut self) { self.shutdown(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    use std::ffi::CString;

    fn run_py(py: Python<'_>, code: &str, locals: &Bound<'_, PyDict>) -> PyResult<()> {
        let code = CString::new(code).expect("python code should not contain embedded nulls");
        py.run(code.as_c_str(), Some(locals), Some(locals))
    }

    #[test]
    fn test_format_last_error_handles_mt5_tuple_payload() -> Result<()> {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            run_py(py, r#"err = (-6, "Terminal: Authorization failed")"#, &locals)?;
            let err = locals
                .get_item("err")?
                .expect("err should be defined");

            let formatted = format_last_error(err.as_any())?;

            assert_eq!(
                formatted,
                "code=-6 description=Terminal: Authorization failed"
            );
            Ok(())
        })
    }

    #[test]
    fn test_format_last_error_falls_back_to_string_payload() -> Result<()> {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            run_py(py, r#"err = "module unavailable""#, &locals)?;
            let err = locals
                .get_item("err")?
                .expect("err should be defined");

            let formatted = format_last_error(err.as_any())?;

            assert_eq!(formatted, "module unavailable");
            Ok(())
        })
    }
}
