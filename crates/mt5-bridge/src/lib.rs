use anyhow::{Result};
use pyo3::prelude::*;
use tracing::{info, error, warn};

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
                let err: String = mt5.getattr("last_error")?.call0()?.extract()?;
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
