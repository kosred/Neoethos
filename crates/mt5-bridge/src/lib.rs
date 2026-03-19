use anyhow::Result;
use pyo3::prelude::*;
use tracing::{info, error};

pub struct MT5Engine {
    connected: bool,
}

impl MT5Engine {
    pub fn new() -> Result<Self> {
        
        Ok(Self { connected: false })
    }

    pub fn initialize(&mut self) -> Result<bool> {
        let res = Python::attach(|py| {
            let mt5 = py.import("MetaTrader5").unwrap();
            let init_result: bool = mt5.getattr("initialize").unwrap().call0().unwrap().extract().unwrap();
            
            if !init_result {
                return false;
            }

            info!("MT5 successfully initialized from Pure Rust.");
            true
        });
        
        self.connected = res;
        Ok(res)
    }

    pub fn terminal_info(&self) -> Result<String> {
        if !self.connected { anyhow::bail!("MT5 is not connected."); }

        let info_str = Python::attach(|py| {
            let mt5 = py.import("MetaTrader5").unwrap();
            let info = mt5.getattr("terminal_info").unwrap().call0().unwrap();
            info.to_string()
        });
        Ok(info_str)
    }

    pub fn shutdown(&mut self) {
        if self.connected {
            Python::attach(|py| {
                if let Ok(mt5) = py.import("MetaTrader5") {
                    let _res = mt5.getattr("shutdown").unwrap().call0().unwrap();
                    info!("MT5 Connection Shutdown.");
                }
            });
            self.connected = false;
        }
    }
}

impl Drop for MT5Engine {
    fn drop(&mut self) { self.shutdown(); }
}
