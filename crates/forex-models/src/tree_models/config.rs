use std::collections::HashMap;
use std::env;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevicePreference {
    Auto,
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamValue {
    Int(i32),
    Float(f64),
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone)]
pub struct TreeModelConfig {
    pub idx: usize,
    pub params: HashMap<String, ParamValue>,
    pub device_pref: DevicePreference,
    pub gpu_only: bool,
    pub cpu_threads: Option<usize>,
}

pub fn cpu_threads_hint() -> usize {
    fn read_threads_env(keys: &[&str]) -> Option<usize> {
        for key in keys {
            if let Ok(val) = env::var(key) {
                if let Ok(parsed) = val.trim().parse::<usize>() {
                    if parsed > 0 { return Some(parsed); }
                }
            }
        }
        None
    }
    read_threads_env(&["FOREX_BOT_RUST_THREADS", "FOREX_BOT_CPU_THREADS", "FOREX_BOT_CPU_BUDGET", "RAYON_NUM_THREADS"])
        .unwrap_or_else(|| num_cpus::get().saturating_sub(1).max(1))
}

pub fn tree_device_preference() -> DevicePreference {
    let raw = env::var("FOREX_BOT_TREE_DEVICE").unwrap_or_else(|_| "auto".to_string()).trim().to_lowercase();
    match raw.as_str() {
        "cpu" => DevicePreference::Cpu,
        "gpu" => DevicePreference::Gpu,
        "auto" => DevicePreference::Auto,
        "0" | "false" | "no" | "off" => DevicePreference::Cpu,
        "1" | "true" | "yes" | "on" => DevicePreference::Gpu,
        _ => DevicePreference::Auto,
    }
}

pub fn gpu_only_mode() -> bool {
    env::var("FOREX_BOT_GPU_ONLY").ok().map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on")).unwrap_or(false)
}

pub fn gpu_count() -> usize {
    #[cfg(feature = "tch")]
    { if tch::Cuda::is_available() { tch::Cuda::device_count() as usize } else { 0 } }
    #[cfg(not(feature = "tch"))]
    {
        match env::var("CUDA_VISIBLE_DEVICES") {
            Ok(devices) => {
                let trimmed = devices.trim();
                if trimmed.is_empty() || trimmed == "-1" { 0 } else { trimmed.split(',').count() }
            }
            Err(_) => 0,
        }
    }
}

pub fn get_early_stop_params(default_patience: usize, default_min_delta: f64) -> (usize, f64) {
    let p = env::var("FOREX_BOT_EARLY_STOP_PATIENCE").ok().and_then(|v| v.parse().ok()).unwrap_or(default_patience);
    let d = env::var("FOREX_BOT_EARLY_STOP_MIN_DELTA").ok().and_then(|v| v.parse().ok()).unwrap_or(default_min_delta);
    (p, d)
}

pub fn param_int(params: &HashMap<String, ParamValue>, key: &str, default: i32) -> i32 {
    match params.get(key) {
        Some(ParamValue::Int(v)) => *v,
        Some(ParamValue::Float(v)) => *v as i32,
        _ => default,
    }
}
