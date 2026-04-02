use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevicePreference {
    Auto,
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
                    if parsed > 0 {
                        return Some(parsed);
                    }
                }
            }
        }
        None
    }
    read_threads_env(&[
        "FOREX_BOT_RUST_THREADS",
        "FOREX_BOT_CPU_THREADS",
        "FOREX_BOT_CPU_BUDGET",
        "RAYON_NUM_THREADS",
    ])
    .unwrap_or_else(|| num_cpus::get().saturating_sub(1).max(1))
}

pub fn tree_device_preference() -> DevicePreference {
    tree_device_preference_for("tree")
}

pub fn tree_device_preference_for(model_name: &str) -> DevicePreference {
    let model_key = format!(
        "FOREX_BOT_{}_DEVICE",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    let raw = env::var(&model_key)
        .or_else(|_| env::var("FOREX_BOT_TREE_DEVICE"))
        .unwrap_or_else(|_| "auto".to_string())
        .trim()
        .to_lowercase();
    match raw.as_str() {
        "cpu" => DevicePreference::Cpu,
        "gpu" | "cuda" => DevicePreference::Gpu,
        "auto" => DevicePreference::Auto,
        "0" | "false" | "no" | "off" => DevicePreference::Cpu,
        "1" | "true" | "yes" | "on" => DevicePreference::Gpu,
        _ => DevicePreference::Auto,
    }
}

pub fn parse_device_preference(value: &str) -> DevicePreference {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => DevicePreference::Cpu,
        "gpu" | "cuda" => DevicePreference::Gpu,
        "auto" => DevicePreference::Auto,
        "0" | "false" | "no" | "off" => DevicePreference::Cpu,
        "1" | "true" | "yes" | "on" => DevicePreference::Gpu,
        _ => DevicePreference::Auto,
    }
}

pub fn gpu_only_mode() -> bool {
    gpu_only_mode_for("tree")
}

pub fn gpu_only_mode_for(model_name: &str) -> bool {
    let model_key = format!(
        "FOREX_BOT_{}_GPU_ONLY",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    env::var(&model_key)
        .or_else(|_| env::var("FOREX_BOT_GPU_ONLY"))
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub fn cpu_threads_hint_for(model_name: &str) -> usize {
    let model_key = format!(
        "FOREX_BOT_{}_THREADS",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    if let Ok(value) = env::var(&model_key) {
        if let Ok(parsed) = value.trim().parse::<usize>() {
            if parsed > 0 {
                return parsed;
            }
        }
    }
    cpu_threads_hint()
}

pub fn gpu_count() -> usize {
    #[cfg(feature = "tch")]
    {
        if tch::Cuda::is_available() {
            let detected = tch::Cuda::device_count() as usize;
            if detected > 0 {
                return detected;
            }
        }
    }

    fn env_gpu_count(keys: &[&str]) -> Option<usize> {
        for key in keys {
            let Ok(devices) = env::var(key) else {
                continue;
            };
            let trimmed = devices.trim();
            if trimmed.is_empty() || trimmed == "-1" || trimmed.eq_ignore_ascii_case("void") {
                return Some(0);
            }
            let count = trimmed
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty() && *value != "-1")
                .count();
            if count > 0 {
                return Some(count);
            }
        }
        None
    }

    fn nvidia_smi_gpu_count() -> Option<usize> {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8(output.stdout).ok()?;
        let count = stdout
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .count();
        Some(count)
    }

    if let Some(count) = env_gpu_count(&["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"]) {
        return count;
    }

    if let Some(count) = nvidia_smi_gpu_count() {
        return count;
    }

    0
}

pub fn get_early_stop_params(default_patience: usize, default_min_delta: f64) -> (usize, f64) {
    let p = env::var("FOREX_BOT_EARLY_STOP_PATIENCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_patience);
    let d = env::var("FOREX_BOT_EARLY_STOP_MIN_DELTA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_min_delta);
    (p, d)
}

pub fn param_int(params: &HashMap<String, ParamValue>, key: &str, default: i32) -> i32 {
    match params.get(key) {
        Some(ParamValue::Int(v)) => *v,
        Some(ParamValue::Float(v)) => *v as i32,
        _ => default,
    }
}

pub fn param_float(params: &HashMap<String, ParamValue>, key: &str, default: f64) -> f64 {
    match params.get(key) {
        Some(ParamValue::Float(v)) => *v,
        Some(ParamValue::Int(v)) => *v as f64,
        _ => default,
    }
}

pub fn param_bool(params: &HashMap<String, ParamValue>, key: &str, default: bool) -> bool {
    match params.get(key) {
        Some(ParamValue::Bool(v)) => *v,
        _ => default,
    }
}

pub fn param_string(params: &HashMap<String, ParamValue>, key: &str, default: &str) -> String {
    match params.get(key) {
        Some(ParamValue::String(v)) => v.clone(),
        _ => default.to_string(),
    }
}

pub fn device_preference_from_params(
    params: &HashMap<String, ParamValue>,
    default: DevicePreference,
) -> DevicePreference {
    for key in ["device", "device_preference", "device_pref"] {
        if let Some(ParamValue::String(value)) = params.get(key) {
            return parse_device_preference(value);
        }
    }
    default
}

pub fn gpu_only_from_params(params: &HashMap<String, ParamValue>, default: bool) -> bool {
    for key in ["gpu_only", "require_gpu"] {
        if let Some(ParamValue::Bool(value)) = params.get(key) {
            return *value;
        }
        if let Some(ParamValue::String(value)) = params.get(key) {
            return matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }
    }
    default
}

pub fn cpu_threads_from_params(params: &HashMap<String, ParamValue>, default: usize) -> usize {
    for key in ["cpu_threads", "threads", "num_threads"] {
        if let Some(ParamValue::Int(value)) = params.get(key) {
            if *value > 0 {
                return *value as usize;
            }
        }
        if let Some(ParamValue::String(value)) = params.get(key) {
            if let Ok(parsed) = value.trim().parse::<usize>() {
                if parsed > 0 {
                    return parsed;
                }
            }
        }
    }
    default
}
