use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::process::Command;
use std::sync::OnceLock;

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

/// Process-wide tree-model runtime config, installed once from the operator's
/// `Settings` at startup via [`install_tree_runtime_from_settings`]. The
/// early-stop reader ([`get_early_stop_params`]) consults this instead of the
/// old `NEOETHOS_BOT_EARLY_STOP_*` env vars (v0.4.36 config-consolidation).
#[derive(Debug, Clone, PartialEq)]
pub struct TreeRuntimeOverrides {
    pub early_stop_patience: Option<usize>,
    pub early_stop_min_delta: Option<f64>,
}

impl Default for TreeRuntimeOverrides {
    fn default() -> Self {
        Self {
            early_stop_patience: None,
            early_stop_min_delta: None,
        }
    }
}

impl TreeRuntimeOverrides {
    /// Build from the operator's config (was the `NEOETHOS_BOT_EARLY_STOP_*`
    /// env vars). A `tree_runtime_from_settings_default_matches_default` test
    /// guarantees a fresh `Settings` reproduces [`Self::default`].
    pub fn from_settings(s: &neoethos_core::Settings) -> Self {
        let c = &s.models.tree_runtime;
        Self {
            early_stop_patience: c.early_stop_patience,
            early_stop_min_delta: c.early_stop_min_delta,
        }
    }
}

static TREE_RUNTIME: OnceLock<TreeRuntimeOverrides> = OnceLock::new();

/// Install the tree-model runtime config from `Settings` (call once at
/// startup, before any model training). The first install wins.
pub fn install_tree_runtime_from_settings(s: &neoethos_core::Settings) {
    let _ = TREE_RUNTIME.set(TreeRuntimeOverrides::from_settings(s));
}

/// Current tree-model runtime config (defaults if never installed — e.g. in
/// unit tests — preserving the historical env-absent behavior).
pub fn current_tree_runtime() -> &'static TreeRuntimeOverrides {
    TREE_RUNTIME.get_or_init(TreeRuntimeOverrides::default)
}

#[cfg(test)]
mod tree_runtime_tests {
    use super::*;

    #[test]
    fn tree_runtime_from_settings_default_matches_default() {
        let s = neoethos_core::Settings::default();
        assert_eq!(
            TreeRuntimeOverrides::from_settings(&s),
            TreeRuntimeOverrides::default()
        );
    }
}

pub fn cpu_threads_hint() -> usize {
    fn read_threads_env(keys: &[&str]) -> Option<usize> {
        for key in keys {
            if let Ok(val) = env::var(key)
                && let Ok(parsed) = val.trim().parse::<usize>()
                && parsed > 0
            {
                return Some(parsed);
            }
        }
        None
    }
    read_threads_env(&[
        "NEOETHOS_BOT_RUST_THREADS",
        "NEOETHOS_BOT_CPU_THREADS",
        "NEOETHOS_BOT_CPU_BUDGET",
        "RAYON_NUM_THREADS",
    ])
    .unwrap_or_else(|| num_cpus::get().saturating_sub(1).max(1))
}

pub fn tree_device_preference() -> DevicePreference {
    tree_device_preference_for("tree")
}

pub fn tree_device_preference_for(model_name: &str) -> DevicePreference {
    let model_key = format!(
        "NEOETHOS_BOT_{}_DEVICE",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    let raw = env::var(&model_key)
        .or_else(|_| env::var("NEOETHOS_BOT_TREE_DEVICE"))
        .unwrap_or_else(|_| "auto".to_string())
        .trim()
        .to_lowercase();
    match raw.as_str() {
        "cpu" => DevicePreference::Cpu,
        value if value == "gpu" || value == "cuda" || value.starts_with("cuda:") => {
            DevicePreference::Gpu
        }
        "auto" => DevicePreference::Auto,
        "0" | "false" | "no" | "off" => DevicePreference::Cpu,
        "1" | "true" | "yes" | "on" => DevicePreference::Gpu,
        _ => DevicePreference::Auto,
    }
}

pub fn parse_device_preference(value: &str) -> DevicePreference {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => DevicePreference::Cpu,
        value if value == "gpu" || value == "cuda" || value.starts_with("cuda:") => {
            DevicePreference::Gpu
        }
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
        "NEOETHOS_BOT_{}_GPU_ONLY",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    env::var(&model_key)
        .or_else(|_| env::var("NEOETHOS_BOT_GPU_ONLY"))
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
        "NEOETHOS_BOT_{}_THREADS",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    if let Ok(value) = env::var(&model_key)
        && let Ok(parsed) = value.trim().parse::<usize>()
        && parsed > 0
    {
        return parsed;
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

    fn parse_visible_devices(devices: &str) -> Option<usize> {
        let trimmed = devices.trim();
        if trimmed.is_empty() || trimmed == "-1" || trimmed.eq_ignore_ascii_case("void") {
            return Some(0);
        }
        let count = trimmed
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty() && *value != "-1")
            .count();
        (count > 0).then_some(count)
    }

    fn env_gpu_count(keys: &[&str]) -> Option<usize> {
        for key in keys {
            let Ok(devices) = env::var(key) else {
                continue;
            };
            if let Some(count) = parse_visible_devices(&devices) {
                return Some(count);
            }
        }
        None
    }

    fn parse_nvidia_smi_output(stdout: &str) -> Option<usize> {
        let count = stdout
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .count();
        Some(count)
    }

    /// GROUP H remediation (operator directive 2026-05-25): subprocess
    /// timeout so a broken-NVML or zombie-rocm-smi cannot hang the
    /// startup GPU probe forever. Spawns the subprocess on a separate
    /// thread and waits up to `timeout`. If the subprocess hangs, the
    /// main thread continues with `None` and the GPU probe falls back
    /// to env-var detection or 0. The subprocess MAY continue running
    /// in the background but the process is not blocked.
    fn run_subprocess_with_timeout(
        mut cmd: std::process::Command,
        timeout: std::time::Duration,
    ) -> Option<std::process::Output> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(cmd.output());
        });
        match rx.recv_timeout(timeout) {
            Ok(Ok(output)) => Some(output),
            Ok(Err(err)) => {
                tracing::debug!(
                    target: "neoethos_models::tree_config",
                    error = %err,
                    "GPU-detect subprocess failed to spawn"
                );
                None
            }
            Err(_) => {
                tracing::warn!(
                    target: "neoethos_models::tree_config",
                    timeout_ms = timeout.as_millis() as u64,
                    "GPU-detect subprocess timed out; treating as no-GPU"
                );
                None
            }
        }
    }

    /// Maximum time we wait for an external GPU-probe subprocess
    /// (`nvidia-smi`, `rocminfo`, `rocm-smi`) before assuming the host
    /// has no working accelerator. 2 seconds is generous — healthy
    /// hosts answer in <100 ms.
    const GPU_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

    fn nvidia_smi_gpu_count() -> Option<usize> {
        let mut cmd = Command::new("nvidia-smi");
        cmd.args(["--query-gpu=name", "--format=csv,noheader"]);
        let output = run_subprocess_with_timeout(cmd, GPU_PROBE_TIMEOUT)?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8(output.stdout).ok()?;
        parse_nvidia_smi_output(&stdout)
    }

    fn parse_rocm_output(stdout: &str) -> Option<usize> {
        let gfx_count = stdout
            .lines()
            .map(str::trim)
            .filter(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("gfx")
                    || lower.starts_with("gpu[")
                    || lower.starts_with("card series")
            })
            .count();
        (gfx_count > 0).then_some(gfx_count)
    }

    fn rocm_gpu_count() -> Option<usize> {
        let rocminfo = run_subprocess_with_timeout(Command::new("rocminfo"), GPU_PROBE_TIMEOUT);
        if let Some(output) = rocminfo
            && output.status.success()
            && let Ok(stdout) = String::from_utf8(output.stdout)
            && let Some(count) = parse_rocm_output(&stdout)
        {
            return Some(count);
        }

        let mut rocm_smi_cmd = Command::new("rocm-smi");
        rocm_smi_cmd.arg("--showproductname");
        let rocm_smi = run_subprocess_with_timeout(rocm_smi_cmd, GPU_PROBE_TIMEOUT);
        if let Some(output) = rocm_smi
            && output.status.success()
            && let Ok(stdout) = String::from_utf8(output.stdout)
            && let Some(count) = parse_rocm_output(&stdout)
        {
            return Some(count);
        }
        None
    }

    if let Some(count) = env_gpu_count(&[
        "FOREX_GPU_VISIBLE_DEVICES",
        "GPU_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "ROCM_VISIBLE_DEVICES",
    ]) {
        return count;
    }

    if let Ok(value) = env::var("FOREX_GPU_COUNT")
        && let Ok(parsed) = value.trim().parse::<usize>()
    {
        return parsed;
    }

    if let Some(count) = nvidia_smi_gpu_count() {
        return count;
    }

    if let Some(count) = rocm_gpu_count() {
        return count;
    }

    0
}

pub fn get_early_stop_params(default_patience: usize, default_min_delta: f64) -> (usize, f64) {
    let rt = current_tree_runtime();
    let p = rt
        .early_stop_patience
        .filter(|p| *p > 0)
        .unwrap_or(default_patience);
    let d = rt.early_stop_min_delta.unwrap_or(default_min_delta);
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
        if let Some(ParamValue::Int(value)) = params.get(key)
            && *value > 0
        {
            return *value as usize;
        }
        if let Some(ParamValue::String(value)) = params.get(key)
            && let Ok(parsed) = value.trim().parse::<usize>()
            && parsed > 0
        {
            return parsed;
        }
    }
    default
}

#[cfg(test)]
mod tests {
    use super::{gpu_count, parse_device_preference};

    #[test]
    fn parse_device_preference_accepts_vendor_aliases() {
        assert!(matches!(
            parse_device_preference("cuda"),
            super::DevicePreference::Gpu
        ));
        assert!(matches!(
            parse_device_preference("gpu"),
            super::DevicePreference::Gpu
        ));
        assert!(matches!(
            parse_device_preference("cpu"),
            super::DevicePreference::Cpu
        ));
    }

    #[test]
    fn gpu_count_reads_generic_override() {
        unsafe {
            std::env::set_var("FOREX_GPU_COUNT", "3");
        }
        assert_eq!(gpu_count(), 3);
        unsafe {
            std::env::remove_var("FOREX_GPU_COUNT");
        }
    }
}
