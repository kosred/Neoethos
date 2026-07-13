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
/// device / GPU / early-stop readers below consult this instead of the old
/// `NEOETHOS_BOT_TREE_DEVICE` / `_GPU_ONLY` / `_EARLY_STOP_*` env vars
/// (v0.4.36 config-consolidation).
#[derive(Debug, Clone, PartialEq)]
pub struct TreeRuntimeOverrides {
    pub device: String,
    pub gpu_only: bool,
    pub gpu_count: Option<usize>,
    pub early_stop_patience: Option<usize>,
    pub early_stop_min_delta: Option<f64>,
}

impl Default for TreeRuntimeOverrides {
    fn default() -> Self {
        Self {
            device: "auto".to_string(),
            gpu_only: false,
            gpu_count: None,
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
            device: if c.device.trim().is_empty() {
                "auto".to_string()
            } else {
                c.device.clone()
            },
            gpu_only: c.gpu_only,
            gpu_count: c.gpu_count,
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
    // Config-driven CPU budget (was NEOETHOS_BOT_RUST_THREADS / _CPU_THREADS /
    // _CPU_BUDGET). Reads the single core hardware knob so the models and the
    // core hardware planner agree on the budget. The standard rayon
    // RAYON_NUM_THREADS is still honored as a fallback, then cores-1.
    if let Some(n) = neoethos_core::system::current_hardware_runtime_overrides()
        .cpu_budget
        .filter(|n| *n > 0)
    {
        return n;
    }
    if let Ok(val) = env::var("RAYON_NUM_THREADS")
        && let Ok(parsed) = val.trim().parse::<usize>()
        && parsed > 0
    {
        return parsed;
    }
    num_cpus::get().saturating_sub(1).max(1)
}

pub fn tree_device_preference() -> DevicePreference {
    tree_device_preference_for("tree")
}

pub fn tree_device_preference_for(_model_name: &str) -> DevicePreference {
    // Config-driven (was NEOETHOS_BOT_{MODEL}_DEVICE → NEOETHOS_BOT_TREE_DEVICE).
    // Per-model overrides are folded into the single global `device` knob;
    // `parse_device_preference` applies the same string vocabulary as before.
    parse_device_preference(&current_tree_runtime().device)
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

pub fn gpu_only_mode_for(_model_name: &str) -> bool {
    // Config-driven (was NEOETHOS_BOT_{MODEL}_GPU_ONLY → NEOETHOS_BOT_GPU_ONLY).
    // Per-model overrides are folded into the single global `gpu_only` knob.
    current_tree_runtime().gpu_only
}

/// Number of models the parallel trainer is running CONCURRENTLY. `1` when
/// training a single model (or when the parallel trainer isn't active), so
/// a lone model still gets the full CPU budget.
static TRAINING_CONCURRENCY: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(1);

/// Tell the per-model thread hint how many models train at once, so each
/// one takes `budget / concurrency` threads instead of the full budget.
/// The parallel trainer sets this via [`TrainingConcurrencyGuard`].
pub(crate) fn set_training_concurrency(n: usize) {
    TRAINING_CONCURRENCY.store(n.max(1), std::sync::atomic::Ordering::Relaxed);
}

fn training_concurrency() -> usize {
    TRAINING_CONCURRENCY
        .load(std::sync::atomic::Ordering::Relaxed)
        .max(1)
}

/// RAII guard: sets the training concurrency for its lifetime and restores
/// it to 1 on drop (even on panic), so the throttle never leaks into a
/// later single-model training run.
pub struct TrainingConcurrencyGuard;

impl TrainingConcurrencyGuard {
    pub fn new(concurrent_models: usize) -> Self {
        set_training_concurrency(concurrent_models);
        Self
    }
}

impl Drop for TrainingConcurrencyGuard {
    fn drop(&mut self) {
        set_training_concurrency(1);
    }
}

pub fn cpu_threads_hint_for(_model_name: &str) -> usize {
    // Per-model NEOETHOS_BOT_{MODEL}_THREADS folded into the single
    // config-driven CPU budget via cpu_threads_hint() (v0.4.36
    // config-consolidation).
    //
    // Thread-oversubscription fix (Fable's pending task #9, 2026-07-13): the
    // parallel trainer runs up to `budget` models AT ONCE, and each tree
    // model (xgboost/lightgbm/catboost) reads THIS hint for its OWN internal
    // thread pool — so without dividing, K concurrent models × budget
    // threads = budget² threads thrashing on `budget` cores (25 threads on a
    // 6-core box). Divide the budget by how many models run concurrently so
    // the product stays ≈ budget. Floored at 1 and never above the budget,
    // so it can only REDUCE oversubscription, never add it.
    (cpu_threads_hint() / training_concurrency()).max(1)
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
        "GPU_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "ROCM_VISIBLE_DEVICES",
    ]) {
        return count;
    }

    // Explicit config override (was the `FOREX_GPU_COUNT` env var). Sits after
    // the standard `*_VISIBLE_DEVICES` probe — the same precedence
    // `FOREX_GPU_COUNT` had — and before the nvidia-smi / rocm subprocess probes.
    if let Some(count) = current_tree_runtime().gpu_count {
        return count;
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
    use super::parse_device_preference;
    use super::{TrainingConcurrencyGuard, cpu_threads_hint, cpu_threads_hint_for};

    #[test]
    fn per_model_threads_divide_by_concurrency_and_restore() {
        // Thread-oversubscription fix: with K models running concurrently,
        // each tree model must get budget/K threads (never the full budget),
        // and the guard must restore the single-model default on drop.
        let base = cpu_threads_hint();
        assert_eq!(cpu_threads_hint_for("xgboost"), base, "lone model gets full budget");
        {
            let _g = TrainingConcurrencyGuard::new(3);
            let throttled = cpu_threads_hint_for("xgboost");
            assert_eq!(throttled, (base / 3).max(1));
            assert!(throttled <= base, "throttle can only reduce, never add");
            assert!(throttled >= 1, "at least one thread per model");
        }
        assert_eq!(
            cpu_threads_hint_for("xgboost"),
            base,
            "budget restored after the guard drops"
        );
    }

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
}
