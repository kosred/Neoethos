use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::process::Command;
use std::sync::OnceLock;

use anyhow::Result;

use crate::common::{
    CudaDevicePolicy, ResolvedCudaDevicePolicy, parse_cuda_device_policy,
    resolve_cuda_device_policy,
};

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
    pub requested_device_policy: String,
    pub device_pref: DevicePreference,
    pub gpu_only: bool,
    pub cpu_threads: Option<usize>,
}

impl TreeModelConfig {
    pub fn resolved_cuda_device(&self) -> Result<ResolvedCudaDevicePolicy> {
        resolve_tree_cuda_device_policy(&self.requested_device_policy)
    }

    pub fn cuda_ordinal(&self) -> Result<Option<usize>> {
        cuda_ordinal_from_tree_policy(&self.requested_device_policy)
    }
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
    pub lightgbm_gpu: bool,
    /// One-release read-only compatibility cap. New runs install the shared
    /// process budget from `system.hardware.cpu_budget` before model startup.
    pub rayon_threads: Option<usize>,
}

impl Default for TreeRuntimeOverrides {
    fn default() -> Self {
        Self {
            device: "auto".to_string(),
            gpu_only: false,
            gpu_count: None,
            early_stop_patience: None,
            early_stop_min_delta: None,
            lightgbm_gpu: false,
            rayon_threads: None,
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
            lightgbm_gpu: c.lightgbm_gpu,
            rayon_threads: s.models.backtest_runtime.rayon_threads.filter(|n| *n > 0),
        }
    }
}

static TREE_RUNTIME: OnceLock<TreeRuntimeOverrides> = OnceLock::new();

/// Install ONLY this registry. `pub(crate)` so the single crate-wide
/// installer in `runtime::install` is the one entry point callers see.
pub(crate) fn set_tree_runtime(s: &neoethos_core::Settings) {
    let _ = TREE_RUNTIME.set(TreeRuntimeOverrides::from_settings(s));
}

/// Install the tree-model runtime config from `Settings` (call once at
/// startup, before any model training). The first install wins.
///
/// Kept under its historical name because the app, the desktop shell and the
/// CLI already call it. It now installs EVERY `neoethos-models` registry via
/// [`crate::runtime::install::install_model_runtime_from_settings`] and emits
/// the retired-env-var report, so no caller has to learn a new name to get the
/// config-only behaviour. New callers should prefer the aggregate directly.
pub fn install_tree_runtime_from_settings(s: &neoethos_core::Settings) {
    crate::runtime::install::install_model_runtime_from_settings(s);
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
    if let Some(installed) = neoethos_core::execution_budget::installed_process_budget() {
        return installed.resolved().effective_worker_limit.get();
    }

    // Unit tests and library-only embedders may reach this before a top-level
    // installer. Use the exact same leaf resolver, optionally narrowed by the
    // one-window legacy field; never recreate a local cores-minus-one rule.
    let mut request = neoethos_core::execution_budget::ExecutionBudgetRequest::detect(
        neoethos_core::execution_budget::CoordinationScope::ProcessLocal,
    );
    request.legacy_persistent_limit = current_tree_runtime().rayon_threads.map(|threads| {
        neoethos_core::execution_budget::BudgetCap::legacy(
            neoethos_core::execution_budget::WorkerLimit::new(threads)
                .expect("loaded legacy CPU cap is positive"),
        )
    });
    neoethos_core::execution_budget::resolve_execution_budget(request)
        .expect("tree-model fallback constructs valid CPU cap provenance")
        .effective_worker_limit
        .get()
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

/// Raw tree-device request, preserving an exact CUDA ordinal and invalid input
/// until the fallible execution boundary validates it.
pub fn tree_device_policy_from_params(
    params: &HashMap<String, ParamValue>,
    _model_name: &str,
) -> String {
    for key in ["device", "device_preference", "device_pref"] {
        if let Some(ParamValue::String(value)) = params.get(key) {
            let trimmed = value.trim();
            return if trimmed.is_empty() {
                "auto".to_string()
            } else {
                trimmed.to_ascii_lowercase()
            };
        }
    }
    let configured = current_tree_runtime().device.trim();
    if configured.is_empty() {
        "auto".to_string()
    } else {
        configured.to_ascii_lowercase()
    }
}

pub fn parse_tree_cuda_device_policy(value: &str) -> Result<CudaDevicePolicy> {
    parse_cuda_device_policy(value)
}

pub fn resolve_tree_cuda_device_policy(value: &str) -> Result<ResolvedCudaDevicePolicy> {
    resolve_cuda_device_policy(value, nvidia_gpu_count())
}

pub fn cuda_ordinal_from_tree_policy(value: &str) -> Result<Option<usize>> {
    Ok(match resolve_tree_cuda_device_policy(value)? {
        ResolvedCudaDevicePolicy::Cpu => None,
        ResolvedCudaDevicePolicy::Cuda { ordinal } => Some(ordinal),
    })
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

/// Whether LightGBM may resolve `device_type=cuda`. Config-driven
/// (`models.tree_runtime.lightgbm_gpu`), default `false`.
///
/// Separate from [`tree_device_preference`] on purpose: `device` states the
/// operator's intent for tree models generally, this states whether LightGBM
/// is permitted to act on it. See `TreeRuntimeConfig::lightgbm_gpu` for why
/// the default is off.
pub fn lightgbm_gpu_allowed() -> bool {
    current_tree_runtime().lightgbm_gpu
}

pub fn gpu_only_mode_for(_model_name: &str) -> bool {
    // Config-driven (was NEOETHOS_BOT_{MODEL}_GPU_ONLY → NEOETHOS_BOT_GPU_ONLY).
    // Per-model overrides are folded into the single global `gpu_only` knob.
    current_tree_runtime().gpu_only
}

/// Process-wide number of model-training slots currently reserved by parallel
/// trainers. Zero means no parallel trainer is active, which is interpreted as
/// one lone model for the per-model thread split.
struct TrainingConcurrencyCounter {
    active: std::sync::atomic::AtomicUsize,
}

impl TrainingConcurrencyCounter {
    const fn new() -> Self {
        Self {
            active: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn current(&self) -> usize {
        self.active
            .load(std::sync::atomic::Ordering::Acquire)
            .max(1)
    }

    fn reserve(&self, concurrent_models: usize) -> TrainingConcurrencyReservation<'_> {
        let reserved = concurrent_models.max(1);
        self.active
            .fetch_add(reserved, std::sync::atomic::Ordering::AcqRel);
        TrainingConcurrencyReservation {
            counter: self,
            reserved,
        }
    }
}

struct TrainingConcurrencyReservation<'a> {
    counter: &'a TrainingConcurrencyCounter,
    reserved: usize,
}

impl Drop for TrainingConcurrencyReservation<'_> {
    fn drop(&mut self) {
        let previous = self
            .counter
            .active
            .fetch_sub(self.reserved, std::sync::atomic::Ordering::AcqRel);
        debug_assert!(
            previous >= self.reserved,
            "training concurrency reservation underflow"
        );
    }
}

static TRAINING_CONCURRENCY: TrainingConcurrencyCounter = TrainingConcurrencyCounter::new();

/// RAII guard: adds this trainer's concurrency to the process-wide reservation
/// and removes exactly that reservation on drop, including during unwinding.
/// Overlapping trainers therefore cannot overwrite or prematurely reset one
/// another's aggregate native-worker throttle.
pub struct TrainingConcurrencyGuard {
    _reservation: TrainingConcurrencyReservation<'static>,
}

impl TrainingConcurrencyGuard {
    pub fn new(concurrent_models: usize) -> Self {
        Self {
            _reservation: TRAINING_CONCURRENCY.reserve(concurrent_models),
        }
    }
}

fn threads_per_model(target_total: usize, concurrent_models: usize) -> usize {
    (target_total / concurrent_models.max(1)).max(1)
}

pub fn cpu_threads_hint_for(_model_name: &str) -> usize {
    // Per-model NEOETHOS_BOT_{MODEL}_THREADS folded into the single
    // config-driven CPU budget via cpu_threads_hint() (v0.4.36
    // config-consolidation).
    //
    // Thread-oversubscription fix (Fable's pending task #9): the parallel
    // trainer runs up to `budget` (cores-1) models AT ONCE, and each tree
    // model (xgboost/lightgbm/catboost) reads THIS hint for its OWN internal
    // pool — so without coordination, K concurrent models each grabbing the
    // full budget = up to cores² threads thrashing on `cores` cores (25 on a
    // 6-core box).
    //
    // The resolved core hardware budget is the authoritative aggregate cap.
    // Divide it across concurrently-training models so outer Rayon workers
    // multiplied by inner native pools cannot recreate cores-squared thrash.
    threads_per_model(cpu_threads_hint(), TRAINING_CONCURRENCY.current())
}

pub fn gpu_count() -> usize {
    // The libtorch (`tch`) probe that used to sit here was removed 2026-08-09
    // (batch D4) with the `tch` feature itself — no build ever enabled it, so
    // this function has always fallen straight through to the env / nvidia-smi
    // probe below. Nothing about the returned count changes.
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

    // ⚠ THESE SIX ARE NOT OUR CONFIGURATION AND ARE DELIBERATELY KEPT.
    //
    // `CUDA_VISIBLE_DEVICES` and its vendor siblings are read by the NVIDIA /
    // ROCm drivers themselves: when one is set, the masked cards do not exist
    // as far as any CUDA context in this process is concerned. Reading them is
    // OBSERVING THE HARDWARE as the driver presents it, not reading a setting
    // out of the environment. Ignoring them would make this function report
    // cards the runtime cannot open — a config file cannot overrule a driver
    // mask. They are therefore not in `RETIRED_ENV_VARS` and must not be.
    let masked = env_gpu_count(&[
        "GPU_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "ROCM_VISIBLE_DEVICES",
    ]);
    // Explicit config count (`models.tree_runtime.gpu_count`, was the
    // `FOREX_GPU_COUNT` env var).
    let configured = current_tree_runtime().gpu_count;

    match (masked, configured) {
        // Both answered and they disagree: the SAFER (smaller) number binds
        // and the disagreement is logged with both values. Taking the larger
        // would either open a card the driver has masked away or oversubscribe
        // one the operator asked to leave alone.
        (Some(mask_count), Some(config_count)) if mask_count != config_count => {
            let effective = mask_count.min(config_count);
            tracing::warn!(
                target: "neoethos_models::tree_config",
                driver_visible_devices = mask_count,
                configured_gpu_count = config_count,
                effective = effective,
                "GPU count disagreement: a *_VISIBLE_DEVICES driver mask reports \
                 {mask_count} card(s) while models.tree_runtime.gpu_count says \
                 {config_count}; the safer (smaller) value {effective} binds"
            );
            return effective;
        }
        (Some(count), _) => return count,
        (None, Some(count)) => return count,
        (None, None) => {}
    }

    if let Some(count) = nvidia_smi_gpu_count() {
        return count;
    }

    if let Some(count) = rocm_gpu_count() {
        return count;
    }

    0
}

/// Count only NVIDIA devices visible to CUDA.
///
/// The general [`gpu_count`] intentionally recognises ROCm for cross-vendor
/// model backends. CUDA model routing must not use that answer: an AMD card is
/// not evidence that CubeCL/Candle/XGBoost CUDA can initialize.
pub fn nvidia_gpu_count() -> usize {
    fn parse_visible_devices(devices: &str) -> Option<usize> {
        let trimmed = devices.trim();
        if trimmed.is_empty()
            || trimmed == "-1"
            || trimmed.eq_ignore_ascii_case("void")
            || trimmed.eq_ignore_ascii_case("none")
        {
            return Some(0);
        }
        if trimmed.eq_ignore_ascii_case("all") {
            return None;
        }
        let count = trimmed
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty() && *value != "-1")
            .count();
        Some(count)
    }

    for key in ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"] {
        if let Ok(devices) = env::var(key)
            && let Some(count) = parse_visible_devices(&devices)
        {
            return count;
        }
    }

    const NVIDIA_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
    let mut command = Command::new("nvidia-smi");
    command.args(["--query-gpu=name", "--format=csv,noheader"]);
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = sender.send(command.output());
    });
    let Ok(Ok(output)) = receiver.recv_timeout(NVIDIA_PROBE_TIMEOUT) else {
        return 0;
    };
    if !output.status.success() {
        return 0;
    }
    let Ok(stdout) = String::from_utf8(output.stdout) else {
        return 0;
    };
    stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .count()
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
            return (*value as usize).min(default.max(1));
        }
        if let Some(ParamValue::String(value)) = params.get(key)
            && let Ok(parsed) = value.trim().parse::<usize>()
            && parsed > 0
        {
            return parsed.min(default.max(1));
        }
    }
    default
}

#[cfg(test)]
mod tests {
    use super::parse_device_preference;
    use super::{TrainingConcurrencyCounter, threads_per_model};

    #[test]
    fn per_model_threads_never_exceed_resolved_cpu_budget() {
        let target_total = 59;
        assert_eq!(threads_per_model(target_total, 1), target_total);
        let per_model = threads_per_model(target_total, 3);
        assert_eq!(per_model, 19);
        assert!(per_model * 3 <= target_total);
        assert_eq!(threads_per_model(2, 8), 1);
    }

    #[test]
    fn overlapping_training_reservations_accumulate_and_release_independently() {
        let counter = TrainingConcurrencyCounter::new();
        assert_eq!(counter.current(), 1, "no trainer means one lone model");
        {
            let first = counter.reserve(3);
            assert_eq!(counter.current(), 3);
            {
                let second = counter.reserve(2);
                assert_eq!(counter.current(), 5);
                drop(first);
                assert_eq!(
                    counter.current(),
                    2,
                    "dropping one trainer keeps the other's reservation"
                );
                drop(second);
            }
        }
        assert_eq!(
            counter.current(),
            1,
            "all reservations released restores the lone-model interpretation"
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

    #[test]
    fn strict_tree_cuda_parser_preserves_exact_ordinals_and_rejects_other_vendors() {
        assert_eq!(
            super::parse_tree_cuda_device_policy("cuda:3").expect("valid CUDA ordinal"),
            crate::common::CudaDevicePolicy::Gpu { ordinal: 3 }
        );
        assert!(super::parse_tree_cuda_device_policy("gpu:bad").is_err());
        assert!(super::parse_tree_cuda_device_policy("rocm:0").is_err());
    }
}
