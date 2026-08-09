//! Typed discovery-evaluation backend policy.
//!
//! Stage 1 / Commit 0.1 of the GPU-native discovery redesign.  The historical
//! configuration exposed overlapping string knobs (`system.enable_gpu_preference`
//! and `models.prop_search_device`) plus `NEOETHOS_REQUIRE_GPU`.  This module is
//! the single typed boundary that resolves those inputs without changing the
//! legacy meanings of `cpu`, `auto`, or `gpu`.

use neoethos_core::Settings;
use std::error::Error;
use std::fmt;
use std::sync::{OnceLock, RwLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DevicePreference {
    Cpu,
    Auto,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FallbackPolicy {
    AllowCpu,
    ForbidCpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceleratorHint {
    Any,
    Cuda,
    Wgpu,
    Vulkan,
    Rocm,
    Metal,
    Dx12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvaluationBackend {
    pub device: DevicePreference,
    pub fallback: FallbackPolicy,
    pub accelerator_hint: AcceleratorHint,
}

impl Default for EvaluationBackend {
    fn default() -> Self {
        Self::AUTO
    }
}

impl EvaluationBackend {
    pub const CPU_CANONICAL: Self = Self {
        device: DevicePreference::Cpu,
        fallback: FallbackPolicy::AllowCpu,
        accelerator_hint: AcceleratorHint::Any,
    };

    pub const AUTO: Self = Self {
        device: DevicePreference::Auto,
        fallback: FallbackPolicy::AllowCpu,
        accelerator_hint: AcceleratorHint::Any,
    };

    pub const GPU_PREFERRED: Self = Self {
        device: DevicePreference::Gpu,
        fallback: FallbackPolicy::AllowCpu,
        accelerator_hint: AcceleratorHint::Any,
    };

    pub const GPU_REQUIRED: Self = Self {
        device: DevicePreference::Gpu,
        fallback: FallbackPolicy::ForbidCpu,
        accelerator_hint: AcceleratorHint::Any,
    };

    pub fn parse(raw: &str) -> Result<Self, BackendConfigError> {
        let normalized = normalize(raw);
        let parsed = match normalized.as_str() {
            "" | "auto" => Self::AUTO,
            "cpu" | "off" | "false" => Self::CPU_CANONICAL,
            "gpu" | "on" | "true" => Self::GPU_PREFERRED,
            "gpu_required" | "gpu-required" => Self::GPU_REQUIRED,
            "cuda" => Self::gpu_with(AcceleratorHint::Cuda, FallbackPolicy::AllowCpu),
            "cuda_required" | "cuda-required" => {
                Self::gpu_with(AcceleratorHint::Cuda, FallbackPolicy::ForbidCpu)
            }
            "wgpu" => Self::gpu_with(AcceleratorHint::Wgpu, FallbackPolicy::AllowCpu),
            "wgpu_required" | "wgpu-required" => {
                Self::gpu_with(AcceleratorHint::Wgpu, FallbackPolicy::ForbidCpu)
            }
            "vulkan" => Self::gpu_with(AcceleratorHint::Vulkan, FallbackPolicy::AllowCpu),
            "vulkan_required" | "vulkan-required" => {
                Self::gpu_with(AcceleratorHint::Vulkan, FallbackPolicy::ForbidCpu)
            }
            "rocm" => Self::gpu_with(AcceleratorHint::Rocm, FallbackPolicy::AllowCpu),
            "rocm_required" | "rocm-required" => {
                Self::gpu_with(AcceleratorHint::Rocm, FallbackPolicy::ForbidCpu)
            }
            "metal" => Self::gpu_with(AcceleratorHint::Metal, FallbackPolicy::AllowCpu),
            "metal_required" | "metal-required" => {
                Self::gpu_with(AcceleratorHint::Metal, FallbackPolicy::ForbidCpu)
            }
            "dx12" => Self::gpu_with(AcceleratorHint::Dx12, FallbackPolicy::AllowCpu),
            "dx12_required" | "dx12-required" => {
                Self::gpu_with(AcceleratorHint::Dx12, FallbackPolicy::ForbidCpu)
            }
            _ => return Err(BackendConfigError::UnknownPreference(raw.trim().to_owned())),
        };
        parsed.validate()?;
        Ok(parsed)
    }

    pub fn resolve_for_discovery(
        global_preference: &str,
        discovery_preference: &str,
        require_gpu_env: Option<&str>,
    ) -> Result<Self, BackendConfigError> {
        // The discovery-specific value wins whenever it is present.  This keeps
        // the existing `models.prop_search_device` behaviour intact; an empty
        // value deliberately inherits the global system preference.
        let selected = if discovery_preference.trim().is_empty() {
            global_preference
        } else {
            discovery_preference
        };
        let mut backend = Self::parse(selected)?;

        if parse_optional_bool("NEOETHOS_REQUIRE_GPU", require_gpu_env)?.unwrap_or(false) {
            // The environment override can only escalate.  It forces a GPU as
            // well as forbidding CPU fallback, avoiding the invalid
            // `Cpu + ForbidCpu` state.
            backend.device = DevicePreference::Gpu;
            backend.fallback = FallbackPolicy::ForbidCpu;
        }
        backend.validate()?;
        Ok(backend)
    }

    pub fn from_settings(
        settings: &Settings,
        require_gpu_env: Option<&str>,
    ) -> Result<Self, BackendConfigError> {
        Self::resolve_for_discovery(
            &settings.system.enable_gpu_preference,
            &settings.models.prop_search_device,
            require_gpu_env,
        )
    }

    pub fn from_settings_and_process_env(settings: &Settings) -> Result<Self, BackendConfigError> {
        let env_value = std::env::var("NEOETHOS_REQUIRE_GPU").ok();
        Self::from_settings(settings, env_value.as_deref())
    }

    pub fn cpu_fallback_allowed(self) -> bool {
        self.fallback == FallbackPolicy::AllowCpu
    }

    pub fn gpu_required(self) -> bool {
        self.device == DevicePreference::Gpu && self.fallback == FallbackPolicy::ForbidCpu
    }

    pub fn validate(self) -> Result<(), BackendConfigError> {
        if self.device == DevicePreference::Cpu && self.fallback == FallbackPolicy::ForbidCpu {
            return Err(BackendConfigError::ConflictingPolicy {
                device: self.device,
                fallback: self.fallback,
            });
        }
        Ok(())
    }

    const fn gpu_with(hint: AcceleratorHint, fallback: FallbackPolicy) -> Self {
        Self {
            device: DevicePreference::Gpu,
            fallback,
            accelerator_hint: hint,
        }
    }
}

fn backend_slot() -> &'static RwLock<EvaluationBackend> {
    static BACKEND: OnceLock<RwLock<EvaluationBackend>> = OnceLock::new();
    BACKEND.get_or_init(|| RwLock::new(EvaluationBackend::AUTO))
}

/// Install the resolved backend for production discovery. The lock is mutable so
/// tests and long-lived frontends may install a freshly loaded config before a new
/// work unit without relying on ambient process environment reads.
pub fn install_evaluation_backend(backend: EvaluationBackend) -> Result<(), BackendConfigError> {
    backend.validate()?;
    let mut slot = backend_slot()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *slot = backend;
    Ok(())
}

pub fn install_evaluation_backend_from_settings(
    settings: &Settings,
) -> Result<EvaluationBackend, BackendConfigError> {
    let backend = EvaluationBackend::from_settings_and_process_env(settings)?;
    install_evaluation_backend(backend)?;
    Ok(backend)
}

pub fn current_evaluation_backend() -> EvaluationBackend {
    *backend_slot()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Production entry point for population evaluation. CPU canonical mode is
/// explicitly audited; optional GPU modes retain the legacy adaptive hybrid;
/// strict GPU mode executes the full logical population on the GPU and can only
/// recover by deterministic, bounded GPU rebatching.
pub fn evaluate_population_core_with_backend(
    inputs: crate::eval::PopulationEvalInputs<'_>,
    backend: EvaluationBackend,
) -> Result<Vec<[f64; 11]>, String> {
    let audit = crate::gpu_native::cpu_strategy::CpuStrategyAuditContext::production(0);
    evaluate_population_core_with_backend_and_audit(inputs, backend, &audit)
}

pub fn evaluate_population_core_with_backend_and_audit(
    inputs: crate::eval::PopulationEvalInputs<'_>,
    backend: EvaluationBackend,
    audit: &crate::gpu_native::cpu_strategy::CpuStrategyAuditContext,
) -> Result<Vec<[f64; 11]>, String> {
    use crate::gpu_native::cpu_strategy::{self, CpuStrategyCategory};

    backend.validate().map_err(|error| error.to_string())?;
    // Which arm ran, and what was actually in the backend slot.
    //
    // Telemetry showed the GA reaching `validation_backtest_population_cpu`
    // while `evaluate_population_core` was never entered — which can only
    // happen through the first arm, yet nothing in the tree installs a CPU
    // backend. Rather than reason about which of those is wrong, the dispatch
    // now states the value it dispatched on.
    {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                target: "neoethos_search::backend",
                device = ?backend.device,
                fallback = ?backend.fallback,
                accelerator = ?backend.accelerator_hint,
                "population evaluation dispatched on this backend"
            );
        });
    }
    // Fail-loud guard (task #35, 2026-08-09): running the GA population eval on
    // the CPU WHILE a usable CUDA card is present is almost always a
    // misconfiguration (a stray `prop_search_device: cpu`), and it silently
    // costs ~97% of a run's wall time. That went unnoticed for eight months
    // because it looked like a healthy-but-slow run. Warn ONCE, loudly, so it
    // can never hide again.
    #[cfg(feature = "gpu-b-adapter")]
    if backend.device == DevicePreference::Cpu
        && crate::gpu_native::prototype_b_population_eval::prototype_b_available()
    {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                target: "neoethos_search::backend",
                "population eval is on the CPU but a CUDA card is available — the GA is \
                 leaving ~97% of the card idle. This is almost certainly a stray \
                 `models.prop_search_device: cpu`; set it to `auto` to route the GA to \
                 the native f64 prototype-B lane."
            );
        });
    }
    match (backend.device, backend.fallback) {
        (DevicePreference::Cpu, _) => cpu_strategy::run(
            backend,
            audit,
            CpuStrategyCategory::PopulationEvaluation,
            "backend::cpu_canonical_population",
            || crate::eval::validation_backtest_population_cpu(inputs),
        )
        .map_err(|error| error.to_string()),
        (DevicePreference::Gpu, FallbackPolicy::ForbidCpu) => {
            evaluate_gpu_required_population(inputs, backend, audit)
        }
        _ => crate::eval::evaluate_population_core(inputs),
    }
}

#[cfg(not(feature = "gpu"))]
fn evaluate_gpu_required_population(
    _inputs: crate::eval::PopulationEvalInputs<'_>,
    _backend: EvaluationBackend,
    audit: &crate::gpu_native::cpu_strategy::CpuStrategyAuditContext,
) -> Result<Vec<[f64; 11]>, String> {
    audit
        .snapshot()
        .assert_zero_executed()
        .map_err(|error| error.to_string())?;
    Err(
        "gpu_required was selected, but neoethos-search was compiled without a GPU backend feature"
            .to_string(),
    )
}

#[cfg(feature = "gpu")]
fn evaluate_gpu_required_population(
    inputs: crate::eval::PopulationEvalInputs<'_>,
    backend: EvaluationBackend,
    audit: &crate::gpu_native::cpu_strategy::CpuStrategyAuditContext,
) -> Result<Vec<[f64; 11]>, String> {
    use crate::cubecl_eval::{
        cuda_eval_backtest_kernel_enabled, cuda_eval_signal_kernel_enabled,
        try_evaluate_population_cuda,
    };
    use crate::gpu_fallback::{GpuAction, GpuAttempt, GpuFailure, GpuRetryPolicy, decide_action};

    if !cuda_eval_signal_kernel_enabled() || !cuda_eval_backtest_kernel_enabled() {
        return Err("gpu_required population evaluation cannot start because a required CubeCL signal/backtest kernel is disabled".to_string());
    }

    let crate::eval::PopulationEvalInputs {
        close,
        high,
        low,
        indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        month_idx,
        day_idx,
        timestamps,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        smc_data,
        gene_smc_flags,
        gate_threshold,
        weights,
        settings,
    } = inputs;

    let n_genes = long_thr.len();
    if n_genes == 0 {
        audit
            .snapshot()
            .assert_zero_executed()
            .map_err(|error| error.to_string())?;
        return Ok(Vec::new());
    }
    if gene_offsets.len() != n_genes + 1 || gene_smc_flags.len() != n_genes {
        return Err(
            "gpu_required population evaluation received inconsistent gene CSR/SMC dimensions"
                .to_string(),
        );
    }

    let retry_policy = GpuRetryPolicy::default();
    let mut attempt = GpuAttempt::first(n_genes);

    loop {
        let batch_size = attempt.batch_size.clamp(1, n_genes);
        let mut output = Vec::with_capacity(n_genes);
        let mut failure: Option<(GpuFailure, String)> = None;
        let mut start = 0usize;

        while start < n_genes {
            let end = (start + batch_size).min(n_genes);
            let idx_start = gene_offsets[start] as usize;
            let idx_end = gene_offsets[end] as usize;
            if idx_start > idx_end || idx_end > gene_indices.len() || idx_end > gene_weights.len() {
                return Err(
                    "gpu_required population evaluation received invalid CSR offsets".to_string(),
                );
            }
            let base = gene_offsets[start];
            let rebased_offsets: Vec<i32> = gene_offsets[start..=end]
                .iter()
                .map(|offset| *offset - base)
                .collect();
            let stop_slice = if stop_vol_mult.len() >= end {
                &stop_vol_mult[start..end]
            } else {
                &[]
            };

            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                try_evaluate_population_cuda(
                    close,
                    high,
                    low,
                    indicators,
                    &rebased_offsets,
                    &gene_indices[idx_start..idx_end],
                    &gene_weights[idx_start..idx_end],
                    &long_thr[start..end],
                    &short_thr[start..end],
                    month_idx,
                    day_idx,
                    timestamps,
                    &sl_pips[start..end],
                    &tp_pips[start..end],
                    stop_slice,
                    smc_data,
                    &gene_smc_flags[start..end],
                    gate_threshold,
                    weights,
                    settings,
                    None,
                )
            }));

            match outcome {
                Ok(Ok(batch)) if batch.len() == end - start => output.extend(batch),
                Ok(Ok(batch)) => {
                    failure = Some((
                        GpuFailure::WrongShape,
                        format!(
                            "GPU returned {} rows for a {}-candidate batch",
                            batch.len(),
                            end - start
                        ),
                    ));
                    break;
                }
                Ok(Err(error)) => {
                    let detail = error.to_string();
                    let lower = detail.to_ascii_lowercase();
                    let kind = if lower.contains("adapter") || lower.contains("no device") {
                        GpuFailure::NoAdapter
                    } else if lower.contains("device lost") {
                        GpuFailure::DeviceLost
                    } else if lower.contains("unsupported") {
                        GpuFailure::UnsupportedBackend
                    } else {
                        GpuFailure::AllocationPressure
                    };
                    failure = Some((kind, detail));
                    break;
                }
                Err(payload) => {
                    let detail = payload
                        .downcast_ref::<&str>()
                        .map(|value| (*value).to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "non-string GPU panic".to_string());
                    failure = Some((GpuFailure::AllocationPressure, detail));
                    break;
                }
            }
            start = end;
        }

        if let Some((kind, detail)) = failure {
            match decide_action(kind, backend, attempt, retry_policy) {
                GpuAction::RetryOnGpu { next_batch_size } => {
                    tracing::warn!(
                        target: "neoethos_search::backend",
                        ?kind,
                        retry_index = attempt.retry_index,
                        previous_batch_size = attempt.batch_size,
                        next_batch_size,
                        error = %detail,
                        "gpu_required population batch failed; retrying only on GPU with a smaller deterministic batch"
                    );
                    attempt = GpuAttempt {
                        retry_index: attempt.retry_index.saturating_add(1),
                        batch_size: next_batch_size,
                    };
                    continue;
                }
                GpuAction::FailLoud => {
                    return Err(format!(
                        "gpu_required population evaluation failed closed after {} retries: {:?}: {}",
                        attempt.retry_index, kind, detail
                    ));
                }
                GpuAction::FallbackToCpu => {
                    return Err(
                        "internal policy error: gpu_required selected a CPU fallback".to_string(),
                    );
                }
            }
        }

        audit
            .snapshot()
            .assert_zero_executed()
            .map_err(|error| error.to_string())?;
        return Ok(output);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendConfigError {
    UnknownPreference(String),
    InvalidBoolean {
        key: &'static str,
        value: String,
    },
    ConflictingPolicy {
        device: DevicePreference,
        fallback: FallbackPolicy,
    },
}

impl fmt::Display for BackendConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPreference(value) => write!(
                f,
                "unknown discovery compute preference `{value}`; expected cpu, auto, gpu, gpu_required, or a supported accelerator hint"
            ),
            Self::InvalidBoolean { key, value } => write!(
                f,
                "invalid boolean value `{value}` for {key}; expected 1/0, true/false, yes/no, or on/off"
            ),
            Self::ConflictingPolicy { device, fallback } => write!(
                f,
                "invalid evaluation backend policy: device={device:?}, fallback={fallback:?}"
            ),
        }
    }
}

impl Error for BackendConfigError {}

fn normalize(raw: &str) -> String {
    raw.trim().to_ascii_lowercase()
}

fn parse_optional_bool(
    key: &'static str,
    raw: Option<&str>,
) -> Result<Option<bool>, BackendConfigError> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let normalized = normalize(raw);
    match normalized.as_str() {
        "" => Ok(None),
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => Err(BackendConfigError::InvalidBoolean {
            key,
            value: raw.trim().to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_values_keep_their_meaning() {
        assert_eq!(
            EvaluationBackend::parse("cpu").unwrap(),
            EvaluationBackend::CPU_CANONICAL
        );
        assert_eq!(
            EvaluationBackend::parse("auto").unwrap(),
            EvaluationBackend::AUTO
        );
        assert_eq!(
            EvaluationBackend::parse("gpu").unwrap(),
            EvaluationBackend::GPU_PREFERRED
        );
    }

    #[test]
    fn gpu_required_is_a_new_strict_value() {
        assert_eq!(
            EvaluationBackend::parse("gpu_required").unwrap(),
            EvaluationBackend::GPU_REQUIRED
        );
        assert!(
            EvaluationBackend::parse("gpu_required")
                .unwrap()
                .gpu_required()
        );
    }

    #[test]
    fn accelerator_hint_is_preserved() {
        let backend = EvaluationBackend::parse("cuda_required").unwrap();
        assert_eq!(backend.device, DevicePreference::Gpu);
        assert_eq!(backend.fallback, FallbackPolicy::ForbidCpu);
        assert_eq!(backend.accelerator_hint, AcceleratorHint::Cuda);
    }

    #[test]
    fn discovery_specific_value_overrides_global() {
        let backend = EvaluationBackend::resolve_for_discovery("gpu", "cpu", None).unwrap();
        assert_eq!(backend, EvaluationBackend::CPU_CANONICAL);

        let inherited = EvaluationBackend::resolve_for_discovery("gpu", "", None).unwrap();
        assert_eq!(inherited, EvaluationBackend::GPU_PREFERRED);
    }

    #[test]
    fn require_gpu_env_only_escalates() {
        for value in ["1", "true", "YES", "on"] {
            let backend =
                EvaluationBackend::resolve_for_discovery("auto", "cpu", Some(value)).unwrap();
            assert_eq!(backend.device, DevicePreference::Gpu);
            assert_eq!(backend.fallback, FallbackPolicy::ForbidCpu);
        }
    }

    #[test]
    fn false_and_empty_env_values_do_not_escalate() {
        for value in ["0", "false", "NO", "off", ""] {
            let backend =
                EvaluationBackend::resolve_for_discovery("auto", "gpu", Some(value)).unwrap();
            assert_eq!(backend, EvaluationBackend::GPU_PREFERRED);
        }
    }

    #[test]
    fn invalid_boolean_fails_loud() {
        let error =
            EvaluationBackend::resolve_for_discovery("auto", "gpu", Some("maybe")).unwrap_err();
        assert!(matches!(error, BackendConfigError::InvalidBoolean { .. }));
    }

    #[test]
    fn cpu_forbid_cpu_is_rejected() {
        let invalid = EvaluationBackend {
            device: DevicePreference::Cpu,
            fallback: FallbackPolicy::ForbidCpu,
            accelerator_hint: AcceleratorHint::Any,
        };
        assert!(matches!(
            invalid.validate(),
            Err(BackendConfigError::ConflictingPolicy { .. })
        ));
    }
}
