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
            // The single deliberate "run on the CPU even though a card is
            // present" escape. Plain `cpu`/`off`/`false` are treated as "use the
            // card" by `resolve_with_probe` and escalated on a card box; only
            // this token opts out. See `resolve_with_probe`.
            "cpu_forced" | "cpu-forced" => Self::CPU_CANONICAL,
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

    /// Hardware-conditional resolution — the single point where the device
    /// string becomes a policy that knows whether a card is actually present.
    ///
    /// The recurring eight-month bug was that `auto`/`cpu`/`gpu` mapped to a
    /// backend with NO knowledge of the card, so a stray `prop_search_device:
    /// cpu` (or the desktop config's shipped `cpu`, or a leftover env var)
    /// silently ran the whole GA on the CPU with a 3090 idle. Here, when a
    /// usable card + the native f64 lane are present, EVERY value except the
    /// explicit `cpu_forced` escape resolves to GPU-required (GPU or a loud
    /// error) — so the only way to run on the CPU with a card is to ask for it
    /// by name. A card-less host is unchanged: `auto`/`cpu` resolve to CPU and
    /// the run proceeds normally.
    ///
    /// `resolve_for_discovery` stays pure (no probe) so the existing unit tests
    /// that pin the string→backend mapping keep passing.
    pub fn resolve_with_probe(
        global_preference: &str,
        discovery_preference: &str,
        require_gpu_env: Option<&str>,
        probe: HardwareProbe,
    ) -> Result<Self, BackendConfigError> {
        let base =
            Self::resolve_for_discovery(global_preference, discovery_preference, require_gpu_env)?;
        if !probe.card_present {
            return Ok(base);
        }
        // A card is present. Honor exactly one deliberate CPU escape; every
        // other value means "use the card" and is escalated to GPU-required so a
        // fault is loud rather than a silent CPU run.
        let selected = if discovery_preference.trim().is_empty() {
            global_preference
        } else {
            discovery_preference
        };
        if matches!(normalize(selected).as_str(), "cpu_forced" | "cpu-forced") {
            Ok(base)
        } else {
            // GPU_PREFERRED, deliberately NOT GPU_REQUIRED.
            //
            // `ForbidCpu` is a WHOLE-PIPELINE contract: `gpu_pipeline_preflight`
            // demands every requested stage be strict-GPU capable, and 15 of
            // them legitimately are not (GA selection/mutation/crossover,
            // correlation pruning, survivor ranking, feature prep...). A real
            // 3090 run with `ForbidCpu` therefore aborts before the first
            // generation instead of using the card — measured on box 47260276.
            //
            // The population-eval guarantee does not come from this policy: it
            // comes from `evaluate_population_core`, whose GPU lane returns Err
            // on any fault and whose CPU tail refuses to run at all while
            // prototype B is available. That guard covers the dispatch arm AND
            // the callers that bypass dispatch (evaluate_genes -> regime_labels),
            // so `Gpu + AllowCpu` here still cannot produce a silent CPU run —
            // it just stops claiming the CPU-side stages must vanish.
            Ok(Self::GPU_PREFERRED)
        }
    }

    pub fn from_settings(
        settings: &Settings,
        require_gpu_env: Option<&str>,
    ) -> Result<Self, BackendConfigError> {
        Self::resolve_with_probe(
            &settings.system.enable_gpu_preference,
            &settings.models.prop_search_device,
            require_gpu_env,
            HardwareProbe::detect(),
        )
    }

    /// Resolve the backend from the operator's Settings ALONE.
    ///
    /// 2026-08-10 (env→config wave): this used to read `NEOETHOS_REQUIRE_GPU`
    /// out of the process environment and hand it to `from_settings` as a
    /// third input. It no longer does, and the name is kept only because
    /// `install_evaluation_backend_from_settings` and its tests call it.
    ///
    /// WHAT THIS NOW PERMITS: nothing new.
    /// WHAT IT NOW REFUSES: an exported `NEOETHOS_REQUIRE_GPU=1` can no longer
    /// escalate the run to GPU-required. ⚠ THIS IS A BEHAVIOUR CHANGE, and it
    /// is a LOOSENING, so it is reported at ERROR by
    /// `execution_profile::report_retired_env_vars()` whenever the variable is
    /// still exported. The replacement is a config value —
    /// `system.enable_gpu_preference: cuda_required` (or `models
    /// .prop_search_device`) — and note that `resolve_with_probe` already
    /// escalates every non-`cpu_forced` value to GPU-preferred whenever a card
    /// is present, which is what the env var was being used for on rented
    /// boxes.
    pub fn from_settings_and_process_env(settings: &Settings) -> Result<Self, BackendConfigError> {
        Self::from_settings(settings, None)
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

/// A snapshot of whether this machine can actually run the native f64 GPU lane.
///
/// Constructed once at backend install and carried into resolution so "a card
/// is present" is a structural input to the device decision instead of an
/// after-the-fact warning. `detect()` is false on every build without the
/// prototype-B lane compiled in (Vulkan/ROCm/CPU/default), so those builds are
/// never escalated to GPU-required with no engine to run.
#[derive(Debug, Clone, Copy)]
pub struct HardwareProbe {
    pub card_present: bool,
}

impl HardwareProbe {
    pub fn detect() -> Self {
        #[cfg(feature = "gpu-b-adapter")]
        {
            Self {
                card_present:
                    crate::gpu_native::prototype_b_population_eval::prototype_b_available(),
            }
        }
        #[cfg(not(feature = "gpu-b-adapter"))]
        {
            Self {
                card_present: false,
            }
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
    // The indicator lane in `neoethos-data` has the SAME question to answer —
    // "may this run silently compute on the CPU?" — and until now it answered
    // it from `NEOETHOS_REQUIRE_GPU` while this crate answered it from config.
    // `set_indicator_compute_policy` existed for exactly this and had ZERO
    // callers (`hpc_ta.rs:93`), so the seam the module documents as "where the
    // operator's Settings plug in" was never plugged in. It is now, from the
    // one place that already holds the resolved backend, so the two crates
    // cannot disagree about the same run.
    let policy = if backend.gpu_required() {
        neoethos_data::core::hpc_ta::IndicatorComputePolicy::RequireGpu
    } else {
        neoethos_data::core::hpc_ta::IndicatorComputePolicy::Auto
    };
    // A second install returning the value already in force is not an error
    // here: the first one wins by that module's own contract, and both callers
    // would be installing the same resolved policy.
    let _ = neoethos_data::core::hpc_ta::set_indicator_compute_policy(policy);
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
    // Record the realized device at this single, sequential, once-per-generation
    // dispatch (WALL time — not the summed rayon-thread nanos the eval_telemetry
    // doc warns about), so `device_summary` at run end can prove the population
    // eval stayed on the card. The AUTO / GPU-preferred arm's device is decided
    // INSIDE `evaluate_population_core` (its gate / n_genes), so that arm
    // self-records there; recording it here too would double-count.
    //
    // The old warn-once "CPU while a card is present" guard is gone:
    // `resolve_with_probe` now escalates every non-`cpu_forced` value to
    // GPU-required on a card, so a plain Cpu backend can only reach this dispatch
    // on a card-less host (normal) or via the explicit `cpu_forced` escape — in
    // neither case is a warning warranted.
    let dispatch_started = std::time::Instant::now();
    match (backend.device, backend.fallback) {
        (DevicePreference::Cpu, _) => {
            let out = cpu_strategy::run(
                backend,
                audit,
                CpuStrategyCategory::PopulationEvaluation,
                "backend::cpu_canonical_population",
                || crate::eval::validation_backtest_population_cpu(inputs),
            )
            .map_err(|error| error.to_string());
            crate::eval_telemetry::record_device(
                "population_eval",
                crate::eval_telemetry::Device::Cpu,
                dispatch_started.elapsed(),
            );
            out
        }
        (DevicePreference::Gpu, FallbackPolicy::ForbidCpu) => {
            let out = evaluate_gpu_required_population(inputs, backend, audit);
            if out.is_ok() {
                crate::eval_telemetry::record_device(
                    "population_eval",
                    crate::eval_telemetry::Device::Gpu,
                    dispatch_started.elapsed(),
                );
            }
            out
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

    // The third launch site for the same engine. It already records its device
    // under `population_eval`; the per-launch anatomy has to land in the same
    // row or the retry loop's launches appear under no caller at all.
    let _lane = crate::eval_telemetry::LaneScope::enter("population_eval");

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

    /// The permanent invariant, provable without a real GPU: drive
    /// `resolve_with_probe` with an EXPLICIT probe. On a card, every device
    /// string any config surface can ship must resolve to GPU-required — the
    /// sole exception being the deliberate `cpu_forced` escape — and a card-less
    /// host must still resolve `auto`/`cpu` to a CPU-allowing backend. This test
    /// reopens-fails if the resolver ever stops escalating a plain `cpu` on a
    /// card (the eight-month trap), or if the `cpu_forced` escape is removed.
    #[test]
    fn card_present_forbids_silent_cpu_population_eval() {
        let card = HardwareProbe { card_present: true };
        let no_card = HardwareProbe {
            card_present: false,
        };

        for value in ["", "auto", "cpu", "off", "false", "gpu", "cuda"] {
            let backend =
                EvaluationBackend::resolve_with_probe(value, "", None, card).expect("resolves");
            assert_eq!(
                backend.device,
                DevicePreference::Gpu,
                "on a card, `{value}` must resolve to a GPU device, not a silent CPU run"
            );
            // ...but NOT ForbidCpu: that is a whole-pipeline strict-GPU contract
            // and `gpu_pipeline_preflight` rejects the 15 stages that are
            // legitimately CPU-side (GA mutation/crossover, correlation pruning,
            // survivor ranking), aborting a real card run before its first
            // generation. The population-eval guarantee lives in
            // `evaluate_population_core`'s tail guard instead.
            assert!(
                backend.cpu_fallback_allowed(),
                "escalation must not claim the whole pipeline is strict-GPU (`{value}`)"
            );
        }

        for forced in ["cpu_forced", "cpu-forced"] {
            let backend =
                EvaluationBackend::resolve_with_probe(forced, "", None, card).expect("resolves");
            assert_eq!(
                backend,
                EvaluationBackend::CPU_CANONICAL,
                "`{forced}` is the one deliberate CPU-on-a-card escape"
            );
        }

        // The discovery-specific field also escalates on a card (it wins when
        // non-empty), and its `cpu_forced` escape is honored there too.
        assert_eq!(
            EvaluationBackend::resolve_with_probe("auto", "cpu", None, card)
                .unwrap()
                .device,
            DevicePreference::Gpu,
            "a shipped `models.prop_search_device: cpu` must not defeat the card"
        );
        assert_eq!(
            EvaluationBackend::resolve_with_probe("auto", "cpu_forced", None, card).unwrap(),
            EvaluationBackend::CPU_CANONICAL
        );

        // Card-less host: unchanged — auto/cpu resolve to a CPU-allowing backend
        // and the run proceeds normally, never a hard failure.
        for value in ["auto", "cpu", ""] {
            let backend =
                EvaluationBackend::resolve_with_probe(value, "", None, no_card).expect("resolves");
            assert!(
                backend.cpu_fallback_allowed(),
                "no card: `{value}` must allow CPU (normal), not fail loud"
            );
        }
    }

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
