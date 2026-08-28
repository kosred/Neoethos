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
        fallback: FallbackPolicy::ForbidCpu,
        accelerator_hint: AcceleratorHint::Any,
    };

    pub const AUTO: Self = Self {
        device: DevicePreference::Auto,
        fallback: FallbackPolicy::ForbidCpu,
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
            "gpu" | "on" | "true" => Self::GPU_REQUIRED,
            "gpu_required" | "gpu-required" => Self::GPU_REQUIRED,
            "cuda" | "cuda_required" | "cuda-required" => Self::gpu_with(AcceleratorHint::Cuda),
            "wgpu" | "wgpu_required" | "wgpu-required" => Self::gpu_with(AcceleratorHint::Wgpu),
            "vulkan" | "vulkan_required" | "vulkan-required" => {
                Self::gpu_with(AcceleratorHint::Vulkan)
            }
            "rocm" | "rocm_required" | "rocm-required" => Self::gpu_with(AcceleratorHint::Rocm),
            "metal" | "metal_required" | "metal-required" => Self::gpu_with(AcceleratorHint::Metal),
            "dx12" | "dx12_required" | "dx12-required" => Self::gpu_with(AcceleratorHint::Dx12),
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
    /// .prop_search_device`). The setting remains descriptive configuration;
    /// only the run-bound native probe may authorize CUDA or CPU execution.
    pub fn from_settings_and_process_env(settings: &Settings) -> Result<Self, BackendConfigError> {
        Self::from_settings(settings, None)
    }

    pub fn cpu_fallback_allowed(self) -> bool {
        false
    }

    pub fn gpu_required(self) -> bool {
        self.device == DevicePreference::Gpu
    }

    pub fn validate(self) -> Result<(), BackendConfigError> {
        Ok(())
    }

    const fn gpu_with(hint: AcceleratorHint) -> Self {
        Self {
            device: DevicePreference::Gpu,
            fallback: FallbackPolicy::ForbidCpu,
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
    // The indicator lane in `neoethos-data` has the SAME question to answer —
    // "may this run silently compute on the CPU?" — and until now it answered
    // it from `NEOETHOS_REQUIRE_GPU` while this crate answered it from config.
    // `set_indicator_compute_policy` existed for exactly this and had ZERO
    // callers (`hpc_ta.rs:93`), so the seam the module documents as "where the
    // operator's Settings plug in" was never plugged in. It is now, from the
    // one place that already holds the resolved backend, so the two crates
    // cannot disagree about the same run.
    let policy = indicator_compute_policy_for_backend(backend);
    match neoethos_data::core::hpc_ta::set_indicator_compute_policy(policy) {
        Ok(()) => {}
        Err(active) if active == policy => {}
        Err(active) => {
            return Err(BackendConfigError::IndicatorPolicyAlreadyResolved {
                requested: policy,
                active,
            });
        }
    }
    // Install the mutable search backend only after the immutable feature lane
    // accepted the same policy. A conflict must not leave the two authorities
    // disagreeing after this function returns an error.
    install_evaluation_backend(backend)?;
    Ok(backend)
}

fn indicator_compute_policy_for_backend(
    backend: EvaluationBackend,
) -> neoethos_data::core::hpc_ta::IndicatorComputePolicy {
    if backend.device == DevicePreference::Gpu {
        neoethos_data::core::hpc_ta::IndicatorComputePolicy::GpuOnly
    } else {
        neoethos_data::core::hpc_ta::IndicatorComputePolicy::Auto
    }
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
    evaluate_population_core_with_backend_and_audit_inner(inputs, backend, &audit, None)
}

pub(crate) fn evaluate_population_core_with_backend_and_evidence(
    inputs: crate::eval::PopulationEvalInputs<'_>,
    backend: EvaluationBackend,
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> Result<Vec<[f64; 11]>, String> {
    let audit = crate::gpu_native::cpu_strategy::CpuStrategyAuditContext::production(0);
    evaluate_population_core_with_backend_and_audit_inner(inputs, backend, &audit, Some(evidence))
}

pub fn evaluate_population_core_with_backend_and_audit(
    inputs: crate::eval::PopulationEvalInputs<'_>,
    backend: EvaluationBackend,
    audit: &crate::gpu_native::cpu_strategy::CpuStrategyAuditContext,
) -> Result<Vec<[f64; 11]>, String> {
    crate::historical_evaluation_authority::require_historical_evaluation_authority_v1()
        .map_err(|error| error.to_string())?;
    evaluate_population_core_with_backend_and_audit_inner(inputs, backend, audit, None)
}

fn evaluate_population_core_with_backend_and_audit_inner(
    inputs: crate::eval::PopulationEvalInputs<'_>,
    backend: EvaluationBackend,
    audit: &crate::gpu_native::cpu_strategy::CpuStrategyAuditContext,
    evidence: Option<&crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>>,
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
    // Configuration does not authorize a device. The exact run-owned evidence
    // below is the sole dispatch authority, so a caller preference cannot turn
    // a visible card, a runtime fault, or a missing adapter into CPU work.
    let dispatch_started = std::time::Instant::now();
    let expected_output_rows = inputs.long_thr.len();
    let evidence = evidence.ok_or_else(|| {
        "population evaluation requires a run-bound sealed device route; refusing detached CPU/GPU dispatch"
            .to_string()
    })?;
    if let Ok(no_gpu_receipt) = evidence.require_cpu_route_receipt_v1() {
        let rows = cpu_strategy::run_with_sealed_no_gpu_receipt(
            no_gpu_receipt,
            audit,
            CpuStrategyCategory::PopulationEvaluation,
            || crate::eval::validation_backtest_population_cpu(inputs),
        );
        crate::eval_telemetry::record_device(
            "population_eval",
            crate::eval_telemetry::Device::Cpu,
            dispatch_started.elapsed(),
        );
        evidence
            .record_successful_population(
                crate::engine_identity::PopulationEvalEngine::Cpu,
                expected_output_rows,
                rows.len(),
            )
            .map_err(|error| error.to_string())?;
        return Ok(rows);
    }
    evidence
        .require_exact_cuda_device_ordinal_v1()
        .map_err(|error| error.to_string())?;
    let out = evaluate_gpu_required_population(inputs, backend, audit, Some(evidence));
    if out.is_ok() {
        crate::eval_telemetry::record_device(
            "population_eval",
            crate::eval_telemetry::Device::Gpu,
            dispatch_started.elapsed(),
        );
    }
    out
}

/// Unit-only backend oracle. It preserves device/fallback dispatch tests while
/// release callers remain unable to bypass the broker-real capability gate.
#[cfg(test)]
pub(crate) fn evaluate_population_core_with_backend_test_oracle(
    inputs: crate::eval::PopulationEvalInputs<'_>,
    backend: EvaluationBackend,
    audit: &crate::gpu_native::cpu_strategy::CpuStrategyAuditContext,
) -> Result<Vec<[f64; 11]>, String> {
    evaluate_population_core_with_backend_and_audit_inner(inputs, backend, audit, None)
}

#[cfg(not(feature = "gpu-b-adapter"))]
fn evaluate_gpu_required_population(
    _inputs: crate::eval::PopulationEvalInputs<'_>,
    _backend: EvaluationBackend,
    audit: &crate::gpu_native::cpu_strategy::CpuStrategyAuditContext,
    _evidence: Option<&crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>>,
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

#[cfg(feature = "gpu-b-adapter")]
fn evaluate_gpu_required_population(
    inputs: crate::eval::PopulationEvalInputs<'_>,
    backend: EvaluationBackend,
    audit: &crate::gpu_native::cpu_strategy::CpuStrategyAuditContext,
    evidence: Option<&crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>>,
) -> Result<Vec<[f64; 11]>, String> {
    use crate::strict_discovery_device_route_v1::{
        StrictNativeFailureActionV1, StrictNativeFailureKindV1,
    };

    // Only the native f64 engine is legal after an exact CUDA ordinal is sealed.
    // A missing adapter, different engine, or runtime fault is a loud refusal.
    // The run-owned fallible probe already proved one exact compatible ordinal.
    // Re-running the legacy lossy runtime/device-count probe here could collapse
    // a device fault into zero and would violate the one-probe authority.
    let readiness = crate::engine_identity::PrototypeBReadiness::Ready;
    let expected_engine = crate::engine_identity::strict_engine_preflight(backend, readiness)
        .map_err(|message| {
            let _ = audit.snapshot().assert_zero_executed();
            message
        })?;
    let evidence = evidence.ok_or_else(|| {
        "strict native population evaluation requires sealed exact run evidence".to_string()
    })?;
    let selected_ordinal = evidence
        .require_exact_cuda_device_ordinal_v1()
        .map_err(|error| error.to_string())?
        .selected_ordinal();
    if expected_engine != crate::engine_identity::PopulationEvalEngine::CudaNativeF64 {
        return Err("strict Discovery routing selected a non-native population engine".to_string());
    }

    // Name the engine before the work starts, so a log that ends in a crash
    // still says what arithmetic was in use.
    {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                target: "neoethos_search::backend",
                ?readiness,
                accelerator_hint = ?backend.accelerator_hint,
                expected_engine = expected_engine.as_str(),
                reproduces_canonical_cpu = expected_engine.reproduces_canonical_cpu(),
                "gpu_required population evaluation resolved its engine"
            );
        });
    }

    let crate::eval::PopulationEvalInputs {
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        gene_smc_flags,
        gate_threshold,
        weights,
        ..
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

    let mut retry_index = 0_u32;
    let max_retries = 4_u32;
    let mut current_batch_size = n_genes;

    loop {
        let batch_size = current_batch_size.clamp(1, n_genes);
        let mut output = Vec::with_capacity(n_genes);
        let mut failure: Option<(StrictNativeFailureKindV1, String)> = None;
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
                crate::gpu_native::prototype_b_population_eval::try_evaluate_population_b(
                    evidence,
                    &rebased_offsets,
                    &gene_indices[idx_start..idx_end],
                    &gene_weights[idx_start..idx_end],
                    &long_thr[start..end],
                    &short_thr[start..end],
                    &sl_pips[start..end],
                    &tp_pips[start..end],
                    stop_slice,
                    &gene_smc_flags[start..end],
                    gate_threshold,
                    weights,
                )
                .map_err(|error| error.to_string())
            }));

            match outcome {
                Ok(Ok(batch)) if batch.len() == end - start => output.extend(batch),
                Ok(Ok(batch)) => {
                    failure = Some((
                        StrictNativeFailureKindV1::WrongShape,
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
                    let kind = if lower.contains("out of memory")
                        || lower.contains("allocation")
                        || lower.contains("capacity")
                    {
                        StrictNativeFailureKindV1::AllocationPressure
                    } else if lower.contains("device lost") {
                        StrictNativeFailureKindV1::DeviceLost
                    } else {
                        StrictNativeFailureKindV1::Unsupported
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
                    failure = Some((StrictNativeFailureKindV1::Unsupported, detail));
                    break;
                }
            }
            start = end;
        }

        if let Some((kind, detail)) = failure {
            match crate::gpu_fallback::decide_strict_population_failure_v1(
                kind,
                selected_ordinal,
                batch_size,
                retry_index,
                max_retries,
            ) {
                StrictNativeFailureActionV1::RetrySameOrdinal {
                    selected_ordinal: retry_ordinal,
                    next_batch_size,
                } => {
                    if retry_ordinal != selected_ordinal {
                        return Err("strict rebatch changed the sealed CUDA ordinal".to_string());
                    }
                    tracing::warn!(
                        target: "neoethos_search::backend",
                        ?kind,
                        retry_index,
                        previous_batch_size = batch_size,
                        next_batch_size,
                        selected_ordinal,
                        error = %detail,
                        "strict native population batch hit allocation pressure; retrying on the same sealed ordinal"
                    );
                    retry_index = retry_index.saturating_add(1);
                    current_batch_size = next_batch_size;
                    continue;
                }
                StrictNativeFailureActionV1::FailLoud { .. } => {
                    return Err(format!(
                        "gpu_required population evaluation failed closed after {} retries: {:?}: {}",
                        retry_index, kind, detail
                    ));
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
    IndicatorPolicyAlreadyResolved {
        requested: neoethos_data::core::hpc_ta::IndicatorComputePolicy,
        active: neoethos_data::core::hpc_ta::IndicatorComputePolicy,
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
            Self::IndicatorPolicyAlreadyResolved { requested, active } => write!(
                f,
                "canonical feature compute policy is already resolved as {active:?}; refusing \
                 conflicting backend policy {requested:?}"
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
    fn configuration_is_descriptive_and_never_authorizes_cpu_substitution() {
        for value in ["", "auto", "cpu", "off", "false", "gpu", "cuda"] {
            let backend = EvaluationBackend::parse(value).expect("known configuration value");
            assert_eq!(backend.fallback, FallbackPolicy::ForbidCpu);
            assert!(!backend.cpu_fallback_allowed());
        }
    }

    #[test]
    fn gpu_device_also_forbids_silent_cpu_indicator_features() {
        use neoethos_data::core::hpc_ta::IndicatorComputePolicy;

        assert_eq!(
            indicator_compute_policy_for_backend(EvaluationBackend::GPU_REQUIRED),
            IndicatorComputePolicy::GpuOnly,
            "GPU-required evaluation must keep feature indicators on the same device"
        );
        assert_eq!(
            indicator_compute_policy_for_backend(EvaluationBackend::CPU_CANONICAL),
            IndicatorComputePolicy::Auto,
            "the explicit CPU lane remains valid"
        );
    }

    #[test]
    fn configuration_values_map_to_strict_descriptions() {
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
            EvaluationBackend::GPU_REQUIRED
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
        assert_eq!(inherited, EvaluationBackend::GPU_REQUIRED);
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
            assert_eq!(backend, EvaluationBackend::GPU_REQUIRED);
        }
    }

    #[test]
    fn invalid_boolean_fails_loud() {
        let error =
            EvaluationBackend::resolve_for_discovery("auto", "gpu", Some("maybe")).unwrap_err();
        assert!(matches!(error, BackendConfigError::InvalidBoolean { .. }));
    }
}
