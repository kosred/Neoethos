from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)

backend_path = Path("crates/neoethos-search/src/backend.rs")
backend = backend_path.read_text(encoding="utf-8")
backend = replace_once(
    backend,
    "use std::fmt;\n",
    "use std::fmt;\nuse std::sync::{OnceLock, RwLock};\n",
    "backend sync imports",
)

marker = "/// Transitional typed entry point. Commit 0.1 deliberately preserves the\n"
start = backend.index(marker)
end = backend.index("\n#[derive(Debug, Clone, PartialEq, Eq)]\npub enum BackendConfigError", start)
replacement = r'''fn backend_slot() -> &'static RwLock<EvaluationBackend> {
    static BACKEND: OnceLock<RwLock<EvaluationBackend>> = OnceLock::new();
    BACKEND.get_or_init(|| RwLock::new(EvaluationBackend::AUTO))
}

/// Install the resolved backend for production discovery. The lock is mutable so
/// tests and long-lived frontends may install a freshly loaded config before a new
/// work unit without relying on ambient process environment reads.
pub fn install_evaluation_backend(
    backend: EvaluationBackend,
) -> Result<(), BackendConfigError> {
    backend.validate()?;
    let mut slot = backend_slot().write().unwrap_or_else(|poisoned| poisoned.into_inner());
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
    *backend_slot().read().unwrap_or_else(|poisoned| poisoned.into_inner())
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
    Err("gpu_required was selected, but neoethos-search was compiled without a GPU backend feature".to_string())
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
    use crate::gpu_fallback::{
        GpuAction, GpuAttempt, GpuFailure, GpuRetryPolicy, decide_action,
    };

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
        return Err("gpu_required population evaluation received inconsistent gene CSR/SMC dimensions".to_string());
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
                return Err("gpu_required population evaluation received invalid CSR offsets".to_string());
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
                        format!("GPU returned {} rows for a {}-candidate batch", batch.len(), end - start),
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
                    return Err("internal policy error: gpu_required selected a CPU fallback".to_string());
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
'''
backend = backend[:start] + replacement + backend[end:]
backend_path.write_text(backend, encoding="utf-8")

lib_path = Path("crates/neoethos-search/src/lib.rs")
lib = lib_path.read_text(encoding="utf-8")
lib = replace_once(
    lib,
    "    AcceleratorHint, BackendConfigError, DevicePreference, EvaluationBackend, FallbackPolicy,\n    evaluate_population_core_with_backend,\n",
    "    AcceleratorHint, BackendConfigError, DevicePreference, EvaluationBackend, FallbackPolicy,\n    current_evaluation_backend, evaluate_population_core_with_backend,\n    evaluate_population_core_with_backend_and_audit, install_evaluation_backend,\n    install_evaluation_backend_from_settings,\n",
    "backend reexports",
)
lib = replace_once(
    lib,
    "pub fn install_search_runtime_overrides_from_settings(s: &neoethos_core::Settings) {\n",
    "pub fn install_search_runtime_overrides_from_settings(s: &neoethos_core::Settings) {\n    install_evaluation_backend_from_settings(s).unwrap_or_else(|error| {\n        panic!(\"invalid discovery evaluation backend configuration: {error}\")\n    });\n",
    "settings backend install",
)
lib_path.write_text(lib, encoding="utf-8")

search_path = Path("crates/neoethos-search/src/genetic/search_engine.rs")
search = search_path.read_text(encoding="utf-8")
fn_start = search.index("pub fn evaluate_genes_cached(")
fn_end = search.index("\n}\n\n/// AREA 2 / Stage A", fn_start) + 2
region = search[fn_start:fn_end]
region = replace_once(
    region,
    "    crate::eval::evaluate_population_core(crate::eval::PopulationEvalInputs {\n",
    "    crate::backend::evaluate_population_core_with_backend(crate::eval::PopulationEvalInputs {\n",
    "GA evaluator opening",
)
region = replace_once(
    region,
    "    })\n    .map_err(|e| anyhow!(e))\n",
    "    }, crate::backend::current_evaluation_backend())\n    .map_err(|e| anyhow!(e))\n",
    "GA evaluator backend argument",
)
search = search[:fn_start] + region + search[fn_end:]
search_path.write_text(search, encoding="utf-8")

cap_path = Path("crates/neoethos-search/src/gpu_native/capability.rs")
cap = cap_path.read_text(encoding="utf-8")
cap = replace_once(
    cap,
    '''            StageCapability {
                stage: S::PopulationEvaluation,
                capability: C::HybridOnly,
                detail: "CubeCL population kernels exist, but the production evaluator uses a CPU lane and CPU recompute",
            },
''',
    '''            StageCapability {
                stage: S::PopulationEvaluation,
                capability: C::StrictGpu,
                detail: "gpu_required routes the complete logical population through CubeCL with bounded GPU-only rebatching and no CPU fallback",
            },
''',
    "population capability",
)
cap = replace_once(
    cap,
    '''        assert!(error.unsupported.iter().any(|item| {
            item.stage == PipelineStage::PopulationEvaluation
                && item.capability == StageGpuCapability::HybridOnly
        }));
''',
    '''        assert!(!error
            .unsupported
            .iter()
            .any(|item| item.stage == PipelineStage::PopulationEvaluation));
''',
    "strict preflight population expectation",
)
cap_path.write_text(cap, encoding="utf-8")

discovery_path = Path("crates/neoethos-search/src/discovery.rs")
discovery = discovery_path.read_text(encoding="utf-8")
needle = "pub fn run_discovery_cycle_with_holdout_and_progress"
fn_index = discovery.index(needle)
brace = discovery.index("{", fn_index)
preflight = '''
    crate::gpu_native::capability::gpu_pipeline_preflight(
        crate::backend::current_evaluation_backend(),
        &crate::gpu_native::capability::GpuCapabilityManifest::stage1_baseline(),
        &crate::gpu_native::capability::PipelineStage::FULL_DISCOVERY,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
'''
if preflight.strip() in discovery:
    raise RuntimeError("discovery strict preflight already inserted")
discovery = discovery[: brace + 1] + preflight + discovery[brace + 1 :]
discovery_path.write_text(discovery, encoding="utf-8")

print("wired strict GPU backend, GA evaluator, capability manifest and discovery preflight")
