//! Execution-environment snapshot for the discovery run profile.
//!
//! SLICE 5 of the search-correctness campaign (2026-08-08): if two
//! identical-config runs differ and nobody can tell WHY, every experiment is
//! unfalsifiable. The [`DiscoveryRunProfile`](crate::discovery::DiscoveryRunProfile)
//! already records the *config* the run was asked to use; this module records
//! the *ambient process state* that ALSO changes what the search selects —
//! RNG seed and GA selection policy, cost/SMC evaluation overrides, backtest
//! arithmetic knobs, thread counts, the adaptive-stop switch, the persistent
//! seen-signature memory (cross-RUN state!), and the whole GPU lane choice
//! (engine, precision, fused eval, kernel toggles, memory budgets).
//!
//! THE RULE this module enforces: every value is captured through the SAME
//! accessor the engine reads (`current_*_runtime_overrides()`, the memoised
//! env registries, OnceLock peeks). The profile can therefore not disagree
//! with the engine about what ran. Where a decision is memoised lazily (fused
//! eval, memory budgets), capture PEEKS and records `None` for "never
//! consulted" rather than forcing the resolution — capturing a profile must
//! never launch GPU work or install budgets as a side effect.
//!
//! The census test in `discovery_tests.rs`
//! (`every_env_knob_is_classified_and_recorded_in_the_run_profile`) scans this
//! crate's sources for env-var names and fails when a knob exists that is
//! neither recorded here (with a verified JSON pointer) nor explicitly
//! classified as diagnostic-only. That is the ratchet that keeps this snapshot
//! complete as knobs are added.

use serde::Serialize;

/// Which compute lane the population evaluation could take, and every ambient
/// knob that picks between them or changes their arithmetic.
///
/// All fields exist on every build (CPU-only included) so the serialized JSON
/// shape is stable; GPU-only facts are `None` when the crate was compiled
/// without the `gpu` feature or when the corresponding decision was never
/// consulted in this process.
#[derive(Debug, Clone, Serialize)]
pub struct GpuLaneProfile {
    /// `cfg!(feature = "gpu")` — was any GPU lane compiled in at all?
    pub compiled_gpu: bool,
    pub compiled_gpu_cuda: bool,
    pub compiled_gpu_vulkan: bool,
    /// Resolved evaluation backend policy (device / fallback / accelerator),
    /// from [`crate::backend::current_evaluation_backend`] — the exact value
    /// `evaluate_population_core_with_backend` dispatches on.
    pub backend_device: String,
    pub backend_fallback: String,
    pub backend_accelerator: String,
    /// Raw `NEOETHOS_REQUIRE_GPU` as seen by this process (escalates the
    /// backend to GPU_REQUIRED and disables the CPU fallback lane).
    pub require_gpu_env: Option<String>,
    /// f64 GPU backtest lane (`NEOETHOS_GPU_F64`); `None` on non-GPU builds.
    pub gpu_f64_backtest: Option<bool>,
    /// The fused VRAM-resident eval decision this process actually made.
    /// `Some(x)` = decided (env override or auto-probe); `None` = never
    /// consulted (or non-GPU build), so it cannot have influenced the run.
    pub fused_eval_decision: Option<bool>,
    /// CUDA env-knob registry (precision, kernel toggles, unit overrides,
    /// device id) — `None` on non-GPU builds.
    pub cuda_precision: Option<String>,
    pub cuda_eval_kernel_enabled: Option<bool>,
    pub cuda_backtest_kernel_enabled: Option<bool>,
    pub cuda_eval_kernel_units: Option<u32>,
    pub cuda_backtest_kernel_units: Option<u32>,
    pub cuda_device_id: Option<usize>,
    /// Installed hardware memory budgets (host / VRAM / per-buffer, MB) —
    /// `None` when the auto-tuner never ran in this process. These bound the
    /// GPU windowing and can demote work to the CPU lane.
    pub host_budget_mb: Option<u64>,
    pub vram_budget_mb: Option<u64>,
    pub gpu_buffer_mb: Option<usize>,
    /// Raw device/budget env overrides as seen by this process. Recorded raw
    /// because their resolvers live inside cfg-gated GPU code with
    /// per-call-site defaults; the raw value is what those resolvers see.
    pub wgpu_device_env: Option<String>,
    pub multi_wgpu_devices_env: Option<String>,
    pub multi_cuda_devices_env: Option<String>,
    pub use_igpu_env: Option<String>,
    pub gpu_buffer_mb_env: Option<String>,
    pub vram_budget_mb_env: Option<String>,
    pub host_budget_mb_env: Option<String>,
}

/// Snapshot of every ambient (process-wide) setting that can change what the
/// search SELECTS, captured at profile-build time. See the module docs for
/// the completeness contract.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionEnvironmentProfile {
    /// `neoethos-search` crate version that produced this profile.
    pub crate_version: String,
    /// GA selection knobs: seed, novelty, patience, tournament/archive,
    /// SMC gate curve, archive scoring, parent/survivor selection policy.
    pub genetic_search: crate::genetic::runtime_overrides::GeneticSearchRuntimeOverrides,
    /// Evaluation-time cost profile + SMC weights.
    pub strategy_eval: crate::genetic::runtime_overrides::StrategyEvaluationRuntimeOverrides,
    /// Canonical backtest arithmetic (initial equity, month buckets) and the
    /// rayon thread override.
    pub backtest: crate::eval::BacktestRuntimeOverrides,
    /// Quality-screen monthly aggregation knobs.
    pub quality: crate::quality::QualityRuntimeOverrides,
    /// Adaptive-stop expected-shortfall caps (tail_max_bars / tail_step).
    pub stop_target: crate::stop_target::StopTargetRuntimeOverrides,
    /// Seen-signature memory — CROSS-RUN state. When `file_path` is set, a
    /// persisted signature file seeds this run's dedup memory, so two
    /// identical-config runs can legitimately generate different candidates.
    /// A reproduction attempt must clear or pin this.
    pub seen_memory: crate::genetic::evolution_math::SeenSignatureMemoryRuntimeOverrides,
    /// SMC gene-injection probabilities used by random gene generation.
    pub smc_search: crate::genetic::smc_indicators::SmcSearchConfig,
    /// Adaptive volatility-scaled stops master switch (default ON;
    /// `NEOETHOS_ADAPTIVE_STOPS=0` is the escape hatch) and its reward:risk.
    pub adaptive_stops_enabled: bool,
    pub adaptive_stops_rr: f64,
    /// Threads the global rayon pool is actually running with. Thread count
    /// changes scheduling; recorded so a cross-machine reproduction can pin it.
    pub effective_rayon_threads: usize,
    /// The host's logical core count as std reports it (`None` if unknown).
    pub available_parallelism: Option<usize>,
    /// Compute-lane facts: backend policy, precision lanes, kernels, budgets.
    pub gpu: GpuLaneProfile,
}

impl ExecutionEnvironmentProfile {
    /// Capture the ambient execution environment through the same accessors
    /// the engine uses. Cheap and side-effect free: no GPU probe, no budget
    /// install, no pool construction beyond what discovery already did.
    pub fn capture() -> Self {
        let backend = crate::backend::current_evaluation_backend();
        Self {
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            genetic_search:
                crate::genetic::runtime_overrides::current_genetic_search_runtime_overrides(),
            strategy_eval:
                crate::genetic::runtime_overrides::current_strategy_evaluation_runtime_overrides(),
            backtest: crate::eval::current_backtest_runtime_overrides(),
            quality: crate::quality::current_quality_runtime_overrides(),
            stop_target: crate::stop_target::current_stop_target_runtime_overrides(),
            seen_memory:
                crate::genetic::evolution_math::current_seen_signature_memory_runtime_overrides(),
            smc_search: crate::genetic::smc_indicators::SmcSearchConfig::from_env(),
            adaptive_stops_enabled: crate::stop_target::adaptive_stops_enabled(),
            adaptive_stops_rr: crate::stop_target::adaptive_stops_rr(),
            effective_rayon_threads: rayon::current_num_threads(),
            available_parallelism: std::thread::available_parallelism().ok().map(|n| n.get()),
            gpu: GpuLaneProfile {
                compiled_gpu: cfg!(feature = "gpu"),
                compiled_gpu_cuda: cfg!(feature = "gpu-cuda"),
                compiled_gpu_vulkan: cfg!(feature = "gpu-vulkan"),
                backend_device: format!("{:?}", backend.device),
                backend_fallback: format!("{:?}", backend.fallback),
                backend_accelerator: format!("{:?}", backend.accelerator_hint),
                require_gpu_env: raw_env("NEOETHOS_REQUIRE_GPU"),
                gpu_f64_backtest: gpu_f64_backtest(),
                fused_eval_decision: fused_eval_decision(),
                cuda_precision: cuda_knobs().map(|k| k.0),
                cuda_eval_kernel_enabled: cuda_knobs().map(|k| k.1),
                cuda_backtest_kernel_enabled: cuda_knobs().map(|k| k.2),
                cuda_eval_kernel_units: cuda_knobs().and_then(|k| k.3),
                cuda_backtest_kernel_units: cuda_knobs().and_then(|k| k.4),
                cuda_device_id: cuda_knobs().map(|k| k.5),
                host_budget_mb: memory_budgets().map(|b| b.0),
                vram_budget_mb: memory_budgets().map(|b| b.1),
                gpu_buffer_mb: memory_budgets().map(|b| b.2),
                wgpu_device_env: raw_env("NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE"),
                multi_wgpu_devices_env: raw_env("NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICES"),
                multi_cuda_devices_env: raw_env("NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICES"),
                use_igpu_env: raw_env("NEOETHOS_BOT_SEARCH_USE_IGPU"),
                gpu_buffer_mb_env: raw_env("NEOETHOS_BOT_SEARCH_GPU_BUFFER_MB"),
                vram_budget_mb_env: raw_env("NEOETHOS_BOT_SEARCH_VRAM_BUDGET_MB"),
                host_budget_mb_env: raw_env("NEOETHOS_BOT_SEARCH_HOST_BUDGET_MB"),
            },
        }
    }
}

fn raw_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

#[cfg(feature = "gpu")]
fn gpu_f64_backtest() -> Option<bool> {
    Some(crate::cubecl_eval::gpu_f64_backtest_enabled())
}

#[cfg(not(feature = "gpu"))]
fn gpu_f64_backtest() -> Option<bool> {
    None
}

#[cfg(feature = "gpu")]
fn fused_eval_decision() -> Option<bool> {
    crate::cubecl_eval::fused_eval_decision_peek()
}

#[cfg(not(feature = "gpu"))]
fn fused_eval_decision() -> Option<bool> {
    None
}

/// `(precision, eval_kernel, backtest_kernel, eval_units, backtest_units, device_id)`
#[cfg(feature = "gpu")]
#[allow(clippy::type_complexity)]
fn cuda_knobs() -> Option<(String, bool, bool, Option<u32>, Option<u32>, usize)> {
    let k = crate::cubecl_eval::cuda_knobs_for_profile();
    Some((
        k.precision,
        k.eval_kernel_enabled,
        k.backtest_kernel_enabled,
        k.eval_kernel_units,
        k.backtest_kernel_units,
        k.cuda_device_id,
    ))
}

#[cfg(not(feature = "gpu"))]
#[allow(clippy::type_complexity)]
fn cuda_knobs() -> Option<(String, bool, bool, Option<u32>, Option<u32>, usize)> {
    None
}

#[cfg(feature = "gpu")]
fn memory_budgets() -> Option<(u64, u64, usize)> {
    crate::cubecl_eval::memory_budgets_for_profile()
}

#[cfg(not(feature = "gpu"))]
fn memory_budgets() -> Option<(u64, u64, usize)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_matches_the_engine_accessors() {
        let profile = ExecutionEnvironmentProfile::capture();
        // The profile must report EXACTLY what the engine accessors report —
        // same OnceLock, same defaults. (The census test in discovery_tests.rs
        // checks coverage; this checks fidelity.)
        assert_eq!(
            profile.genetic_search,
            crate::genetic::runtime_overrides::current_genetic_search_runtime_overrides()
        );
        assert_eq!(
            profile.strategy_eval,
            crate::genetic::runtime_overrides::current_strategy_evaluation_runtime_overrides()
        );
        assert_eq!(
            profile.backtest,
            crate::eval::current_backtest_runtime_overrides()
        );
        assert_eq!(
            profile.quality,
            crate::quality::current_quality_runtime_overrides()
        );
        assert_eq!(
            profile.stop_target,
            crate::stop_target::current_stop_target_runtime_overrides()
        );
        assert_eq!(
            profile.seen_memory,
            crate::genetic::evolution_math::current_seen_signature_memory_runtime_overrides()
        );
        assert!(profile.effective_rayon_threads >= 1);
    }

    #[test]
    fn cpu_only_builds_serialize_gpu_facts_as_null_not_absent() {
        // The JSON SHAPE must be identical across CPU-only and GPU builds so
        // profile diffs between machines compare the VALUES, not the schema.
        let profile = ExecutionEnvironmentProfile::capture();
        let json = serde_json::to_value(&profile).expect("profile must serialize");
        for pointer in [
            "/gpu/gpu_f64_backtest",
            "/gpu/fused_eval_decision",
            "/gpu/cuda_precision",
            "/gpu/vram_budget_mb",
        ] {
            assert!(
                json.pointer(pointer).is_some(),
                "GPU profile field {pointer} must exist (as null when unknown) on every build"
            );
        }
    }
}
