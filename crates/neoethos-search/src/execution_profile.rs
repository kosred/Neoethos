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

// ─── RETIRED ENVIRONMENT VARIABLES ──────────────────────────────────────────
//
// 2026-08-10, the env→config wave. Every name below USED to change what this
// crate computed. None of them does any more: each is now either a typed value
// resolved from the single `Settings`, or a quantity derived from the probed
// hardware.
//
// THE FAILURE MODE THIS CLOSES is not "the env var exists". It is "the env var
// is exported on a box, the operator believes it is in force, and the run
// quietly means something else". A retired name that is still set in the shell
// is therefore not ignored quietly — it is reported at ERROR, by name, with the
// value that was found and the thing that decides instead.
//
// Retired names that still have a useful raw recorder appear in `raw_env(...)`
// captures below. Arithmetic switches that were deleted outright are reported
// at startup but do not retain a misleading active profile field.

/// `(env var, what decides it now)`. Production names only — test-gating names
/// (`NEOETHOS_REQUIRE_GPU` inside `#[cfg(test)]`, `FUSED_TEST_NSAMPLES`) are
/// deliberately absent because they still do exactly what they say.
/// `pub(crate)` so the env-knob census test can assert that anything it
/// classifies as `Retired` is genuinely declared retired here, rather than
/// letting the exemption become a place to park a knob that is still read.
pub(crate) const RETIRED_ENV_VARS: &[(&str, &str)] = &[
    // ── backend / GPU policy ──
    (
        "NEOETHOS_REQUIRE_GPU",
        "system.enable_gpu_preference / models.prop_search_device (use a *_required value); \
         with a card present the backend already escalates to GPU-preferred on its own",
    ),
    // ── cubecl lane selection ──
    (
        "NEOETHOS_BOT_SEARCH_EVAL_PRECISION",
        "the compiled search lanes are f64; no runtime precision switch remains",
    ),
    (
        "NEOETHOS_BOT_TRAIN_PRECISION",
        "the compiled lane; config field routed to config.rs",
    ),
    (
        "FOREX_TRAIN_PRECISION",
        "the compiled lane; config field routed to config.rs",
    ),
    (
        "NEOETHOS_GPU_F64",
        "CubeCL search arithmetic is unconditionally f64",
    ),
    (
        "NEOETHOS_BOT_SEARCH_EVAL_CUDA_KERNEL",
        "always on — the kernel is the lane",
    ),
    (
        "NEOETHOS_BOT_SEARCH_BACKTEST_CUDA_KERNEL",
        "always on — the kernel is the lane",
    ),
    (
        "NEOETHOS_BOT_SEARCH_EVAL_KERNEL_UNITS",
        "the launch geometry the kernel computes from the work size",
    ),
    (
        "NEOETHOS_BOT_SEARCH_BACKTEST_KERNEL_UNITS",
        "the launch geometry the kernel computes from the work size",
    ),
    (
        "NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE",
        "the per-lane device_override the scheduler passes (default device 0)",
    ),
    (
        "NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE",
        "the per-lane device_override the scheduler passes",
    ),
    (
        "NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICES",
        "the scheduler's per-process device assignment",
    ),
    (
        "NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICES",
        "the scheduler's per-process device assignment",
    ),
    (
        "NEOETHOS_BOT_SEARCH_USE_IGPU",
        "the hardware probe (an integrated GPU is detected, not declared)",
    ),
    (
        "NEOETHOS_GPU_FUSED_EVAL",
        "auto-detection from the probe + whether prototype B owns population eval",
    ),
    (
        "NEOETHOS_GPU_TIMING",
        "the DEBUG log level on target neoethos_search::gpu_timing",
    ),
    (
        "NEOETHOS_BOT_SEARCH_VRAM_LOG",
        "the DEBUG log level on target neoethos_search::cubecl_eval",
    ),
    // ── memory budgets: hardware, never a user parameter (never-OOM invariant) ──
    (
        "NEOETHOS_BOT_SEARCH_GPU_BUFFER_MB",
        "the device-probed per-buffer cap (auto_tune_memory_budgets)",
    ),
    (
        "NEOETHOS_BOT_SEARCH_VRAM_BUDGET_MB",
        "the probed VRAM budget (auto_tune_memory_budgets)",
    ),
    (
        "NEOETHOS_BOT_SEARCH_HOST_BUDGET_MB",
        "the probed host-RAM budget (auto_tune_memory_budgets)",
    ),
    // ── backtest arithmetic ──
    (
        "NEOETHOS_BOT_BACKTEST_INITIAL_EQUITY",
        "models.backtest_runtime.initial_equity",
    ),
    (
        "NEOETHOS_BOT_BACKTEST_MAX_MONTH_BUCKETS",
        "models.backtest_runtime.month_capacity",
    ),
    (
        "NEOETHOS_BOT_RUST_THREADS",
        "models.backtest_runtime.rayon_threads",
    ),
    (
        "RAYON_NUM_THREADS",
        "models.backtest_runtime.rayon_threads (this crate; tree_models still reads it)",
    ),
    // ── quality scoring ──
    (
        "NEOETHOS_BOT_PROP_MIN_TRADES_PER_MONTH",
        "models.quality_runtime.min_trades_per_month",
    ),
    (
        "NEOETHOS_BOT_TRADING_DAYS_PER_MONTH",
        "models.quality_runtime.trading_days_per_month",
    ),
    // ── seen-signature memory ──
    (
        "NEOETHOS_BOT_PROP_SEEN_FLUSH_EVERY",
        "models.seen_signature_runtime.flush_every",
    ),
    (
        "NEOETHOS_BOT_PROP_SEEN_LOAD_MAX",
        "models.seen_signature_runtime.load_max",
    ),
    (
        "NEOETHOS_BOT_PROP_SEEN_MAX_ENTRIES",
        "models.seen_signature_runtime.max_entries (0 now means DERIVE from RAM, not unbounded)",
    ),
    (
        "NEOETHOS_BOT_PROP_SEEN_FILE",
        "models.seen_signature_runtime.file_path",
    ),
    // ── adaptive stops ──
    (
        "NEOETHOS_ADAPTIVE_STOPS",
        "models.stop_target_runtime.adaptive_stops_enabled (config field routed to config.rs; \
         the typed default is ON)",
    ),
    (
        "NEOETHOS_ADAPTIVE_STOP_RR",
        "models.stop_target_runtime.adaptive_stops_rr (config field routed to config.rs; \
         the typed default is 2.0)",
    ),
    // ── SMC gene injection ──
    (
        "NEOETHOS_BOT_PROP_SMC_ENABLE_P",
        "models.smc_search_runtime.p_* (per-flag)",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_FORCE_RATIO",
        "models.smc_search_runtime.force_ratio",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_MIN_FLAGS",
        "models.smc_search_runtime.min_flags",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_FORCE_ENABLED",
        "models.smc_search_runtime.force_enabled",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_P_OB",
        "models.smc_search_runtime.p_ob",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_P_FVG",
        "models.smc_search_runtime.p_fvg",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_P_LIQ",
        "models.smc_search_runtime.p_liq",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_P_PREMIUM",
        "models.smc_search_runtime.p_premium",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_P_INDUCEMENT",
        "models.smc_search_runtime.p_inducement",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_P_MTF",
        "models.smc_search_runtime.p_mtf",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_P_BOS",
        "models.smc_search_runtime.p_bos",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_P_CHOCH",
        "models.smc_search_runtime.p_choch",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_P_EQH",
        "models.smc_search_runtime.p_eqh",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_P_EQL",
        "models.smc_search_runtime.p_eql",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_P_DISPLACEMENT",
        "models.smc_search_runtime.p_displacement",
    ),
    // ── feature cube (neoethos-data) ──
    (
        "NEOETHOS_FEATURE_CUBE_MODE",
        "models.data_runtime.feature_cube_mode (auto|disk), recorded in the run profile at \
         /execution/feature_cube_mode. There is no `ram` value: forcing RAM was the arm that \
         returned before the free-RAM check",
    ),
];

/// The GA-selection, cost-profile and SMC-weight names, retired with
/// `GeneticSearchRuntimeOverrides::from_env` and
/// `StrategyEvaluationRuntimeOverrides::from_env`.
///
/// Split out only because they share one replacement sentence each and listing
/// 44 near-identical rows above would bury the ones with a story. Reported
/// exactly like [`RETIRED_ENV_VARS`].
///
/// These are the most dangerous names in the whole set: the RNG SEED, the
/// parent/survivor selection policy, and the spread/commission/pip-value the
/// P&L is computed from. Every one of them could change what a run selected
/// while appearing in no config file and no artifact.
pub(crate) const RETIRED_SEARCH_ENV_VARS: &[(&str, &str)] = &[
    ("NEOETHOS_BOT_SEARCH_SEED", "models.search_runtime.seed"),
    (
        "NEOETHOS_BOT_NOVELTY_WEIGHT",
        "models.search_runtime.novelty_weight",
    ),
    (
        "NEOETHOS_BOT_PROP_STAGNATION_GENS",
        "models.search_runtime.stagnation_patience",
    ),
    (
        "NEOETHOS_BOT_PROP_CONVERGENCE_GENS",
        "models.search_runtime.convergence_patience",
    ),
    (
        "NEOETHOS_BOT_PROP_CONVERGENCE_MIN_ELAPSED_FRAC",
        "models.search_runtime.convergence_min_elapsed_fraction",
    ),
    (
        "NEOETHOS_BOT_PROP_MIN_IMPROVEMENT",
        "models.search_runtime.min_improvement",
    ),
    (
        "NEOETHOS_BOT_PROP_TOURNAMENT_SIZE",
        "models.search_runtime.tournament_size_override",
    ),
    (
        "NEOETHOS_BOT_PROP_ARCHIVE_CAP",
        "models.search_runtime.archive_cap_override",
    ),
    (
        "NEOETHOS_BOT_PROP_SEEN_RETRY",
        "models.search_runtime.seen_retry_attempts",
    ),
    (
        "NEOETHOS_BOT_PROP_ARCHIVE_MODE",
        "models.search_runtime.archive_scoring.mode",
    ),
    (
        "NEOETHOS_BOT_PROP_ARCHIVE_MIN_NET",
        "models.search_runtime.archive_scoring.min_net",
    ),
    (
        "NEOETHOS_BOT_PROP_ARCHIVE_MIN_PF",
        "models.search_runtime.archive_scoring.min_pf",
    ),
    (
        "NEOETHOS_BOT_PROP_ARCHIVE_MIN_SHARPE",
        "models.search_runtime.archive_scoring.min_sharpe",
    ),
    (
        "NEOETHOS_BOT_PROP_PARENT_SELECTION",
        "models.search_runtime.selection.parent",
    ),
    (
        "NEOETHOS_BOT_PROP_SURVIVOR_SELECTION",
        "models.search_runtime.selection.survivor",
    ),
    (
        "NEOETHOS_BOT_PROP_RANDOM_IMMIGRANTS",
        "models.search_runtime.selection.immigrant_ratio",
    ),
    (
        "NEOETHOS_BOT_PROP_SURVIVOR_FRACTION",
        "models.search_runtime.selection.survivor_fraction",
    ),
    (
        "NEOETHOS_BOT_PROP_ELITE_FRACTION",
        "models.search_runtime.selection.survivor_fraction (the older spelling)",
    ),
    (
        "NEOETHOS_BOT_PROP_SELECTION_TEMPERATURE",
        "models.search_runtime.selection.temperature",
    ),
    (
        "NEOETHOS_BOT_DISABLE_SMC_GATE",
        "models.search_runtime.smc_gate.disable_gate",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_GATE_START",
        "models.search_runtime.smc_gate.start",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_GATE_END",
        "models.search_runtime.smc_gate.end",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_GATE_CURVE",
        "models.search_runtime.smc_gate.curve",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_GATE_STAGNATION_STEP",
        "models.search_runtime.smc_gate.stagnation_step",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_GATE",
        "models.eval_runtime.smc_gate_threshold (and the search_runtime curve)",
    ),
    ("NEOETHOS_BOT_PROP_SYMBOL", "system.symbol"),
    (
        "NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY",
        "risk.account_currency / system.account_currency",
    ),
    (
        "NEOETHOS_BOT_PROP_PIP_VALUE",
        "models.eval_runtime.pip_value",
    ),
    (
        "NEOETHOS_BOT_PROP_PIP_VALUE_PER_LOT",
        "models.eval_runtime.pip_value_per_lot",
    ),
    (
        "NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE",
        "models.eval_runtime.quote_to_account_rate",
    ),
    (
        "NEOETHOS_BOT_PROP_SPREAD_PIPS",
        "risk.backtest_spread_pips (the eval_runtime copy never wins)",
    ),
    (
        "NEOETHOS_BOT_PROP_COMMISSION",
        "risk.commission_per_lot (the eval_runtime copy never wins)",
    ),
    (
        "NEOETHOS_BOT_REJECT_PIP_FALLBACK",
        "models.eval_runtime.reject_pip_fallback",
    ),
    ("NEOETHOS_BOT_PROP_SMC_W_OB", "models.eval_runtime.smc_w_ob"),
    (
        "NEOETHOS_BOT_PROP_SMC_W_FVG",
        "models.eval_runtime.smc_w_fvg",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_W_LIQ",
        "models.eval_runtime.smc_w_liq",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_W_PREMIUM",
        "models.eval_runtime.smc_w_premium",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_W_INDUCEMENT",
        "models.eval_runtime.smc_w_inducement",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_W_MTF",
        "models.eval_runtime.smc_w_mtf",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_W_BOS",
        "models.eval_runtime.smc_w_bos",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_W_CHOCH",
        "models.eval_runtime.smc_w_choch",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_W_EQH",
        "models.eval_runtime.smc_w_eqh",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_W_EQL",
        "models.eval_runtime.smc_w_eql",
    ),
    (
        "NEOETHOS_BOT_PROP_SMC_W_DISPLACEMENT",
        "models.eval_runtime.smc_w_displacement",
    ),
];

/// Report, once per process and at ERROR, every retired environment variable
/// that is still exported — by name, with the value found, and with what
/// decides that quantity now.
///
/// Called from [`crate::eval::install_backtest_runtime_overrides_from_settings`],
/// which every production binary reaches through
/// `install_search_runtime_overrides_from_settings` at startup. There is no
/// second call site by design: a substitution announced twice is a substitution
/// nobody reads.
pub fn report_retired_env_vars() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut found = 0usize;
        for (name, replacement) in RETIRED_ENV_VARS.iter().chain(RETIRED_SEARCH_ENV_VARS) {
            let Ok(value) = std::env::var(name) else {
                continue;
            };
            if value.trim().is_empty() {
                continue;
            }
            found += 1;
            tracing::error!(
                target: "neoethos_search::retired_env",
                env_var = %name,
                value_found = %value,
                decided_by = %replacement,
                "RETIRED ENVIRONMENT VARIABLE IS SET AND WAS IGNORED — this value did NOT \
                 reach the run. Remove it from the shell and set the named config key instead."
            );
        }
        if found > 0 {
            tracing::error!(
                target: "neoethos_search::retired_env",
                count = found,
                "{found} retired NEOETHOS/FOREX environment variable(s) were set and ignored. \
                 The environment no longer configures this crate: one config, no env."
            );
        }
    });
}

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
    /// Where the multi-timeframe feature cube was assembled —
    /// `models.data_runtime.feature_cube_mode`, as installed in this process
    /// (`"auto"` = derive from the free-RAM probe, `"disk"` = always stream).
    ///
    /// Recorded because the RAM and disk assemblies are only bit-identical
    /// BY TEST, not by construction: if they ever diverge, this is the field
    /// that says which one produced the cube a given run searched over. It
    /// replaces `NEOETHOS_FEATURE_CUBE_MODE`, which could move the same
    /// decision from a shell with no trace in any artifact.
    pub feature_cube_mode: String,
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
            smc_search: crate::genetic::smc_indicators::SmcSearchConfig::current(),
            adaptive_stops_enabled: crate::stop_target::adaptive_stops_enabled(),
            adaptive_stops_rr: crate::stop_target::adaptive_stops_rr(),
            feature_cube_mode: neoethos_data::current_feature_cube_policy()
                .as_str()
                .to_string(),
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

/// Record an ambient environment variable into the run profile.
///
/// KEPT ON PURPOSE, and it is the only `env::var` in this crate outside the
/// retired-env reporter and `#[cfg(test)]`. It is a RECORDER, not a reader:
/// nothing branches on what it returns, so it cannot change what a run
/// computes. Since 2026-08-10 every name below is retired, which makes these
/// fields strictly more useful than before — a profile that shows
/// `require_gpu_env: Some("1")` next to a CPU-fallback run is the evidence
/// that a stale export was present and correctly ignored.
///
/// It also keeps the retired names present in this crate's source, which is
/// what the `discovery_tests.rs` env census matches its table against.
fn raw_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
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
