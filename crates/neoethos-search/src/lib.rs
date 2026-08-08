mod artifact_io;
// `pub mod challenge;` — DELETED 2026-05-26 (operator directive: dual-mode product).
// `ChallengeOptimizer` had zero callers in the workspace; the prop-firm risk
// allocation it scaffolded is handled downstream by the prop-firm validation
// gates in `crates/neoethos-search/src/discovery.rs` + dual-mode separation
// (PropFirm vs Risky). 161 LOC removed.
pub mod checkpoint;
// SL/TP-faithful CUDA eval/backtest kernel via cubecl 0.9. The CPU
// fallback ships as the default no-gpu build — `evaluate_population_core`
// in `eval.rs` routes through this kernel when the `gpu` feature is on
// and through the CPU loop otherwise. The legacy `discovery_gpu`,
// `hpc_gpu_discovery`, `hpc`, and `cubecl_ga` modules were removed in
// the 2026-05-24 audit (F-070, F-077, F-085, F-092, F-094) — they were
// ~3500 LOC of feature-gated orphan code targeting a single cloud
// instance (Hyperstack N3) with zero external callers, plus a synthetic
// 0.0002 cost violation.
#[cfg(feature = "gpu")]
mod cubecl_eval;
// Pure CSR population partitioning for multi-GPU sharding (Stage 2). Not GPU-
// gated: it is plain slice math, so it compiles + unit-tests on any build. The
// device-execution glue that consumes it lives in `eval.rs` behind `gpu`.
pub mod discovery;
pub mod discovery_ledger;
// `mod scheduler_assignment;` — DELETED 2026-05-25 (verbose-build pass):
// the file was a 19-LOC orphan with zero callers. The scheduler-driven
// GPU routing it scaffolded is dispatched directly via `BackendKind`
// matching at the `cubecl_eval` boundary; the conversion helper this
// module shipped was never wired. If the scheduler-driven routing
// lands later, reintroduce a fresh helper at that time.

pub mod backend;
pub mod eval;
pub mod eval_telemetry;
pub mod export_state;
pub mod fx_rates;
pub mod funnel_profile;
pub mod gpu_fallback;
pub mod gpu_native;
// `pub mod gauntlet;` — DELETED 2026-05-26 (operator directive: dual-mode product).
// `StrategyGauntlet` had zero callers in the workspace; the quality floors
// (win-rate, profit-factor, drawdown caps) it scaffolded are now enforced by
// `FilteringConfig` in `genetic::strategy_gene` + the prop-firm validation
// gates in `discovery.rs`. 194 LOC removed.
pub mod genetic;
pub mod live_portfolio;
pub mod orchestration;
pub mod parity;
pub mod portfolio;
pub mod quality;
// **Scoring unification — Phase A (operator-approved 2026-05-25)**
// Shared "ingredient" functions + four canonical named scoring formulas
// (ga_fitness / archive_score / window_score / quality_score). The
// audit's six divergent scoring functions migrate to this layout in
// Phase B; Phase C unifies the weight tables (gated by
// `ScoringVersion` bump to 2). See `scoring/mod.rs` for the full
// migration plan + doctrine references.
pub mod scoring;
// **Regime classifier consolidation — Phase A (operator-approved 2026-05-25)**
// F-013 + F-048 + F-064 cluster: three divergent regime systems
// (feature-bucket, time-window, ADX/Hurst/EMA) consolidate onto the
// F-064 cascade as canonical. Phase B migrates the other two callers
// to consume `regime::infer_regime_canonical` + the typed `Regime`
// enum. See `regime/mod.rs` for the migration plan.
pub mod regime;
pub mod stop_target;
#[cfg(feature = "strategy-db")]
pub mod strategy_db;
pub mod validation;

pub use backend::{
    AcceleratorHint, BackendConfigError, DevicePreference, EvaluationBackend, FallbackPolicy,
    current_evaluation_backend, evaluate_population_core_with_backend,
    evaluate_population_core_with_backend_and_audit, install_evaluation_backend,
    install_evaluation_backend_from_settings,
};
// `pub use challenge::{ChallengeOptimizer, ChallengeTarget};` — DELETED 2026-05-26.
pub use discovery::{
    DEFAULT_OOS_HOLDOUT_FRACTION, DiscoveryConfig, DiscoveryPerKindEvidenceHashes,
    DiscoveryProgress, DiscoveryResult, DiscoveryRunProfile, DiscoveryRuntimeOverrides,
    DiscoveryValidationGates, GeneOosResult, LoggedStrategyTrades, Stage1Window,
    build_discovery_profile, compute_discovery_forward_test_artifacts,
    compute_discovery_prop_firm_artifacts, discovery_per_kind_evidence_hashes,
    discovery_validation_evidence_manifest,
    discovery_validation_evidence_manifest_excluding_live_sim, ensure_non_empty_portfolio,
    ensure_portfolio_export_ready, faithful_oos_eval, live_validation_evidence_from_discovery,
    run_discovery_cycle, run_discovery_cycle_with_holdout,
    run_discovery_cycle_with_holdout_and_progress, run_discovery_cycle_with_progress,
    save_canonical_backtest_artifacts, save_discovery_profile_json,
    save_forward_test_validation_artifacts, save_funnel_json, save_portfolio_json,
    save_promotion_summary_json, save_prop_firm_validation_artifacts, save_quality_report_json,
    save_trade_log_json, save_walkforward_validation_artifacts,
};
pub use discovery_ledger::{
    DiscoverySearchLedger, GeneRecord, SearchMetadata, ledger_path, load_prior_ledger,
    save_discovery_ledger, seed_seen_from_ledger,
};
pub use eval::{
    BacktestMetrics, BacktestRuntimeOverrides, BacktestSettings,
    current_backtest_runtime_overrides, evaluate_population_core, fast_evaluate_strategy_core,
    install_backtest_runtime_overrides, install_backtest_runtime_overrides_from_env,
    install_backtest_runtime_overrides_from_settings, simulate_trades_core,
};
// `pub use gauntlet::{GauntletConfig, StrategyGauntlet};` — DELETED 2026-05-26.
pub use genetic::{
    ArchiveScoringOverrides, CostProfileRuntimeOverrides, EvaluationConfig, EvolutionSearchPolicy,
    FilteringConfig, Gene, GeneticSearchRuntimeOverrides, ParentSelectionPolicy, SearchResult,
    SeenSignatureMemoryRuntimeOverrides, SelectionPolicyOverrides, SmcGateOverrides,
    SmcWeightRuntimeOverrides, StrategyEvaluationRuntimeOverrides, SurvivorSelectionPolicy,
    current_determinism_policy, current_genetic_search_runtime_overrides,
    current_seen_signature_memory_runtime_overrides, current_strategy_evaluation_runtime_overrides,
    default_pip_size, evaluate_genes, evolve_search, evolve_search_with_progress,
    evolve_search_with_progress_and_limits, install_genetic_search_runtime_overrides,
    install_genetic_search_runtime_overrides_from_env,
    install_genetic_search_runtime_overrides_from_settings,
    install_seen_signature_memory_runtime_overrides,
    install_seen_signature_memory_runtime_overrides_from_env,
    install_seen_signature_memory_runtime_overrides_from_settings,
    install_smc_search_config_from_env, install_smc_search_config_from_settings,
    install_strategy_evaluation_runtime_overrides,
    install_strategy_evaluation_runtime_overrides_from_env,
    install_strategy_evaluation_runtime_overrides_from_settings, migration_enabled,
    month_day_indices, push_migrants, random_search, set_migration_enabled, set_search_cancel,
    signals_for_gene, signals_for_gene_full, take_elites,
};
pub use live_portfolio::{
    LIVE_PORTFOLIO_SCHEMA_VERSION, LivePortfolioArtifact, load_live_portfolio_json,
    project_features_to_effective, save_live_portfolio_json,
};
pub use neoethos_core::contracts::DeterminismPolicy;
pub use orchestration::{BatchDiscoverySummary, DiscoveryOrchestrator};
pub use portfolio::{AllocationResult, PortfolioOptimizer, SymbolMetrics};
pub use quality::{
    QualityRuntimeOverrides, StrategyMetrics, StrategyQualityAnalyzer, StrategyRanker, Trade,
    current_quality_runtime_overrides, install_quality_runtime_overrides,
    install_quality_runtime_overrides_from_env, install_quality_runtime_overrides_from_settings,
};
pub use stop_target::{
    StopDistanceError, StopTargetRuntimeOverrides, StopTargetSettings, adaptive_base_pips_series,
    adaptive_sl_tp_pips_series, adaptive_stops_enabled, adaptive_stops_rr,
    compute_stop_distance_series, current_stop_target_runtime_overrides, infer_stop_target_pips,
    install_stop_target_runtime_overrides_from_settings,
};
pub use validation::{
    CANONICAL_BACKTEST_ARTIFACT_KIND, CANONICAL_BACKTEST_SCHEMA_VERSION,
    CanonicalBacktestArtifactFile, CanonicalBacktestScope, CombinatorialPurgedCV,
    FORWARD_TEST_VALIDATION_ARTIFACT_KIND, FORWARD_TEST_VALIDATION_SCHEMA_VERSION,
    ForwardTestInput, ForwardTestSummary, ForwardTestValidationArtifactFile,
    ForwardTestValidationScope, LIVE_EXECUTION_SIMULATION_ARTIFACT_KIND,
    LIVE_EXECUTION_SIMULATION_SCHEMA_VERSION, LiveExecutionRuntimeModel,
    LiveExecutionSimulationArtifactFile, LiveExecutionSimulationScope,
    LiveExecutionSimulationSummary, PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND,
    PROP_FIRM_RISK_VALIDATION_SCHEMA_VERSION, PropFirmRiskInput, PropFirmRiskRules,
    PropFirmRiskValidationArtifactFile, PropFirmRiskValidationScope, PropFirmRiskValidationSummary,
    WALKFORWARD_VALIDATION_ARTIFACT_KIND, WALKFORWARD_VALIDATION_SCHEMA_VERSION,
    WalkforwardSplitResult, WalkforwardSummary, WalkforwardValidationArtifactFile,
    WalkforwardValidationScope, compute_forward_test_summary, compute_prop_firm_risk_summary,
    embargoed_walkforward_backtest, read_canonical_backtest_artifact,
    read_forward_test_validation_artifact, read_live_execution_simulation_artifact,
    read_prop_firm_risk_validation_artifact, read_walkforward_validation_artifact,
    write_canonical_backtest_artifact_atomic, write_forward_test_validation_artifact_atomic,
    write_live_execution_simulation_artifact_atomic,
    write_prop_firm_risk_validation_artifact_atomic, write_walkforward_validation_artifact_atomic,
};

// REMOVED 2026-08-03: `install_search_runtime_overrides_from_env()`.
//
// Its doc claimed "Production binaries (`neoethos-cli`, `neoethos-app`) invoke
// this once at startup". They never did — it had ZERO callers, while
// `knob_catalog.rs` went on advertising ~35 `NEOETHOS_BOT_*` switches to the UI
// as "still honoured for backward compatibility". Setting any of them did
// nothing, and the operator was told otherwise by his own program.
//
// `install_search_runtime_overrides_from_settings` below strictly supersedes it:
// it covers all six boundaries (each marked ✓ config) and additionally installs
// the evaluation backend, which the env version never did. It IS called —
// neoethos-app/src/lib.rs:29, neoethos-cli/src/main.rs:33, and the discovery
// probe. The six `*_from_env` children remain; they are still reachable from
// tests, and retiring them belongs with the wider env→config migration.
//
// This is the sixth "mechanism exists, no production caller" defect closed on
// this branch. The pattern never crashes; the run just quietly means less than
// it claims.

/// Config-driven entry point — installs every typed runtime-override boundary
/// from the single [`neoethos_core::Settings`], plus the evaluation backend.
///
/// All six boundaries read config; the ✓ markers below are the record of that
/// migration completing. (This doc previously said "the remaining five still
/// read env until their migration lands", which the ✓ markers two lines down
/// already contradicted — corrected 2026-08-03.)
///
/// Production binaries call this once at startup after loading `Settings`, and
/// it is now the ONLY way these boundaries get installed.
pub fn install_search_runtime_overrides_from_settings(s: &neoethos_core::Settings) {
    install_evaluation_backend_from_settings(s).unwrap_or_else(|error| {
        panic!("invalid discovery evaluation backend configuration: {error}")
    });
    install_backtest_runtime_overrides_from_settings(s); // ✓ S2d config
    install_quality_runtime_overrides_from_settings(s); // ✓ S2c config
    install_genetic_search_runtime_overrides_from_settings(s); // ✓ S2a config
    install_strategy_evaluation_runtime_overrides_from_settings(s); // ✓ S2b config
    install_smc_search_config_from_settings(s); // ✓ S2e config
    install_seen_signature_memory_runtime_overrides_from_settings(s); // ✓ S2f config
    install_stop_target_runtime_overrides_from_settings(s); // ✓ S2g config (2026-08-04)
}
