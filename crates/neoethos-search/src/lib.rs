mod artifact_io;
pub mod challenge;
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
pub mod discovery;
mod scheduler_assignment;

pub mod eval;
pub mod export_state;
pub mod funnel_profile;
pub mod gauntlet;
pub mod genetic;
pub mod orchestration;
pub mod parity;
pub mod portfolio;
pub mod quality;
pub mod stop_target;
#[cfg(feature = "strategy-db")]
pub mod strategy_db;
pub mod validation;

pub use challenge::{ChallengeOptimizer, ChallengeTarget};
pub use discovery::{
    DiscoveryConfig, DiscoveryPerKindEvidenceHashes, DiscoveryProgress, DiscoveryResult,
    DiscoveryRunProfile, DiscoveryRuntimeOverrides, DiscoveryValidationGates, LoggedStrategyTrades,
    Stage1Window, build_discovery_profile, compute_discovery_forward_test_artifacts,
    compute_discovery_prop_firm_artifacts, discovery_per_kind_evidence_hashes,
    discovery_validation_evidence_manifest,
    discovery_validation_evidence_manifest_excluding_live_sim, ensure_non_empty_portfolio,
    ensure_portfolio_export_ready, live_validation_evidence_from_discovery, run_discovery_cycle,
    run_discovery_cycle_with_progress, save_canonical_backtest_artifacts,
    save_discovery_profile_json, save_forward_test_validation_artifacts, save_portfolio_json,
    save_promotion_summary_json, save_prop_firm_validation_artifacts, save_quality_report_json,
    save_trade_log_json, save_walkforward_validation_artifacts,
};
pub use eval::{
    BacktestMetrics, BacktestRuntimeOverrides, BacktestSettings,
    current_backtest_runtime_overrides, evaluate_population_core, fast_evaluate_strategy_core,
    install_backtest_runtime_overrides, install_backtest_runtime_overrides_from_env,
    simulate_trades_core,
};
pub use gauntlet::{GauntletConfig, StrategyGauntlet};
pub use genetic::{
    ArchiveScoringOverrides, CostProfileRuntimeOverrides, EvaluationConfig, EvolutionSearchPolicy,
    FilteringConfig, Gene, GeneticSearchRuntimeOverrides, ParentSelectionPolicy, SearchResult,
    SeenSignatureMemoryRuntimeOverrides, SelectionPolicyOverrides, SmcGateOverrides,
    SmcWeightRuntimeOverrides, StrategyEvaluationRuntimeOverrides, SurvivorSelectionPolicy,
    current_determinism_policy, current_genetic_search_runtime_overrides,
    current_seen_signature_memory_runtime_overrides, current_strategy_evaluation_runtime_overrides,
    evaluate_genes, evolve_search, evolve_search_with_progress,
    evolve_search_with_progress_and_limits, install_genetic_search_runtime_overrides,
    install_genetic_search_runtime_overrides_from_env,
    install_seen_signature_memory_runtime_overrides,
    install_seen_signature_memory_runtime_overrides_from_env, install_smc_search_config_from_env,
    install_strategy_evaluation_runtime_overrides,
    install_strategy_evaluation_runtime_overrides_from_env, month_day_indices, random_search,
    signals_for_gene,
};
pub use neoethos_core::contracts::DeterminismPolicy;
pub use orchestration::{BatchDiscoverySummary, DiscoveryOrchestrator};
pub use portfolio::{AllocationResult, PortfolioOptimizer, SymbolMetrics};
pub use quality::{
    QualityRuntimeOverrides, StrategyMetrics, StrategyQualityAnalyzer, StrategyRanker, Trade,
    current_quality_runtime_overrides, install_quality_runtime_overrides,
    install_quality_runtime_overrides_from_env,
};
pub use stop_target::{StopTargetSettings, compute_stop_distance_series, infer_stop_target_pips};
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

/// Convenience entry point that installs every typed runtime-override
/// boundary from the legacy `FOREX_BOT_*` env vars in a single call.
/// Production binaries (`neoethos-cli`, `neoethos-app`) invoke this once at
/// startup so the search crate itself never reads `std::env` during a run.
pub fn install_search_runtime_overrides_from_env() {
    install_backtest_runtime_overrides_from_env();
    install_quality_runtime_overrides_from_env();
    install_genetic_search_runtime_overrides_from_env();
    install_strategy_evaluation_runtime_overrides_from_env();
    install_smc_search_config_from_env();
    install_seen_signature_memory_runtime_overrides_from_env();
}
