mod artifact_io;
// `pub mod challenge;` — DELETED 2026-05-26 (operator directive: dual-mode product).
// `ChallengeOptimizer` had zero callers in the workspace; the prop-firm risk
// allocation it scaffolded is handled downstream by the prop-firm validation
// gates in `crates/neoethos-search/src/discovery.rs` + dual-mode separation
// (PropFirm vs Risky). 161 LOC removed.
pub mod checkpoint;
// **The consumer for `trial_returns`** (2026-08-10). The reader for the
// per-trial return matrix, plus the two statistics that judge the SEARCH rather
// than the winner: the Deflated Sharpe Ratio and PBO by CSCV. Before this
// module the matrix was written and never read, which meant no result this
// project produced was falsifiable — including a good one. Both statistics
// REFUSE with a named reason when the matrix is too short or too small to
// support them, and both are attached to every trial-returns manifest by
// `TrialReturnsWriter::finish`, from where the discovery ledger embeds them.
pub mod deflated;
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
mod prefilter_schema_v1;
// `mod scheduler_assignment;` — DELETED 2026-05-25 (verbose-build pass):
// the file was a 19-LOC orphan with zero callers. The scheduler-driven
// GPU routing it scaffolded is dispatched directly via `BackendKind`
// matching at the `cubecl_eval` boundary; the conversion helper this
// module shipped was never wired. If the scheduler-driven routing
// lands later, reintroduce a fresh helper at that time.

pub mod backend;
#[cfg(any(
    test,
    feature = "gpu-b-adapter",
    feature = "resident-search-slice2-compile-contract"
))]
mod canonical_discovery_config_digest_v1;
mod canonical_native_discovery_request_v1;
mod canonical_native_discovery_run_v1;
#[cfg(all(feature = "gpu-cuda", target_os = "linux"))]
mod canonical_native_generation_zero_publication_v1;
#[cfg(all(test, feature = "gpu-cuda", target_os = "linux"))]
mod canonical_native_generation_zero_publication_v1_tests;
#[cfg(test)]
mod canonical_native_generation_zero_result_size_plan_v1_tests;
mod canonical_native_generation_zero_result_v1;
mod canonical_native_root_io_v1;
mod canonical_native_runtime_authority_v1;
pub mod canonical_trendbar_research;
pub mod data_selection;
mod exact_resident_dataset_authority_v1;
#[cfg(test)]
#[path = "exact_resident_dataset_authority_v1_contract.rs"]
mod exact_resident_dataset_authority_v1_contract;
// Which arithmetic actually evaluated a population. Not GPU-gated: the CPU lane
// records itself too, and the discovery profile has to be able to name the
// engine on every build.
pub mod engine_identity;
pub mod eval;
pub mod eval_telemetry;
#[cfg(any(
    test,
    feature = "gpu-b-adapter",
    feature = "resident-search-slice2-compile-contract"
))]
pub mod gpu_resident_current_config_plan_v1;
#[cfg(any(
    feature = "gpu-b-native",
    feature = "resident-search-slice2-compile-contract"
))]
pub mod resident_search_slice2_v3 {
    pub use crate::gpu_resident_current_config_plan_v1::FullResidentDiscoveryDeadlineReceiptV1;
    pub use neoethos_gpu_cuda::resident_search_slice2_v3::{
        ResidentArchiveKnnCalibrationReceiptV2, ResidentSearchArchiveStagedV3,
        ResidentSearchGenerationChainV3, ResidentSearchRankEnqueuedV3,
        ResidentSearchRejectedAuthorityV3, ResidentSearchTerminalPendingV3,
        ResidentSearchTerminalReceiptV3, ResidentSearchTransitionErrorV3,
        ResidentSearchTryCompleteV3,
    };
}
#[cfg(feature = "gpu-b-native")]
#[path = "gpu_full_discovery/gpu_resident_trim_prefilter_view_v1.rs"]
pub mod gpu_resident_trim_prefilter_view_v1;
mod native_population_residency_receipt_v1;
mod population_engine_run_receipt_v1;
#[cfg(test)]
#[path = "population_engine_run_receipt_v1_contract.rs"]
mod population_engine_run_receipt_v1_contract;
mod population_execution_evidence_v1;
#[cfg(test)]
#[path = "population_execution_evidence_v1_contract.rs"]
mod population_execution_evidence_v1_contract;
mod population_execution_run_receipt_v2;
#[cfg(feature = "gpu-cuda")]
mod prepared_discovery_run_input_v3;
mod strict_discovery_device_route_v1;
#[cfg(test)]
#[path = "strict_discovery_device_route_v1_contract.rs"]
mod strict_discovery_device_route_v1_contract;
#[cfg(feature = "gpu-cuda")]
mod strict_resident_feature_store_v3;
// SLICE 5 (2026-08-08): ambient execution-environment snapshot for the
// discovery run profile — every process-wide knob that can change what the
// search selects, captured through the same accessors the engine reads.
pub mod execution_profile;
pub mod export_state;
pub mod funnel_profile;
pub mod fx_rates;
pub mod goal_report;
#[cfg(any(test, feature = "gpu-b-adapter"))]
mod gpu_fallback;
pub mod gpu_native;
mod historical_evaluation_authority;
pub mod historical_research;
pub mod historical_search_cli;
pub mod historical_search_receipt_prep;
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
mod quote_validated_outer_holdout_v1;
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
// MEASUREMENT SLICE (2026-08-09). Two modules whose value does not depend on any
// other change landing:
//   `run_identity`  — the resolved-config stamp + the config-identity gate that
//                     refuses a run whose payoff floor is unreachable under its
//                     own trailing settings, clamps and cost.
//   `trial_returns` — every trial's per-period return series, persisted. Without
//                     it DSR and PBO are uncomputable and no result this project
//                     produces is falsifiable.
pub mod run_identity;
pub mod stop_target;
#[cfg(feature = "strategy-db")]
pub mod strategy_db;
pub mod trial_returns;
pub mod validation;
pub mod validation_snapshot;

pub use backend::{
    AcceleratorHint, BackendConfigError, DevicePreference, EvaluationBackend, FallbackPolicy,
    current_evaluation_backend, evaluate_population_core_with_backend,
    evaluate_population_core_with_backend_and_audit, install_evaluation_backend,
    install_evaluation_backend_from_settings,
};
pub use canonical_trendbar_research::{
    CANONICAL_TRENDBAR_RESEARCH_DISCOVERY_RESULT_SCHEMA_VERSION_V3,
    CANONICAL_TRENDBAR_RESEARCH_EXECUTION_SCHEMA_VERSION_V3,
    CANONICAL_TRENDBAR_SCREENING_COST_SCHEMA_VERSION_V2,
    CanonicalTrendbarResearchCostAssumptionsV2, CanonicalTrendbarResearchDiscoveryResultV3,
    CanonicalTrendbarResearchExecutionContractV3, CanonicalTrendbarScreeningCostEnvelopeV2,
};
pub use engine_identity::{
    PopulationEvalEngine, PrototypeBReadiness, accelerator_hint_is_compiled, compiled_accelerators,
    prototype_b_readiness, strict_engine_preflight,
};
pub use exact_resident_dataset_authority_v1::{
    EXACT_RESIDENT_DATASET_AUTHORITY_SCHEMA_VERSION_V1, ExactResidentDatasetAuthorityErrorCodeV1,
    ExactResidentDatasetAuthorityErrorV1, ExactResidentDatasetAuthorityV1,
    ExactResidentDatasetViewV1,
};
pub use native_population_residency_receipt_v1::NativePopulationResidencyReceiptV1;
pub use population_engine_run_receipt_v1::{
    POPULATION_ENGINE_RUN_RECEIPT_SCHEMA_VERSION_V1, PopulationEngineRunReceiptErrorCodeV1,
    PopulationEngineRunReceiptErrorV1, PopulationEngineRunReceiptV1,
};
pub use population_execution_run_receipt_v2::ExactPopulationExecutionRunReceiptV2;
#[cfg(feature = "gpu-cuda")]
pub use prepared_discovery_run_input_v3::{
    PreparedCanonicalDiscoveryRunInputV3, PreparedCpuCanonicalDiscoveryRunInputV3,
    PreparedCpuCanonicalTrendbarResearchRunV3, PreparedNativeCudaCanonicalDiscoveryRunInputV3,
    dispatch_canonical_discovery_data_preparation_v3, prepare_canonical_discovery_run_input_v3,
    run_prepared_canonical_discovery_with_holdout_and_progress_v3,
    run_prepared_canonical_trendbar_research_with_cpu_training_handoff_v3,
    run_prepared_canonical_trendbar_research_with_holdout_and_progress_v3,
};
pub use strict_discovery_device_route_v1::{
    ExactCudaDeviceOrdinalV1, SealedNoCompatibleGpuProbeReceiptV1,
    SealedStrictDiscoveryDeviceAdmissionV1, acquire_strict_discovery_device_admission_v1,
};
// `pub use challenge::{ChallengeOptimizer, ChallengeTarget};` — DELETED 2026-05-26.
pub use data_selection::{
    CanonicalDataSelectionError, CanonicalFeatureComputePolicyV1,
    CanonicalFeatureExecutionReceiptV1, CanonicalFeatureMathLaneV1,
    CanonicalSearchArtifactEnvelopeV2, CanonicalSearchArtifactScopeV2,
    CanonicalSearchEvaluatedWindowV1, CanonicalSearchInputReceiptV2, CanonicalSearchRunInputV2,
    CanonicalSearchSourceBindingReceiptV1, CanonicalSearchSourceSegmentReceiptV1,
    CanonicalSearchWindowRoleV1, ExactCanonicalSeries,
};
pub use deflated::{
    DecodeError, DecodedTrialMatrix, DecodedTrialRow, DeflatedSharpeReport, PboReport,
    TRIAL_STATISTICS_SCHEMA, TrialStatisticsReport, analyse_bytes, analyse_matrix, analyse_run,
    decode as decode_trial_matrix, deflated_sharpe, pbo_cscv, read_matrix as read_trial_matrix,
    redeflate_sharpe,
};
pub use discovery::{
    DEFAULT_OOS_HOLDOUT_FRACTION, DiscoveryConfig, DiscoveryPerKindEvidenceHashes,
    DiscoveryProgress, DiscoveryResult, DiscoveryRunProfile, DiscoveryRuntimeOverrides,
    DiscoveryValidationGates, GeneOosResult, LoggedStrategyTrades,
    PROMOTION_SUMMARY_ARTIFACT_KIND_V3, PromotionOutOfSampleVerdictV2, PromotionStrategyEvidenceV2,
    PromotionSummaryAuthorityPayloadV3, QuoteValidatedDiscoveryResultV1, Stage1Window,
    build_discovery_profile, canonical_discovery_normalization_training_rows,
    compute_discovery_forward_test_artifacts, compute_discovery_prop_firm_artifacts,
    discovery_per_kind_evidence_hashes, discovery_validation_evidence_manifest,
    discovery_validation_evidence_manifest_excluding_live_sim, ensure_non_empty_portfolio,
    ensure_portfolio_export_ready, faithful_oos_eval, live_validation_evidence_from_discovery,
    run_canonical_trendbar_research_discovery_with_holdout_and_progress, run_discovery_cycle,
    run_discovery_cycle_with_holdout, run_discovery_cycle_with_holdout_and_progress,
    run_discovery_cycle_with_progress,
    run_discovery_cycle_with_quote_validated_outer_holdout_and_progress,
    save_canonical_backtest_artifacts, save_discovery_profile_json,
    save_forward_test_validation_artifacts, save_funnel_json, save_portfolio_json,
    save_promotion_summary_json, save_prop_firm_validation_artifacts, save_quality_report_json,
    save_trade_log_json, save_walkforward_validation_artifacts,
};
pub use discovery_ledger::{
    DiscoverySearchLedger, GeneRecord, SearchMetadata, ledger_path, load_prior_ledger,
    save_discovery_ledger, seed_seen_from_ledger,
};
pub use execution_profile::{ExecutionEnvironmentProfile, GpuLaneProfile};
pub use historical_research::{
    HISTORICAL_CANDIDATE_RANKING_POLICY_ID, HISTORICAL_CANDIDATE_SCAN_SCHEMA_VERSION,
    HISTORICAL_CANDIDATE_SIGNAL_GENERATOR_ID,
    HISTORICAL_RESEARCH_EXECUTION_CONTRACT_SCHEMA_VERSION, HISTORICAL_RESEARCH_SCHEMA_VERSION,
    HistoricalCandidateDistanceSourceV1, HistoricalCandidateFailurePolicyV1,
    HistoricalCandidateFailureStageV1, HistoricalCandidateFailureV1, HistoricalCandidateRankV1,
    HistoricalCandidateResultStatusV1, HistoricalCandidateResultV2,
    HistoricalCandidateScanContractV2, HistoricalCandidateScanError, HistoricalCandidateScanKindV1,
    HistoricalCandidateScanRequestV2, HistoricalCandidateScanResultV2,
    HistoricalResearchAccountingV1, HistoricalResearchArtifactClassV1,
    HistoricalResearchArtifactV2, HistoricalResearchBackendV1, HistoricalResearchEntryReferenceV1,
    HistoricalResearchError, HistoricalResearchExecutionContractV1, HistoricalResearchGeometryV1,
    HistoricalResearchIntrabarAmbiguityV1, HistoricalResearchMetricsV1,
    HistoricalResearchPriceBasisV1, HistoricalResearchPriceNativeVolatilityDistanceV1,
    HistoricalResearchPromotionEligibilityV1, HistoricalResearchRequestV2,
    HistoricalResearchSeriesBindingV1, HistoricalResearchSignalTimingV1,
    HistoricalResearchSignalV1, historical_candidate_signal_identity_sha256,
    run_historical_research_v2, scan_historical_candidates_v2,
};
pub use run_identity::{
    BindingConstraint, MEASURED_TRAILING_PAYOFF_CEILING, PayoffCeiling, PayoffCeilingInputs,
    ResolvedConfigStamp, assert_payoff_floor_reachable, cost_pips_round_trip,
    max_achievable_payoff, payoff_inputs_for_config, stamp_resolved_config,
};
pub use trial_returns::{
    TrialReturnMatrix, TrialReturnRow, TrialReturnsManifest, TrialReturnsWriter,
    month_keys_spanning, period_returns, write_trial_returns,
};

pub use eval::{
    BacktestMetrics, BacktestRuntimeOverrides, BacktestSettings,
    current_backtest_runtime_overrides, evaluate_population_core,
    install_backtest_runtime_overrides, install_backtest_runtime_overrides_from_settings,
    simulate_trades_broker_real,
};
// `pub use gauntlet::{GauntletConfig, StrategyGauntlet};` — DELETED 2026-05-26.
pub use genetic::{
    ArchiveScoringOverrides, CostProfileRuntimeOverrides, EvaluationConfig, EvolutionSearchPolicy,
    FilteringConfig, Gene, GeneticSearchRuntimeOverrides, ParentSelectionPolicy, SearchResult,
    SeenSignatureMemoryRuntimeOverrides, SelectionPolicyOverrides, SmcGateOverrides,
    SmcWeightRuntimeOverrides, StrategyEvaluationRuntimeOverrides, SurvivorSelectionPolicy,
    current_determinism_policy, current_genetic_search_runtime_overrides,
    current_seen_signature_memory_runtime_overrides, current_strategy_evaluation_runtime_overrides,
    evaluate_genes, evolve_search, evolve_search_with_progress,
    evolve_search_with_progress_and_limits, install_genetic_search_runtime_overrides,
    install_genetic_search_runtime_overrides_from_settings,
    install_seen_signature_memory_runtime_overrides,
    install_seen_signature_memory_runtime_overrides_from_settings,
    install_smc_search_config_from_settings, install_strategy_evaluation_runtime_overrides,
    install_strategy_evaluation_runtime_overrides_from_settings, migration_enabled,
    month_day_indices, push_migrants, random_search, set_migration_enabled, set_search_cancel,
    signals_for_gene, signals_for_gene_full, take_elites,
};
pub use live_portfolio::{
    LIVE_PORTFOLIO_SCHEMA_VERSION, LivePortfolioArtifact, current_retired_rules,
    install_retired_rules_from_settings, load_live_portfolio_json, project_features_to_effective,
    save_live_portfolio_json,
};
pub use neoethos_core::contracts::DeterminismPolicy;
pub use orchestration::{BatchDiscoverySummary, DiscoveryOrchestrator};
pub use portfolio::{AllocationResult, PortfolioOptimizer, SymbolMetrics};
pub use quality::{
    QualityRuntimeOverrides, StrategyMetrics, StrategyQualityAnalyzer, StrategyRanker, Trade,
    current_quality_runtime_overrides, install_quality_runtime_overrides,
    install_quality_runtime_overrides_from_settings,
};
pub use quote_validated_outer_holdout_v1::{
    LockedPortfolioOuterHoldoutReplaySetV1, QUOTE_VALIDATED_OUTER_HOLDOUT_SCHEMA_VERSION_V1,
    QuoteValidatedOuterHoldoutArtifactClassV1, QuoteValidatedOuterHoldoutErrorCodeV1,
    QuoteValidatedOuterHoldoutErrorV1, QuoteValidatedOuterHoldoutMetricsV1,
    QuoteValidatedOuterHoldoutPromotionEligibilityV1, QuoteValidatedOuterHoldoutReceiptV1,
    QuoteValidatedOuterHoldoutResearchEvidenceV1, QuoteValidatedOuterHoldoutTradeOutcomeV1,
    canonical_locked_portfolio_identity_sha256_v1, evaluate_locked_portfolio_outer_holdout_v1,
};
pub use stop_target::{
    StopDistanceError, StopTargetRuntimeOverrides, StopTargetSettings, adaptive_base_pips_series,
    adaptive_sl_tp_pips_series, adaptive_stops_enabled, adaptive_stops_rr,
    compute_stop_distance_series, current_stop_target_runtime_overrides, infer_stop_target_pips,
    install_stop_target_runtime_overrides_from_settings,
};
pub use validation::{
    CANONICAL_BACKTEST_ARTIFACT_KIND, CANONICAL_BACKTEST_SCHEMA_VERSION,
    CanonicalBacktestArtifactFile, CanonicalBacktestPayloadV3, CombinatorialPurgedCV,
    FORWARD_TEST_VALIDATION_ARTIFACT_KIND, FORWARD_TEST_VALIDATION_SCHEMA_VERSION,
    ForwardTestInput, ForwardTestSummary, ForwardTestValidationArtifactFile,
    ForwardTestValidationPayloadV3, LIVE_EXECUTION_SIMULATION_ARTIFACT_KIND,
    LIVE_EXECUTION_SIMULATION_SCHEMA_VERSION, LiveExecutionRuntimeModel,
    LiveExecutionSimulationArtifactFile, LiveExecutionSimulationScope,
    LiveExecutionSimulationSummary, PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND,
    PROP_FIRM_RISK_VALIDATION_SCHEMA_VERSION, PropFirmRiskInput, PropFirmRiskRules,
    PropFirmRiskValidationArtifactFile, PropFirmRiskValidationPayloadV2,
    PropFirmRiskValidationSummary, ValidationStrategyIdentityV2,
    WALKFORWARD_VALIDATION_ARTIFACT_KIND, WALKFORWARD_VALIDATION_SCHEMA_VERSION,
    WalkforwardSplitResult, WalkforwardSummary, WalkforwardValidationArtifactFile,
    WalkforwardValidationPayloadV2, compute_forward_test_summary, compute_prop_firm_risk_summary,
    embargoed_walkforward_backtest, read_canonical_backtest_artifact,
    read_forward_test_validation_artifact, read_live_execution_simulation_artifact,
    read_prop_firm_risk_validation_artifact, read_walkforward_validation_artifact,
    write_canonical_backtest_artifact_atomic, write_forward_test_validation_artifact_atomic,
    write_live_execution_simulation_artifact_atomic,
    write_prop_firm_risk_validation_artifact_atomic, write_walkforward_validation_artifact_atomic,
};
pub use validation_snapshot::{
    DISCOVERY_VALIDATION_SNAPSHOT_MANIFEST_KIND,
    DISCOVERY_VALIDATION_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    DISCOVERY_VALIDATION_SNAPSHOT_POINTER_SCHEMA_VERSION,
    DiscoveryValidationSnapshotManifestEnvelopeV2, DiscoveryValidationSnapshotManifestV2,
    DiscoveryValidationSnapshotMemberV1, DiscoveryValidationSnapshotPointerV1,
    DiscoveryValidationSnapshotStrategyV1, ValidatedDiscoveryValidationSnapshotV2,
    load_discovery_validation_snapshot, save_discovery_validation_snapshot,
};

/// Config-driven entry point — installs every typed runtime-override boundary
/// from the single [`neoethos_core::Settings`], plus the evaluation backend.
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
    crate::genetic::install_gene_stop_bounds_overrides_from_settings(s); // ✓ S2h config (2026-08-09)
    // ✓ config (2026-08-10) — the recipient for the retired
    // NEOETHOS_FEATURE_CUBE_MODE. It is installed HERE rather than through
    // `neoethos_data::install_data_runtime_overrides`, whose two callers are in
    // `neoethos-app` and `neoethos-cli`: widening that function's arity would
    // break their build inside a change that cannot repair them. This function
    // is already called by both binaries with the resolved Settings, so routing
    // it here is what makes the field reach production instead of decorating
    // the struct.
    neoethos_data::install_feature_cube_policy(s.models.data_runtime.feature_cube_mode);
    // ✓ config (2026-08-10, #219) — the auto-cull blacklist reaches the SEARCH.
    // Until today `neoethos-search` held zero references to it: the live loop
    // retired a losing strategy, queued a rediscovery, and the GA was free to
    // re-derive the identical rule on that very run. This reads
    // `<system.data_dir>/strategy_blacklist.json` — the same file
    // `app_services::strategy_blacklist` writes — through the one identity
    // definition in `neoethos_core::strategy_identity`.
    crate::live_portfolio::install_retired_rules_from_settings(s);
}
