pub mod diversity;
pub mod evolution_math;
pub mod regime_labels;
pub mod runtime_overrides;
pub mod search_engine;
pub mod seed_templates;
pub mod smc_indicators;
pub mod strategy_gene;

// **2026-05-26 (operator directive: dual-mode product)**: `DiversityArchiveConfig`,
// `select_diverse_archive`, and `archive_quality_score` were removed — they
// had zero callers outside their own tests. Diversity now happens at the
// `seen_signature_memory` (working-population dedup) + `correlation pruning`
// (final-portfolio dedup) layers. See the comment block in `diversity.rs`.
//
// Earlier (2026-05-25): `score_from_metrics` was already removed in favour of
// `crate::scoring::ga_fitness`. The two diversity helpers retained below
// (`DiversityKey` + `diversity_key` + `smc_mask`) are kept for telemetry/funnel
// use — emitting them next to each gene is still valuable diagnostics.
pub use diversity::{DiversityKey, EvalMetrics, diversity_key, smc_mask};
pub use evolution_math::{
    EvolutionSearchPolicy, ParentSelectionPolicy, SeenSignatureMemory,
    SeenSignatureMemoryRuntimeOverrides, SurvivorSelectionPolicy, apply_metrics, crossover,
    current_seen_signature_memory_runtime_overrides, current_threshold_ladder,
    derive_adaptive_threshold_ladder_from_features, gene_signature_hash, generate_random_genes,
    install_adaptive_threshold_ladder, install_seen_signature_memory_runtime_overrides,
    install_seen_signature_memory_runtime_overrides_from_env,
    install_seen_signature_memory_runtime_overrides_from_settings, mutate, new_random_gene,
    reset_gene_metrics, select_parent_index, select_survivor_indices,
    unique_candidate_or_retry,
};
pub use regime_labels::{
    RegimeLabelPolicy, RegimeWindow, StrategyRegimeProfile, WindowPerformanceLabel,
    build_rolling_regime_windows, label_strategies_by_regime_windows, rank_training_profiles,
    window_quality_score,
};
pub use runtime_overrides::{
    ArchiveScoringOverrides, CostProfileRuntimeOverrides, GeneticSearchRuntimeOverrides,
    SelectionPolicyOverrides, SmcGateOverrides, SmcWeightRuntimeOverrides,
    StrategyEvaluationRuntimeOverrides, current_determinism_policy,
    current_genetic_search_runtime_overrides, current_strategy_evaluation_runtime_overrides,
    install_genetic_search_runtime_overrides, install_genetic_search_runtime_overrides_from_env,
    install_genetic_search_runtime_overrides_from_settings,
    install_strategy_evaluation_runtime_overrides,
    install_strategy_evaluation_runtime_overrides_from_env,
    install_strategy_evaluation_runtime_overrides_from_settings,
};
pub use search_engine::{
    WalkforwardPopulationGenePack, evaluate_genes, evolve_search, evolve_search_with_progress,
    evolve_search_with_progress_and_limits, month_day_indices, random_search, set_search_cancel,
    signals_and_confidence_for_gene_full, signals_and_confidence_for_gene_with_config,
    signals_for_gene, signals_for_gene_full, validation_genes_population,
    validation_genes_population_gathered, validation_genes_population_window,
};
pub use seed_templates::seed_professional_templates;
pub use smc_indicators::{
    SmcSearchConfig, build_smc_arrays, derive_smc_arrays, enforce_min_structural_smc_flags,
    enforce_population_smc_ratio, install_smc_search_config_from_env,
    install_smc_search_config_from_settings, randomize_smc_flags,
};
pub use strategy_gene::{EvaluationConfig, FilteringConfig, Gene, SearchResult};
