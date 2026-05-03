pub mod evolution_math;
pub mod search_engine;
pub mod smc_indicators;
pub mod strategy_gene;

pub use evolution_math::{
    EvolutionSearchPolicy, ParentSelectionPolicy, SeenSignatureMemory, SurvivorSelectionPolicy,
    apply_metrics, crossover, gene_signature_hash, generate_random_genes, mutate, new_random_gene,
    reset_derived_metrics, score_from_metrics, select_parent_index, select_survivor_indices,
    unique_candidate_or_retry,
};
pub use search_engine::{
    evaluate_genes, evolve_search, evolve_search_with_progress,
    evolve_search_with_progress_and_limits, month_day_indices, random_search, signals_for_gene,
    signals_for_gene_full,
};
pub use smc_indicators::{
    SmcSearchConfig, build_smc_arrays, derive_smc_arrays, enforce_min_structural_smc_flags,
    enforce_population_smc_ratio, randomize_smc_flags,
};
pub use strategy_gene::{EvaluationConfig, FilteringConfig, Gene, SearchResult};
