pub mod evolution_math;
pub mod search_engine;
pub mod smc_indicators;
pub mod strategy_gene;

pub use evolution_math::{
    apply_metrics, crossover, gene_signature_hash, generate_random_genes, mutate, new_random_gene,
    score_from_metrics, unique_candidate_or_retry, SeenSignatureMemory,
};
pub use search_engine::{
    evaluate_genes, evolve_search, evolve_search_with_progress, month_day_indices, random_search,
    signals_for_gene,
};
pub use smc_indicators::{
    build_smc_arrays, derive_smc_arrays, enforce_min_structural_smc_flags,
    enforce_population_smc_ratio, randomize_smc_flags, SmcSearchConfig,
};
pub use strategy_gene::{EvaluationConfig, FilteringConfig, Gene, SearchResult};
