pub mod strategy_gene;
pub mod smc_indicators;
pub mod evolution_math;
pub mod search_engine;

pub use strategy_gene::{Gene, FilteringConfig, SearchResult, EvaluationConfig};
pub use smc_indicators::{SmcSearchConfig, derive_smc_arrays, build_smc_arrays, randomize_smc_flags, enforce_min_structural_smc_flags, enforce_population_smc_ratio};
pub use evolution_math::{SeenSignatureMemory, gene_signature_hash, unique_candidate_or_retry, new_random_gene, generate_random_genes, crossover, mutate, score_from_metrics, apply_metrics};
pub use search_engine::{month_day_indices, signals_for_gene, evaluate_genes, random_search, evolve_search};
