pub mod challenge;
pub mod discovery;
#[cfg(feature = "gpu")]
pub mod discovery_gpu;
#[cfg(not(feature = "gpu"))]
pub mod discovery_gpu {
    use anyhow::{bail, Result};
    use forex_data::{FeatureCache, FeatureFrame, Ohlcv, SymbolDataset};
    use serde::Serialize;
    use std::path::Path;

    #[derive(Debug, Clone)]
    pub struct GpuDiscoveryConfig {
        pub population: usize,
        pub generations: usize,
        pub elite_fraction: f64,
        pub sigma: f64,
        pub crossover_rate: f64,
        pub threshold_scale: f64,
        pub threshold_margin: f64,
        pub threshold_clip: f64,
        pub window_bars: usize,
        pub segments: usize,
        pub min_trades_per_day: f64,
        pub trade_penalty: f64,
        pub dd_limit: f64,
        pub dd_penalty: f64,
        pub robust_weight: f64,
        pub pos_window_fraction: f64,
        pub pos_penalty: f64,
        pub chunk_size: usize,
        pub devices: Vec<i64>,
    }

    impl Default for GpuDiscoveryConfig {
        fn default() -> Self {
            Self {
                population: 24000,
                generations: 200,
                elite_fraction: 0.05,
                sigma: 0.5,
                crossover_rate: 0.35,
                threshold_scale: 0.10,
                threshold_margin: 0.02,
                threshold_clip: 0.30,
                window_bars: 1440 * 22 * 6,
                segments: 4,
                min_trades_per_day: 1.0,
                trade_penalty: 25.0,
                dd_limit: 0.04,
                dd_penalty: 200.0,
                robust_weight: 0.2,
                pos_window_fraction: 0.5,
                pos_penalty: 15.0,
                chunk_size: 2048,
                devices: Vec::new(),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct GpuDiscoveryResult {
        pub genomes: Vec<Vec<f32>>,
        pub fitness: Vec<f32>,
        pub feature_names: Vec<String>,
        pub timeframes: Vec<String>,
    }

    #[derive(Debug, Serialize)]
    struct GenomeExport<'a> {
        fitness: f32,
        genome: &'a [f32],
    }

    pub fn save_gpu_genomes(path: impl AsRef<Path>, result: &GpuDiscoveryResult) -> Result<()> {
        let mut payload = Vec::new();
        for (g, f) in result.genomes.iter().zip(result.fitness.iter()) {
            payload.push(GenomeExport {
                fitness: *f,
                genome: g,
            });
        }
        let json = serde_json::to_string_pretty(&payload)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn build_feature_cube(
        _dataset: &SymbolDataset,
        _base_tf: &str,
        _timeframes: &[&str],
        _cache: Option<&FeatureCache>,
    ) -> Result<(Vec<FeatureFrame>, Vec<String>, Ohlcv)> {
        bail!("GPU discovery is disabled (compile with feature 'gpu' to enable).");
    }

    pub fn run_gpu_discovery(
        _frames: Vec<FeatureFrame>,
        _features: Vec<String>,
        _ohlcv: Ohlcv,
        _config: &GpuDiscoveryConfig,
    ) -> Result<GpuDiscoveryResult> {
        bail!("GPU discovery is disabled (compile with feature 'gpu' to enable).");
    }
}

// HPC-specific modules - only compiled on GPU-enabled builds.
#[cfg(feature = "gpu")]
pub mod hpc;
#[cfg(feature = "gpu")]
pub mod hpc_gpu_discovery;
#[cfg(feature = "gpu")]
pub mod hpc_simd;

// Re-export HPC functions when GPU feature is enabled.
#[cfg(feature = "gpu")]
pub use hpc::{
    detect_hyperstack_n3, force_hpc_mode, get_gpu_cpu_affinity, get_optimal_chunk_size,
    get_optimal_population, get_validation_cpu_cores, is_hpc_mode, is_nvlink_pair,
    print_hpc_config, set_thread_affinity,
};
#[cfg(feature = "gpu")]
pub use hpc_gpu_discovery::{run_island_model_discovery, IslandConfig};
#[cfg(feature = "gpu")]
pub use hpc_simd::{batch_evaluate_simd, compute_sharpe_ratio, has_avx2};

pub mod eval;
pub mod gauntlet;
pub mod genetic;
pub mod portfolio;
pub mod quality;
pub mod stop_target;

pub use challenge::{ChallengeOptimizer, ChallengeTarget};
pub use discovery::{run_discovery_cycle, save_portfolio_json, DiscoveryConfig, DiscoveryResult};
pub use discovery_gpu::{
    build_feature_cube, run_gpu_discovery, save_gpu_genomes, GpuDiscoveryConfig, GpuDiscoveryResult,
};
pub use eval::{evaluate_population_core, fast_evaluate_strategy_core};
pub use gauntlet::{GauntletConfig, StrategyGauntlet};
pub use genetic::{
    evaluate_genes, evolve_search, random_search, signals_for_gene, EvaluationConfig, Gene,
    SearchResult,
};
pub use portfolio::{AllocationResult, PortfolioOptimizer, SymbolMetrics};
pub use quality::{StrategyMetrics, StrategyQualityAnalyzer, StrategyRanker, Trade};
pub use stop_target::{compute_stop_distance_series, infer_stop_target_pips, StopTargetSettings};
