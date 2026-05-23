use crate::base::{
    ExpertModel, build_runtime_prediction_with_details, canonical_three_class_label_mapping,
    dataframe_to_float32_array, feature_columns_from_dataframe, three_class_runtime_confidence,
    try_build_runtime_artifact_metadata,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{
    CapabilityState, ModelFamily, append_runtime_degraded_reason, gpu_policy_cpu_fallback_reason,
};
use crate::runtime::prediction::RuntimePrediction;
use anyhow::{Context, Result, bail};
use chrono::{Duration, TimeZone, Utc};
use ndarray::Array2;
use neoethos_core::storage::json::{
    JsonBackupWriteConfig, read_json as read_json_artifact,
    write_json_with_backup as write_json_artifact_with_backup,
};
use neoethos_data::{FeatureFrame, Ohlcv};
use neoethos_search::genetic::{
    Gene, ParentSelectionPolicy, SeenSignatureMemory, SmcSearchConfig, SurvivorSelectionPolicy,
    crossover, generate_random_genes, mutate, select_parent_index, select_survivor_indices,
    signals_for_gene, unique_candidate_or_retry,
};
use neoethos_search::{DiscoveryConfig, FilteringConfig, run_discovery_cycle};
use polars::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use tracing::info;

const ARTIFACT_FILE_NAME: &str = "genetic_portfolio.json";
const METADATA_FILE_NAME: &str = "metadata.json";
const MODEL_NAME: &str = "genetic";
const DEFAULT_MAX_LABEL_EVALUATIONS: usize = 25_000;
const DEFAULT_MAX_DISCOVERY_CANDIDATES: usize = 25_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum GeneticBackendMode {
    DiscoveryBacked,
    LabelSearch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct GeneticArtifact {
    population_size: usize,
    generations: usize,
    max_indicators: usize,
    portfolio_size: usize,
    train_years: usize,
    val_years: usize,
    symbol: Option<String>,
    feature_columns: Vec<String>,
    backend_mode: GeneticBackendMode,
    parent_selection: ParentSelectionPolicy,
    survivor_selection: SurvivorSelectionPolicy,
    survivor_fraction: f64,
    immigrant_fraction: f64,
    selection_temperature: f64,
    tournament_size: usize,
    best_fitness: f64,
    portfolio: Vec<Gene>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_metadata: Option<RuntimeArtifactMetadata>,
}

impl Default for GeneticArtifact {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 10,
            max_indicators: 8,
            portfolio_size: 12,
            train_years: 0,
            val_years: 0,
            symbol: None,
            feature_columns: Vec::new(),
            backend_mode: GeneticBackendMode::LabelSearch,
            parent_selection: ParentSelectionPolicy::RankWeighted,
            survivor_selection: SurvivorSelectionPolicy::RankWeighted,
            survivor_fraction: 0.10,
            immigrant_fraction: 0.18,
            selection_temperature: 0.75,
            tournament_size: 5,
            best_fitness: 0.0,
            portfolio: Vec::new(),
            runtime_metadata: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneticStrategyExpert {
    population_size: usize,
    generations: usize,
    max_indicators: usize,
    portfolio_size: usize,
    train_years: usize,
    val_years: usize,
    symbol: Option<String>,
    feature_columns: Vec<String>,
    backend_mode: GeneticBackendMode,
    parent_selection: ParentSelectionPolicy,
    survivor_selection: SurvivorSelectionPolicy,
    survivor_fraction: f64,
    immigrant_fraction: f64,
    selection_temperature: f64,
    tournament_size: usize,
    portfolio: Vec<Gene>,
    best_fitness: f64,
    runtime_metadata: Option<RuntimeArtifactMetadata>,
}

impl GeneticStrategyExpert {
    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`] adapter.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(default)
    }

    fn max_label_evaluations() -> usize {
        Self::env_usize(
            "FOREX_GENETIC_MAX_LABEL_EVALS",
            DEFAULT_MAX_LABEL_EVALUATIONS,
        )
    }

    fn max_discovery_candidates() -> usize {
        Self::env_usize(
            "FOREX_GENETIC_MAX_DISCOVERY_CANDIDATES",
            DEFAULT_MAX_DISCOVERY_CANDIDATES,
        )
    }

    fn effective_generation_count_with_budget(
        population_size: usize,
        configured_generations: usize,
        max_label_evaluations: usize,
    ) -> usize {
        let population_size = population_size.max(1);
        let configured_generations = configured_generations.max(1);
        let budget_limited_generations = max_label_evaluations.max(1) / population_size;
        budget_limited_generations.clamp(1, configured_generations)
    }

    fn effective_generation_count(population_size: usize, configured_generations: usize) -> usize {
        Self::effective_generation_count_with_budget(
            population_size,
            configured_generations,
            Self::max_label_evaluations(),
        )
    }

    pub fn new(population_size: usize, generations: usize, max_indicators: usize) -> Result<Self> {
        Ok(Self {
            population_size: population_size.max(8),
            generations: generations.max(1),
            max_indicators: max_indicators.max(1),
            portfolio_size: population_size.clamp(4, 24),
            train_years: 0,
            val_years: 0,
            symbol: None,
            feature_columns: Vec::new(),
            backend_mode: GeneticBackendMode::LabelSearch,
            parent_selection: ParentSelectionPolicy::RankWeighted,
            survivor_selection: SurvivorSelectionPolicy::RankWeighted,
            survivor_fraction: 0.10,
            immigrant_fraction: 0.18,
            selection_temperature: 0.75,
            tournament_size: (population_size / 10).max(3),
            portfolio: Vec::new(),
            best_fitness: 0.0,
            runtime_metadata: None,
        })
    }

    pub fn with_portfolio_size(mut self, portfolio_size: usize) -> Self {
        self.portfolio_size = portfolio_size.clamp(1, self.population_size.max(1));
        self
    }

    pub fn with_search_policy(
        mut self,
        parent_selection: ParentSelectionPolicy,
        survivor_selection: SurvivorSelectionPolicy,
        survivor_fraction: f64,
        immigrant_fraction: f64,
        selection_temperature: f64,
        tournament_size: usize,
    ) -> Self {
        self.parent_selection = parent_selection;
        self.survivor_selection = survivor_selection;
        self.survivor_fraction = survivor_fraction.clamp(0.0, 0.95);
        self.immigrant_fraction = immigrant_fraction.clamp(0.0, 0.95);
        self.selection_temperature = selection_temperature.max(1e-3);
        self.tournament_size = tournament_size.max(2);
        self
    }

    pub fn with_history_window(mut self, train_years: usize, val_years: usize) -> Self {
        self.train_years = train_years;
        self.val_years = val_years;
        self
    }

    fn labels_from_series(y: &Series) -> Result<Vec<i32>> {
        let labels = y
            .cast(&DataType::Int32)
            .context("cast genetic labels to Int32")?;
        labels
            .i32()
            .context("access genetic labels as Int32")?
            .into_iter()
            .map(|value| match value {
                Some(label @ -1..=1) => Ok(label),
                Some(other) => {
                    bail!("unsupported genetic-model label: {other}; expected one of -1, 0, 1")
                }
                None => bail!("genetic-model labels may not contain nulls"),
            })
            .collect()
    }

    fn class_index_from_signal(signal: i8) -> usize {
        match signal {
            1 => 1,
            -1 => 2,
            _ => 0,
        }
    }

    fn class_index_from_label(label: i32) -> usize {
        match label {
            1 => 1,
            -1 => 2,
            _ => 0,
        }
    }

    fn timestamps_from_frame(df: &DataFrame) -> Vec<i64> {
        for name in ["timestamp", "time", "date", "datetime"] {
            if let Ok(column) = df.column(name)
                && let Ok(series) = column.as_materialized_series().cast(&DataType::Int64)
                && let Ok(values) = series.i64()
            {
                return values
                    .into_iter()
                    .enumerate()
                    .map(|(idx, value)| value.unwrap_or(idx as i64))
                    .collect();
            }
        }
        (0..df.height()).map(|idx| idx as i64).collect()
    }

    fn feature_frame_from_df(df: &DataFrame, timestamps: Option<Vec<i64>>) -> Result<FeatureFrame> {
        Ok(FeatureFrame {
            timestamps: timestamps.unwrap_or_else(|| Self::timestamps_from_frame(df)),
            names: feature_columns_from_dataframe(df),
            data: dataframe_to_float32_array(df).context("build genetic feature frame matrix")?,
        })
    }

    fn numeric_column(df: &DataFrame, names: &[&str]) -> Result<Option<Vec<f64>>> {
        for name in names {
            if let Ok(column) = df.column(name) {
                let series = column
                    .as_materialized_series()
                    .cast(&DataType::Float64)
                    .with_context(|| format!("cast genetic OHLCV column {name} to Float64"))?;
                let values = series
                    .f64()
                    .with_context(|| format!("access genetic OHLCV column {name} as Float64"))?
                    .into_iter()
                    .enumerate()
                    .map(|(idx, value)| {
                        let value = value.with_context(|| {
                            format!(
                                "genetic OHLCV column {name} contains null at row {idx}; discovery-backed training requires fully materialized market data"
                            )
                        })?;
                        if !value.is_finite() {
                            bail!(
                                "genetic OHLCV column {name} contains non-finite value {} at row {}",
                                value,
                                idx
                            );
                        }
                        Ok(value)
                    })
                    .collect::<Result<Vec<_>>>()?;
                return Ok(Some(values));
            }
        }
        Ok(None)
    }

    fn extract_ohlcv(df: &DataFrame) -> Result<Option<Ohlcv>> {
        let open = Self::numeric_column(df, &["open", "o"])?;
        let high = Self::numeric_column(df, &["high", "h"])?;
        let low = Self::numeric_column(df, &["low", "l"])?;
        let close = Self::numeric_column(df, &["close", "c"])?;
        if open.is_none() && high.is_none() && low.is_none() && close.is_none() {
            return Ok(None);
        }

        let open =
            open.context("genetic OHLCV extraction found close-like data but no open column")?;
        let high =
            high.context("genetic OHLCV extraction found close-like data but no high column")?;
        let low =
            low.context("genetic OHLCV extraction found close-like data but no low column")?;
        let close =
            close.context("genetic OHLCV extraction found incomplete OHLCV market columns")?;
        let len = close.len();
        if open.len() != len || high.len() != len || low.len() != len {
            bail!("genetic OHLCV columns have inconsistent lengths");
        }
        Ok(Some(Ohlcv {
            timestamp: Some(Self::timestamps_from_frame(df)),
            open,
            high,
            low,
            close,
            volume: Self::numeric_column(df, &["volume", "vol", "v"])?,
        }))
    }

    fn discovery_config(&self) -> DiscoveryConfig {
        let max_candidates = Self::max_discovery_candidates();
        let candidate_count = self
            .population_size
            .saturating_mul(self.generations.max(1))
            .max(self.population_size)
            .min(max_candidates);
        DiscoveryConfig {
            population: self.population_size,
            generations: self.generations,
            max_indicators: self.max_indicators,
            candidate_count,
            portfolio_size: self.portfolio_size.clamp(1, self.population_size.max(1)),
            corr_threshold: 0.90,
            min_trades_per_day: 0.10,
            filtering: FilteringConfig::default(),
            ..DiscoveryConfig::default()
        }
    }

    fn timestamp_to_utc(raw: i64) -> Option<chrono::DateTime<Utc>> {
        let abs = raw.unsigned_abs();
        let seconds = if abs > 10_000_000_000_000_000 {
            raw / 1_000_000_000
        } else if abs > 10_000_000_000_000 {
            raw / 1_000_000
        } else if abs > 10_000_000_000 {
            raw / 1_000
        } else {
            raw
        };
        Utc.timestamp_opt(seconds, 0).single()
    }

    fn slice_rows_by_history_window(
        &self,
        timestamps: &[i64],
        row_count: usize,
    ) -> Option<Vec<usize>> {
        if timestamps.len() != row_count || timestamps.is_empty() {
            return None;
        }

        let latest = Self::timestamp_to_utc(*timestamps.last()?)?;
        let val_cutoff = if self.val_years > 0 {
            latest - Duration::days((365 * self.val_years.max(1)) as i64)
        } else {
            latest + Duration::seconds(1)
        };
        let train_start = if self.train_years > 0 {
            val_cutoff - Duration::days((365 * self.train_years.max(1)) as i64)
        } else {
            Utc.timestamp_opt(0, 0).single()?
        };

        let selected = timestamps
            .iter()
            .enumerate()
            .filter_map(|(idx, raw)| {
                let ts = Self::timestamp_to_utc(*raw)?;
                if ts >= train_start && ts < val_cutoff {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if selected.len() >= row_count.clamp(64, 512).min(row_count) {
            Some(selected)
        } else {
            None
        }
    }

    fn split_label_train_val_indices(row_count: usize) -> (Vec<usize>, Vec<usize>) {
        if row_count <= 4 {
            return ((0..row_count).collect(), Vec::new());
        }

        let val_rows = ((row_count as f32) * 0.2).round() as usize;
        let val_rows = val_rows.clamp(1, row_count.saturating_sub(1));
        let train_rows = row_count - val_rows;

        let train = (0..train_rows).collect::<Vec<_>>();
        let val = (train_rows..row_count).collect::<Vec<_>>();
        (train, val)
    }

    fn slice_feature_frame(features: &FeatureFrame, indices: &[usize]) -> FeatureFrame {
        let timestamps = indices
            .iter()
            .filter_map(|idx| features.timestamps.get(*idx).copied())
            .collect::<Vec<_>>();
        let names = features.names.clone();
        let mut data = Array2::<f32>::zeros((indices.len(), features.data.ncols()));
        for (out_row, src_row) in indices.iter().copied().enumerate() {
            for col in 0..features.data.ncols() {
                data[(out_row, col)] = features.data[(src_row, col)];
            }
        }

        FeatureFrame {
            timestamps,
            names,
            data,
        }
    }

    fn slice_ohlcv(ohlcv: &Ohlcv, indices: &[usize]) -> Ohlcv {
        let take = |values: &[f64]| {
            indices
                .iter()
                .filter_map(|idx| values.get(*idx).copied())
                .collect::<Vec<_>>()
        };

        Ohlcv {
            timestamp: ohlcv.timestamp.as_ref().map(|timestamps| {
                indices
                    .iter()
                    .filter_map(|idx| timestamps.get(*idx).copied())
                    .collect::<Vec<_>>()
            }),
            open: take(&ohlcv.open),
            high: take(&ohlcv.high),
            low: take(&ohlcv.low),
            close: take(&ohlcv.close),
            volume: ohlcv.volume.as_ref().map(|values| {
                indices
                    .iter()
                    .filter_map(|idx| values.get(*idx).copied())
                    .collect::<Vec<_>>()
            }),
        }
    }

    fn slice_labels(labels: &[i32], indices: &[usize]) -> Vec<i32> {
        indices
            .iter()
            .filter_map(|idx| labels.get(*idx).copied())
            .collect::<Vec<_>>()
    }

    fn train_with_discovery(&self, features: &FeatureFrame, ohlcv: &Ohlcv) -> Result<Vec<Gene>> {
        let maybe_indices =
            self.slice_rows_by_history_window(&features.timestamps, features.data.nrows());
        let scoped_features;
        let scoped_ohlcv;
        let (features, ohlcv) = if let Some(indices) = maybe_indices.as_ref() {
            scoped_features = Self::slice_feature_frame(features, indices);
            scoped_ohlcv = Self::slice_ohlcv(ohlcv, indices);
            (&scoped_features, &scoped_ohlcv)
        } else {
            (features, ohlcv)
        };

        let resolved_config = self.discovery_config().with_env_runtime_overrides();
        // Surface the resolved determinism policy so operators can
        // correlate neoethos-models genetic-search runs with the typed
        // policy persisted on the discovery profile (Phase 51).
        // Reproducible runs require `Deterministic { seed }`; the two
        // non-deterministic variants are still permitted but flagged
        // here so they are visible in run logs.
        let determinism_policy = neoethos_search::current_determinism_policy();
        tracing::info!(
            target: "neoethos_models::genetic",
            ?determinism_policy,
            "running Rust-native genetic discovery"
        );
        let result = run_discovery_cycle(features, ohlcv, &resolved_config)
            .context("run Rust-native discovery-backed genetic search")?;
        if !result.portfolio.is_empty() {
            return Ok(result.portfolio);
        }
        if !result.candidates.is_empty() {
            return Ok(result.candidates.into_iter().take(8).collect());
        }
        bail!("discovery-backed genetic search returned no candidate strategies")
    }

    fn evaluate_gene_against_labels(
        features: &FeatureFrame,
        labels: &[i32],
        gene: &mut Gene,
        generation: usize,
    ) -> f64 {
        let signals = signals_for_gene(features, gene);
        let mut confusion = [[0usize; 3]; 3];
        let mut non_neutral_predictions = 0usize;
        let mut directional_hits = 0usize;

        for (signal, label) in signals.iter().zip(labels.iter()) {
            let actual = Self::class_index_from_label(*label);
            let predicted = Self::class_index_from_signal(*signal);
            confusion[actual][predicted] += 1;
            if predicted != 0 {
                non_neutral_predictions += 1;
                if predicted == actual {
                    directional_hits += 1;
                }
            }
        }

        let total = labels.len().max(1) as f64;
        let accuracy = (0..3).map(|idx| confusion[idx][idx]).sum::<usize>() as f64 / total;
        let coverage = non_neutral_predictions as f64 / total;
        let directional_precision = if non_neutral_predictions > 0 {
            directional_hits as f64 / non_neutral_predictions as f64
        } else {
            0.0
        };

        let mut macro_f1 = 0.0;
        for (class_idx, row) in confusion.iter().enumerate() {
            let tp = row[class_idx] as f64;
            let predicted = confusion
                .iter()
                .map(|other| other[class_idx])
                .sum::<usize>() as f64;
            let actual = row.iter().sum::<usize>() as f64;
            let precision = if predicted > 0.0 { tp / predicted } else { 0.0 };
            let recall = if actual > 0.0 { tp / actual } else { 0.0 };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            macro_f1 += f1;
        }
        macro_f1 /= 3.0;

        let neutrality_penalty = if coverage < 0.05 {
            (0.05 - coverage) * 50.0
        } else {
            0.0
        };
        let consistency = 0.6 * accuracy + 0.4 * macro_f1;
        let fitness = (accuracy * 100.0)
            + (macro_f1 * 100.0)
            + (coverage * 20.0)
            + (directional_precision * 30.0)
            - neutrality_penalty;

        gene.fitness = fitness;
        gene.sharpe_ratio = macro_f1;
        gene.win_rate = directional_precision;
        gene.max_drawdown = (1.0 - consistency).clamp(0.0, 1.0) * 0.25;
        gene.profit_factor = 1.0 + directional_precision * 2.5;
        gene.expectancy = directional_precision - (1.0 - directional_precision) * 0.5;
        gene.trades_count = non_neutral_predictions;
        gene.generation = generation;
        gene.consistency = consistency;
        fitness
    }

    fn train_with_labels(&self, features: &FeatureFrame, y: &Series) -> Result<Vec<Gene>> {
        let labels = Self::labels_from_series(y)?;
        let n_indicators = features.data.ncols();
        if n_indicators == 0 {
            bail!("genetic label-search requires at least one feature column");
        }
        let (train_indices, val_indices) = Self::split_label_train_val_indices(labels.len());
        let train_features = Self::slice_feature_frame(features, &train_indices);
        let train_labels = Self::slice_labels(&labels, &train_indices);
        let val_features =
            (!val_indices.is_empty()).then(|| Self::slice_feature_frame(features, &val_indices));
        let val_labels =
            (!val_indices.is_empty()).then(|| Self::slice_labels(&labels, &val_indices));

        let smc_cfg = SmcSearchConfig::from_env();
        let population_size = self.population_size.max(16);
        let effective_generations =
            Self::effective_generation_count(population_size, self.generations);
        // GA helpers in neoethos-search now accept `&mut impl Rng` for
        // determinism; create a single RNG up front and thread it through.
        let mut rng = rand::rng();
        let mut population = generate_random_genes(
            population_size,
            n_indicators,
            self.max_indicators,
            0,
            &smc_cfg,
            &mut rng,
        );
        let mut seen_memory = SeenSignatureMemory::default();
        population = population
            .into_iter()
            .map(|gene| {
                unique_candidate_or_retry(
                    gene,
                    &mut seen_memory,
                    n_indicators,
                    self.max_indicators,
                    0,
                    12,
                    &smc_cfg,
                    &mut rng,
                )
            })
            .collect();
        for gene in &mut population {
            gene.normalize(n_indicators, 1);
        }

        let portfolio_size = self.portfolio_size.clamp(1, population_size.max(1));
        let base_survivor_fraction = self.survivor_fraction.clamp(0.0, 0.95);
        let parent_selection = self.parent_selection;
        let survivor_selection = self.survivor_selection;
        let selection_temperature = self.selection_temperature.max(1e-3);
        let tournament_size = self.tournament_size.max(2);
        let immigrant_fraction = self.immigrant_fraction.clamp(0.0, 0.95);
        // `rng` is the seeded one created at the top of this fn (kept across
        // generations so consumers can reproduce by setting a known seed).

        if effective_generations < self.generations {
            info!(
                "genetic label-search budget capped generations from {} to {} (population={}, max_label_evals={})",
                self.generations,
                effective_generations,
                population_size,
                Self::max_label_evaluations()
            );
        }

        for generation in 0..effective_generations {
            let mut scored = population
                .into_iter()
                .map(|mut gene| {
                    let train_score = Self::evaluate_gene_against_labels(
                        &train_features,
                        &train_labels,
                        &mut gene,
                        generation,
                    );
                    let score = if let (Some(val_features), Some(val_labels)) =
                        (val_features.as_ref(), val_labels.as_ref())
                    {
                        let mut val_gene = gene.clone();
                        let val_score = Self::evaluate_gene_against_labels(
                            val_features,
                            val_labels,
                            &mut val_gene,
                            generation,
                        );
                        gene.fitness = 0.65 * train_score + 0.35 * val_score;
                        gene.sharpe_ratio = 0.65 * gene.sharpe_ratio + 0.35 * val_gene.sharpe_ratio;
                        gene.win_rate = 0.65 * gene.win_rate + 0.35 * val_gene.win_rate;
                        gene.max_drawdown = 0.65 * gene.max_drawdown + 0.35 * val_gene.max_drawdown;
                        gene.profit_factor =
                            0.65 * gene.profit_factor + 0.35 * val_gene.profit_factor;
                        gene.expectancy = 0.65 * gene.expectancy + 0.35 * val_gene.expectancy;
                        gene.consistency = 0.65 * gene.consistency + 0.35 * val_gene.consistency;
                        gene.trades_count = ((gene.trades_count as f64 * 0.65)
                            + (val_gene.trades_count as f64 * 0.35))
                            .round() as usize;
                        gene.fitness
                    } else {
                        train_score
                    };
                    (score, gene)
                })
                .collect::<Vec<_>>();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

            let score_vector: Vec<f64> = scored.iter().map(|(score, _)| *score).collect();
            let survivor_count = match survivor_selection {
                SurvivorSelectionPolicy::Generational => 0,
                _ => ((population_size as f64) * base_survivor_fraction)
                    .round()
                    .max(2.0) as usize,
            }
            .min(scored.len())
            .max(
                if matches!(survivor_selection, SurvivorSelectionPolicy::Generational) {
                    0
                } else {
                    2
                },
            );
            let survivor_indices = select_survivor_indices(
                &score_vector,
                survivor_count,
                survivor_selection,
                selection_temperature,
                tournament_size,
                &mut rng,
            );
            let survivors = survivor_indices
                .iter()
                .map(|idx| scored[*idx].1.clone())
                .collect::<Vec<_>>();

            let mut next_generation = survivors;
            let immigrant_count = ((population_size as f64) * immigrant_fraction).round() as usize;
            let immigrant_count =
                immigrant_count.min(population_size.saturating_sub(next_generation.len()));
            for _ in 0..immigrant_count {
                let new_gene = neoethos_search::genetic::new_random_gene(
                    n_indicators,
                    self.max_indicators,
                    generation + 1,
                    &smc_cfg,
                    &mut rng,
                );
                let immigrant = unique_candidate_or_retry(
                    new_gene,
                    &mut seen_memory,
                    n_indicators,
                    self.max_indicators,
                    generation + 1,
                    12,
                    &smc_cfg,
                    &mut rng,
                );
                next_generation.push(immigrant);
            }
            let parent_indices: Vec<usize> = (0..scored.len()).collect();
            while next_generation.len() < population_size {
                let parent_a_idx = select_parent_index(
                    &score_vector,
                    &parent_indices,
                    parent_selection,
                    tournament_size,
                    selection_temperature,
                    &mut rng,
                );
                let mut parent_b_idx = select_parent_index(
                    &score_vector,
                    &parent_indices,
                    parent_selection,
                    tournament_size,
                    selection_temperature,
                    &mut rng,
                );
                if parent_indices.len() > 1 {
                    let mut retries = 0usize;
                    while parent_b_idx == parent_a_idx && retries < 4 {
                        parent_b_idx = select_parent_index(
                            &score_vector,
                            &parent_indices,
                            parent_selection,
                            tournament_size,
                            selection_temperature,
                            &mut rng,
                        );
                        retries += 1;
                    }
                }
                let parent_a = &scored[parent_a_idx].1;
                let parent_b = &scored[parent_b_idx].1;
                let mut child = crossover(parent_a, parent_b, generation + 1, &mut rng);
                if rng.random_bool(0.8) {
                    child = mutate(
                        &child,
                        n_indicators,
                        self.max_indicators,
                        generation + 1,
                        &smc_cfg,
                        0,
                        &mut rng,
                    );
                }
                if rng.random_bool(0.25) {
                    let challenger_idx = select_parent_index(
                        &score_vector,
                        &parent_indices,
                        parent_selection,
                        tournament_size,
                        selection_temperature,
                        &mut rng,
                    );
                    let challenger = &scored[challenger_idx].1;
                    child = crossover(&child, challenger, generation + 1, &mut rng);
                }
                next_generation.push(unique_candidate_or_retry(
                    child,
                    &mut seen_memory,
                    n_indicators,
                    self.max_indicators,
                    generation + 1,
                    12,
                    &smc_cfg,
                    &mut rng,
                ));
            }

            population = next_generation;
        }

        let mut final_scored = population
            .into_iter()
            .map(|mut gene| {
                let train_score = Self::evaluate_gene_against_labels(
                    &train_features,
                    &train_labels,
                    &mut gene,
                    effective_generations,
                );
                let score = if let (Some(val_features), Some(val_labels)) =
                    (val_features.as_ref(), val_labels.as_ref())
                {
                    let mut val_gene = gene.clone();
                    let val_score = Self::evaluate_gene_against_labels(
                        val_features,
                        val_labels,
                        &mut val_gene,
                        effective_generations,
                    );
                    gene.fitness = 0.65 * train_score + 0.35 * val_score;
                    gene.sharpe_ratio = 0.65 * gene.sharpe_ratio + 0.35 * val_gene.sharpe_ratio;
                    gene.win_rate = 0.65 * gene.win_rate + 0.35 * val_gene.win_rate;
                    gene.max_drawdown = 0.65 * gene.max_drawdown + 0.35 * val_gene.max_drawdown;
                    gene.profit_factor = 0.65 * gene.profit_factor + 0.35 * val_gene.profit_factor;
                    gene.expectancy = 0.65 * gene.expectancy + 0.35 * val_gene.expectancy;
                    gene.consistency = 0.65 * gene.consistency + 0.35 * val_gene.consistency;
                    gene.trades_count = ((gene.trades_count as f64 * 0.65)
                        + (val_gene.trades_count as f64 * 0.35))
                        .round() as usize;
                    gene.fitness
                } else {
                    train_score
                };
                (score, gene)
            })
            .collect::<Vec<_>>();
        final_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let mut portfolio = final_scored
            .into_iter()
            .map(|(_, gene)| gene)
            .take(portfolio_size)
            .collect::<Vec<_>>();

        if portfolio.is_empty() {
            bail!("genetic label-search produced an empty portfolio");
        }

        for (idx, gene) in portfolio.iter_mut().enumerate() {
            if gene.strategy_id.trim().is_empty() {
                gene.strategy_id = format!("genetic_gene_{idx}");
            }
        }

        Ok(portfolio)
    }

    fn feature_frame_for_prediction(&self, x: &DataFrame) -> Result<FeatureFrame> {
        let actual_columns = feature_columns_from_dataframe(x);
        if !self.feature_columns.is_empty() && self.feature_columns != actual_columns {
            bail!(
                "feature column mismatch for genetic model; expected {:?}, got {:?}",
                self.feature_columns,
                actual_columns
            );
        }
        Self::feature_frame_from_df(x, None)
    }

    fn artifact_path(path: &Path) -> PathBuf {
        path.join(ARTIFACT_FILE_NAME)
    }

    fn runtime_metadata_path(path: &Path) -> PathBuf {
        path.join(METADATA_FILE_NAME)
    }

    fn staged_artifact_dir(path: &Path) -> PathBuf {
        path.with_extension("tmp_artifact")
    }

    fn backup_artifact_dir(path: &Path) -> PathBuf {
        path.with_extension("bak_artifact")
    }

    fn cleanup_artifact_dir(path: &Path) -> Result<()> {
        if path.exists() {
            std::fs::remove_dir_all(path)
                .with_context(|| format!("remove staged genetic artifact {}", path.display()))?;
        }
        Ok(())
    }

    fn replace_artifact_dir(staged_path: &Path, target_path: &Path) -> Result<()> {
        let backup_path = Self::backup_artifact_dir(target_path);
        Self::cleanup_artifact_dir(&backup_path)?;
        if target_path.exists() {
            std::fs::rename(target_path, &backup_path).with_context(|| {
                format!(
                    "move previous genetic artifact into backup {}",
                    backup_path.display()
                )
            })?;
        }
        if let Err(error) = std::fs::rename(staged_path, target_path) {
            if backup_path.exists() {
                if let Err(restore_err) = std::fs::rename(&backup_path, target_path) {
                    tracing::error!(
                        target: "neoethos_models::artifact",
                        backup = %backup_path.display(),
                        target = %target_path.display(),
                        error = %restore_err,
                        "failed to restore backup after staged-rename failure;                      artifact directory may be in an inconsistent state"
                    );
                }
            }
            bail!(
                "rename staged genetic artifact into {} failed: {}",
                target_path.display(),
                error
            );
        }
        Self::cleanup_artifact_dir(&backup_path)?;
        Ok(())
    }

    fn validate_gene(gene: &Gene, feature_count: usize, context: &str) -> Result<()> {
        if gene.indices.is_empty() {
            bail!("{context} contains a gene without feature indices");
        }
        if gene.indices.len() != gene.weights.len() {
            bail!(
                "{context} contains a gene with {} indices but {} weights",
                gene.indices.len(),
                gene.weights.len()
            );
        }
        if gene.indices.iter().any(|idx| *idx >= feature_count.max(1)) {
            bail!(
                "{context} contains a gene that references indices beyond the persisted feature schema"
            );
        }
        if gene
            .weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
        {
            bail!("{context} contains a gene with non-finite or non-positive weights");
        }
        if !gene.long_threshold.is_finite() || !gene.short_threshold.is_finite() {
            bail!("{context} contains a gene with non-finite thresholds");
        }
        if gene.long_threshold <= gene.short_threshold {
            bail!("{context} contains a gene with invalid threshold ordering");
        }
        if gene.strategy_id.trim().is_empty() {
            bail!("{context} contains a gene with an empty strategy identifier");
        }
        for metric in [
            gene.fitness,
            gene.sharpe_ratio,
            gene.win_rate,
            gene.max_drawdown,
            gene.profit_factor,
            gene.expectancy,
            gene.slice_pass_rate,
            gene.consistency,
            gene.tp_pips,
            gene.sl_pips,
        ] {
            if !metric.is_finite() {
                bail!("{context} contains a gene with non-finite metrics");
            }
        }
        Ok(())
    }

    fn validate_portfolio(
        portfolio: &[Gene],
        feature_columns: &[String],
        context: &str,
    ) -> Result<()> {
        if feature_columns.is_empty() {
            bail!("{context} is missing feature columns");
        }
        if portfolio.is_empty() {
            bail!("{context} has no portfolio");
        }
        for gene in portfolio {
            Self::validate_gene(gene, feature_columns.len(), context)?;
        }
        Ok(())
    }

    fn validate_best_fitness(best_fitness: f64, portfolio: &[Gene], context: &str) -> Result<()> {
        let portfolio_best = portfolio
            .iter()
            .map(|gene| gene.fitness)
            .reduce(f64::max)
            .context("genetic portfolio unexpectedly empty while validating best fitness")?;
        if (portfolio_best - best_fitness).abs() > 1e-6 {
            bail!(
                "{context} best_fitness {} does not match portfolio max fitness {}",
                best_fitness,
                portfolio_best
            );
        }
        Ok(())
    }

    fn training_summary(&self, features: &FeatureFrame) -> TrainingSummaryMetadata {
        let dataset_rows = features.data.nrows();
        if matches!(self.backend_mode, GeneticBackendMode::DiscoveryBacked) {
            let train_rows = self
                .slice_rows_by_history_window(&features.timestamps, dataset_rows)
                .map(|indices| indices.len())
                .unwrap_or(dataset_rows);
            let val_rows = dataset_rows.saturating_sub(train_rows);
            TrainingSummaryMetadata::new(dataset_rows, train_rows, val_rows)
        } else {
            let (train_indices, val_indices) = Self::split_label_train_val_indices(dataset_rows);
            TrainingSummaryMetadata::new(dataset_rows, train_indices.len(), val_indices.len())
        }
    }

    fn build_runtime_metadata(
        feature_columns: Vec<String>,
        training_summary: TrainingSummaryMetadata,
    ) -> Result<RuntimeArtifactMetadata> {
        try_build_runtime_artifact_metadata(
            MODEL_NAME,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
            feature_columns,
            canonical_three_class_label_mapping(),
            training_summary,
        )
    }

    fn validate_runtime_metadata(
        metadata: &RuntimeArtifactMetadata,
        expected_feature_columns: &[String],
    ) -> Result<()> {
        if expected_feature_columns.is_empty() {
            bail!("persisted genetic artifact is missing feature columns");
        }
        if metadata.model_name != MODEL_NAME {
            bail!(
                "runtime metadata mismatch for {MODEL_NAME}: expected model name {MODEL_NAME}, got {}",
                metadata.model_name
            );
        }
        if metadata.family != ModelFamily::Evolutionary {
            bail!(
                "runtime metadata mismatch for {MODEL_NAME}: expected family {:?}, got {:?}",
                ModelFamily::Evolutionary,
                metadata.family
            );
        }
        if metadata.state != CapabilityState::Implemented {
            bail!(
                "runtime metadata mismatch for {MODEL_NAME}: expected state {:?}, got {:?}",
                CapabilityState::Implemented,
                metadata.state
            );
        }
        if metadata.label_mapping != canonical_three_class_label_mapping() {
            bail!("runtime metadata mismatch for {MODEL_NAME}: unexpected label mapping");
        }
        if metadata.feature_columns != expected_feature_columns {
            bail!(
                "runtime metadata mismatch for {MODEL_NAME}: expected feature columns {:?}, got {:?}",
                expected_feature_columns,
                metadata.feature_columns
            );
        }
        if metadata.training_summary.dataset_rows == 0 {
            bail!("runtime metadata mismatch for {MODEL_NAME}: dataset_rows must be positive");
        }
        if metadata.training_summary.train_rows == 0 {
            bail!("runtime metadata mismatch for {MODEL_NAME}: train_rows must be positive");
        }
        if metadata.training_summary.train_rows + metadata.training_summary.val_rows
            != metadata.training_summary.dataset_rows
        {
            bail!(
                "runtime metadata mismatch for {MODEL_NAME}: training rows {} + validation rows {} must equal dataset rows {}",
                metadata.training_summary.train_rows,
                metadata.training_summary.val_rows,
                metadata.training_summary.dataset_rows
            );
        }
        Ok(())
    }

    fn write_runtime_metadata(path: &Path, metadata: &RuntimeArtifactMetadata) -> Result<()> {
        let metadata_path = Self::runtime_metadata_path(path);
        write_json_artifact_with_backup(
            &metadata_path,
            metadata,
            JsonBackupWriteConfig {
                artifact_label: "genetic runtime metadata",
                temp_extension: "tmp",
                backup_extension: "bak",
            },
        )
    }

    fn read_runtime_metadata(path: &Path) -> Result<Option<RuntimeArtifactMetadata>> {
        let metadata_path = Self::runtime_metadata_path(path);
        if !metadata_path.exists() {
            return Ok(None);
        }
        let metadata = read_json_artifact(&metadata_path, "genetic runtime metadata")?;
        Ok(Some(metadata))
    }

    fn resolve_runtime_metadata_from_artifact(
        path: &Path,
        artifact: &GeneticArtifact,
    ) -> Result<RuntimeArtifactMetadata> {
        match Self::read_runtime_metadata(path)? {
            Some(metadata) => {
                if let Some(embedded) = artifact.runtime_metadata.as_ref()
                    && embedded != &metadata
                {
                    bail!("runtime metadata file does not match genetic artifact");
                }
                Self::validate_runtime_metadata(&metadata, &artifact.feature_columns)?;
                Ok(metadata)
            }
            None => {
                let fallback = artifact
                    .runtime_metadata
                    .clone()
                    .context("genetic artifact is missing runtime metadata in both sidecar and artifact payload")?;
                Self::validate_runtime_metadata(&fallback, &artifact.feature_columns)?;
                tracing::warn!(
                    path = %path.display(),
                    "genetic runtime metadata sidecar missing; using embedded artifact runtime metadata"
                );
                Ok(fallback)
            }
        }
    }

    fn runtime_backend(&self) -> String {
        match self.backend_mode {
            GeneticBackendMode::DiscoveryBacked => "genetic_discovery_backed_cpu".to_string(),
            GeneticBackendMode::LabelSearch => "genetic_label_search_cpu".to_string(),
        }
    }

    fn runtime_degraded_reason(&self) -> Option<String> {
        if self.feature_columns.is_empty() {
            return Some("genetic_feature_schema_missing".to_string());
        }
        if self.portfolio.is_empty() {
            return Some("genetic_portfolio_missing".to_string());
        }
        if Self::validate_portfolio(
            &self.portfolio,
            &self.feature_columns,
            "genetic runtime state",
        )
        .is_err()
        {
            return Some("genetic_portfolio_state_invalid".to_string());
        }
        let Some(metadata) = self.runtime_metadata.as_ref() else {
            return Some("genetic_runtime_metadata_missing".to_string());
        };
        if Self::validate_runtime_metadata(metadata, &self.feature_columns).is_err() {
            return Some("genetic_runtime_metadata_invalid".to_string());
        }
        None
    }

    fn ensure_runtime_state_ready(&self) -> Result<&RuntimeArtifactMetadata> {
        if self.feature_columns.is_empty() {
            bail!("genetic runtime is missing persisted feature columns");
        }
        Self::validate_portfolio(
            &self.portfolio,
            &self.feature_columns,
            "genetic runtime state",
        )?;
        let metadata = self
            .runtime_metadata
            .as_ref()
            .context("genetic runtime metadata missing")?;
        Self::validate_runtime_metadata(metadata, &self.feature_columns)?;
        Ok(metadata)
    }

    fn runtime_details(&self) -> (Option<String>, Option<String>) {
        let gpu_cpu_fallback = gpu_policy_cpu_fallback_reason("genetic");
        let degraded_reason = self.runtime_degraded_reason();
        let backend = if degraded_reason.is_some() {
            Some("genetic_unknown".to_string())
        } else {
            Some(self.runtime_backend())
        };
        (
            backend,
            append_runtime_degraded_reason(degraded_reason, gpu_cpu_fallback),
        )
    }

    fn validate_artifact(artifact: &GeneticArtifact) -> Result<()> {
        if artifact.population_size == 0 {
            bail!("genetic artifact population_size must be greater than zero");
        }
        if artifact.generations == 0 {
            bail!("genetic artifact generations must be greater than zero");
        }
        if artifact.max_indicators == 0 {
            bail!("genetic artifact max_indicators must be greater than zero");
        }
        if artifact.portfolio_size == 0 {
            bail!("genetic artifact portfolio_size must be greater than zero");
        }
        if artifact.portfolio_size > artifact.population_size {
            bail!("genetic artifact portfolio_size may not exceed population_size");
        }
        if artifact.tournament_size < 2 {
            bail!("genetic artifact tournament_size must be at least two");
        }
        if artifact.tournament_size > artifact.population_size {
            bail!("genetic artifact tournament_size may not exceed population_size");
        }
        if artifact.feature_columns.is_empty() {
            bail!("genetic artifact must persist feature columns");
        }
        if !artifact.best_fitness.is_finite() {
            bail!("genetic artifact best_fitness must be finite");
        }
        if !artifact.selection_temperature.is_finite() || artifact.selection_temperature <= 0.0 {
            bail!("genetic artifact selection_temperature must be finite and positive");
        }
        if !(0.0..1.0).contains(&artifact.survivor_fraction) {
            bail!("genetic artifact survivor_fraction must lie strictly between 0 and 1");
        }
        if !(0.0..1.0).contains(&artifact.immigrant_fraction) {
            bail!("genetic artifact immigrant_fraction must lie strictly between 0 and 1");
        }
        if artifact.survivor_fraction + artifact.immigrant_fraction >= 1.0 {
            bail!("genetic artifact survivor_fraction + immigrant_fraction must be less than 1");
        }
        if artifact.portfolio.is_empty() {
            bail!("genetic artifact has no portfolio");
        }
        Self::validate_portfolio(
            &artifact.portfolio,
            &artifact.feature_columns,
            "genetic artifact",
        )?;
        Self::validate_best_fitness(
            artifact.best_fitness,
            &artifact.portfolio,
            "genetic artifact",
        )?;
        let runtime_metadata = artifact
            .runtime_metadata
            .as_ref()
            .context("genetic artifact is missing runtime metadata")?;
        Self::validate_runtime_metadata(runtime_metadata, &artifact.feature_columns)?;
        Ok(())
    }

    fn write_artifact(path: &Path, artifact: &GeneticArtifact) -> Result<()> {
        Self::validate_artifact(artifact)?;
        std::fs::create_dir_all(path)
            .with_context(|| format!("create genetic artifact directory {}", path.display()))?;
        let artifact_path = Self::artifact_path(path);
        write_json_artifact_with_backup(
            &artifact_path,
            artifact,
            JsonBackupWriteConfig {
                artifact_label: "genetic artifact",
                temp_extension: "tmp",
                backup_extension: "bak",
            },
        )
    }

    fn read_artifact(path: &Path) -> Result<GeneticArtifact> {
        let artifact_path = Self::artifact_path(path);
        let artifact: GeneticArtifact = read_json_artifact(&artifact_path, "genetic artifact")?;
        Self::validate_artifact(&artifact)?;
        Ok(artifact)
    }

    pub fn fit(
        &mut self,
        x: &DataFrame,
        y: &Series,
        metadata: Option<&DataFrame>,
        symbol: Option<&str>,
    ) -> Result<()> {
        let timestamps = metadata.map(Self::timestamps_from_frame);
        let features = Self::feature_frame_from_df(x, timestamps)?;
        self.feature_columns = features.names.clone();
        self.symbol = symbol.map(|value| value.to_string());

        let metadata_ohlcv = match metadata {
            Some(frame) => Self::extract_ohlcv(frame)?,
            None => None,
        };
        let feature_ohlcv = if metadata_ohlcv.is_none() {
            Self::extract_ohlcv(x)?
        } else {
            None
        };

        let portfolio = if let Some(ohlcv) = metadata_ohlcv.or(feature_ohlcv) {
            self.backend_mode = GeneticBackendMode::DiscoveryBacked;
            self.train_with_discovery(&features, &ohlcv)?
        } else {
            self.backend_mode = GeneticBackendMode::LabelSearch;
            self.train_with_labels(&features, y)?
        };

        self.best_fitness = portfolio
            .iter()
            .map(|gene| gene.fitness)
            .reduce(f64::max)
            .ok_or_else(|| anyhow::anyhow!("genetic training produced an empty portfolio"))?;
        self.portfolio = portfolio;
        Self::validate_best_fitness(self.best_fitness, &self.portfolio, "genetic runtime state")?;
        let training_summary = self.training_summary(&features);
        self.runtime_metadata = Some(Self::build_runtime_metadata(
            self.feature_columns.clone(),
            training_summary,
        )?);

        info!(
            "Genetic expert fitted in {:?} mode (population={}, generations={}, portfolio={})",
            self.backend_mode,
            self.population_size,
            self.generations,
            self.portfolio.len()
        );
        Ok(())
    }

    pub fn predict_proba(
        &self,
        x: &DataFrame,
        _metadata: Option<&DataFrame>,
        _symbol: Option<&str>,
    ) -> Result<Array2<f32>> {
        if self.portfolio.is_empty() {
            bail!("genetic model has no trained portfolio")
        }

        let features = self.feature_frame_for_prediction(x)?;
        let n_samples = features.data.nrows();
        let mut probabilities = Array2::zeros((n_samples, 3));

        for gene in &self.portfolio {
            let signals = signals_for_gene(&features, gene);
            let vote_weight = (gene.fitness.max(1.0) * (1.0 + gene.consistency.max(0.0))) as f32;
            for (row_idx, signal) in signals.into_iter().enumerate() {
                probabilities[(row_idx, Self::class_index_from_signal(signal))] += vote_weight;
            }
        }

        for row_idx in 0..n_samples {
            let row_sum = probabilities[(row_idx, 0)]
                + probabilities[(row_idx, 1)]
                + probabilities[(row_idx, 2)];
            if row_sum <= f32::EPSILON {
                probabilities[(row_idx, 0)] = 1.0;
                probabilities[(row_idx, 1)] = 0.0;
                probabilities[(row_idx, 2)] = 0.0;
            } else {
                probabilities[(row_idx, 0)] /= row_sum;
                probabilities[(row_idx, 1)] /= row_sum;
                probabilities[(row_idx, 2)] /= row_sum;
            }
        }

        Ok(probabilities)
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        self.ensure_runtime_state_ready()?;
        let probabilities = self.predict_proba(x, None, None)?;
        let (execution_backend, degraded_reason) = self.runtime_details();
        let mut predictions = Vec::with_capacity(probabilities.nrows());
        for row in probabilities.outer_iter() {
            let row_values = [row[0], row[1], row[2]];
            let (confidence, abstain_recommended) = three_class_runtime_confidence(row_values)?;
            predictions.push(build_runtime_prediction_with_details(
                MODEL_NAME.to_string(),
                ModelFamily::Evolutionary,
                CapabilityState::Implemented,
                row_values,
                Some(confidence),
                Some(abstain_recommended),
                execution_backend.clone(),
                degraded_reason.clone(),
            )?);
        }
        Ok(predictions)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let runtime_metadata = self.ensure_runtime_state_ready()?.clone();
        let artifact = GeneticArtifact {
            population_size: self.population_size,
            generations: self.generations,
            max_indicators: self.max_indicators,
            portfolio_size: self.portfolio_size,
            train_years: self.train_years,
            val_years: self.val_years,
            symbol: self.symbol.clone(),
            feature_columns: self.feature_columns.clone(),
            backend_mode: self.backend_mode,
            parent_selection: self.parent_selection,
            survivor_selection: self.survivor_selection,
            survivor_fraction: self.survivor_fraction,
            immigrant_fraction: self.immigrant_fraction,
            selection_temperature: self.selection_temperature,
            tournament_size: self.tournament_size,
            best_fitness: self.best_fitness,
            portfolio: self.portfolio.clone(),
            runtime_metadata: Some(runtime_metadata.clone()),
        };
        let staged_path = Self::staged_artifact_dir(path);
        Self::cleanup_artifact_dir(&staged_path)?;
        std::fs::create_dir_all(&staged_path).with_context(|| {
            format!(
                "create staged genetic artifact directory {}",
                staged_path.display()
            )
        })?;
        if let Err(error) = (|| -> Result<()> {
            Self::write_artifact(&staged_path, &artifact)?;
            Self::write_runtime_metadata(&staged_path, &runtime_metadata)?;
            Ok(())
        })() {
            let _ = Self::cleanup_artifact_dir(&staged_path);
            return Err(error);
        }
        Self::replace_artifact_dir(&staged_path, path)?;
        info!("Saved genetic expert to: {:?}", path);
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let artifact = Self::read_artifact(path)?;
        let runtime_metadata = Self::resolve_runtime_metadata_from_artifact(path, &artifact)?;
        self.population_size = artifact.population_size;
        self.generations = artifact.generations;
        self.max_indicators = artifact.max_indicators;
        self.portfolio_size = artifact.portfolio_size;
        self.train_years = artifact.train_years;
        self.val_years = artifact.val_years;
        self.symbol = artifact.symbol;
        self.feature_columns = artifact.feature_columns;
        self.backend_mode = artifact.backend_mode;
        self.parent_selection = artifact.parent_selection;
        self.survivor_selection = artifact.survivor_selection;
        self.survivor_fraction = artifact.survivor_fraction;
        self.immigrant_fraction = artifact.immigrant_fraction;
        self.selection_temperature = artifact.selection_temperature;
        self.tournament_size = artifact.tournament_size;
        self.best_fitness = artifact.best_fitness;
        self.portfolio = artifact.portfolio;
        self.runtime_metadata = Some(runtime_metadata);
        info!("Loaded genetic expert from: {:?}", path);
        Ok(())
    }

    pub fn portfolio_size(&self) -> Result<usize> {
        Ok(self.portfolio.len())
    }

    pub fn best_fitness(&self) -> Result<f64> {
        Ok(self.best_fitness)
    }
}

impl ExpertModel for GeneticStrategyExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        GeneticStrategyExpert::fit(self, x, y, None, None)
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        GeneticStrategyExpert::predict_proba(self, x, None, None)
    }

    fn save(&self, path: &Path) -> Result<()> {
        GeneticStrategyExpert::save(self, path)
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        GeneticStrategyExpert::load(self, path)
    }
}

impl Default for GeneticStrategyExpert {
    fn default() -> Self {
        Self::new(50, 10, 8).expect("default genetic strategy expert should initialize")
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_model_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be monotonic")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "neoethos_models_{name}_{}_{}",
            std::process::id(),
            stamp
        ))
    }

    fn sample_gene() -> Gene {
        Gene {
            indices: vec![0],
            weights: vec![1.0],
            long_threshold: 0.5,
            short_threshold: -0.5,
            fitness: 12.0,
            sharpe_ratio: 0.8,
            win_rate: 0.6,
            max_drawdown: 0.12,
            profit_factor: 1.4,
            expectancy: 0.25,
            trades_count: 18,
            generation: 1,
            strategy_id: "genetic_gene_0".to_string(),
            tp_pips: 15.0,
            sl_pips: 8.0,
            slice_pass_rate: 0.75,
            consistency: 0.68,
            ..Gene::default()
        }
    }

    fn sample_runtime_metadata(feature_columns: Vec<String>) -> RuntimeArtifactMetadata {
        GeneticStrategyExpert::build_runtime_metadata(
            feature_columns,
            TrainingSummaryMetadata::new(24, 24, 0),
        )
        .expect("build metadata")
    }

    #[test]
    fn genetic_load_rejects_empty_portfolio_artifacts() -> Result<()> {
        let path = temp_model_dir("genetic_empty_portfolio");
        std::fs::create_dir_all(&path)?;

        let artifact = GeneticArtifact {
            portfolio: Vec::new(),
            feature_columns: vec!["f1".to_string()],
            runtime_metadata: Some(sample_runtime_metadata(vec!["f1".to_string()])),
            ..GeneticArtifact::default()
        };
        let payload = serde_json::to_vec_pretty(&artifact)?;
        std::fs::write(GeneticStrategyExpert::artifact_path(&path), payload)?;

        let mut expert = GeneticStrategyExpert::default();
        let err = expert
            .load(&path)
            .expect_err("empty portfolio artifacts should be rejected");
        assert!(err.to_string().contains("no portfolio"));

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn genetic_load_rejects_gene_indices_outside_feature_schema() -> Result<()> {
        let path = temp_model_dir("genetic_invalid_gene_indices");
        std::fs::create_dir_all(&path)?;

        let mut gene = sample_gene();
        gene.indices = vec![1];
        let artifact = GeneticArtifact {
            portfolio: vec![gene],
            feature_columns: vec!["f1".to_string()],
            runtime_metadata: Some(sample_runtime_metadata(vec!["f1".to_string()])),
            ..GeneticArtifact::default()
        };
        let payload = serde_json::to_vec_pretty(&artifact)?;
        std::fs::write(GeneticStrategyExpert::artifact_path(&path), payload)?;

        let mut expert = GeneticStrategyExpert::default();
        let err = expert
            .load(&path)
            .expect_err("out-of-range genetic gene indices should be rejected");
        assert!(err.to_string().contains("references indices beyond"));

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn genetic_load_rejects_mismatched_best_fitness() -> Result<()> {
        let path = temp_model_dir("genetic_best_fitness_mismatch");
        std::fs::create_dir_all(&path)?;

        let mut gene = sample_gene();
        gene.fitness = 8.0;
        let artifact = GeneticArtifact {
            best_fitness: 12.0,
            portfolio: vec![gene],
            feature_columns: vec!["f1".to_string()],
            runtime_metadata: Some(sample_runtime_metadata(vec!["f1".to_string()])),
            ..GeneticArtifact::default()
        };
        let payload = serde_json::to_vec_pretty(&artifact)?;
        std::fs::write(GeneticStrategyExpert::artifact_path(&path), payload)?;

        let mut expert = GeneticStrategyExpert::default();
        let err = expert
            .load(&path)
            .expect_err("mismatched best fitness should be rejected");
        assert!(err.to_string().contains("best_fitness"));

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn genetic_load_uses_embedded_runtime_metadata_when_sidecar_missing() -> Result<()> {
        let path = temp_model_dir("genetic_missing_runtime_sidecar");
        std::fs::create_dir_all(&path)?;

        let artifact = GeneticArtifact {
            feature_columns: vec!["f1".to_string()],
            runtime_metadata: Some(sample_runtime_metadata(vec!["f1".to_string()])),
            portfolio: vec![sample_gene()],
            best_fitness: 12.0,
            ..GeneticArtifact::default()
        };
        let payload = serde_json::to_vec_pretty(&artifact)?;
        std::fs::write(GeneticStrategyExpert::artifact_path(&path), payload)?;

        let mut expert = GeneticStrategyExpert::default();
        expert.load(&path)?;
        assert_eq!(expert.feature_columns, vec!["f1".to_string()]);
        assert!(expert.runtime_metadata.is_some());

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn genetic_predict_runtime_rejects_missing_runtime_metadata() -> Result<()> {
        let mut expert = GeneticStrategyExpert::default();
        expert.feature_columns = vec!["f1".to_string()];
        expert.portfolio = vec![sample_gene()];
        expert.best_fitness = 12.0;

        let df = DataFrame::new(vec![Series::new("f1".into(), vec![0.8_f64, -0.4]).into()])?;
        let err = expert
            .predict_runtime(&df)
            .expect_err("missing runtime metadata should fail runtime predictions");
        assert!(err.to_string().contains("runtime metadata"));
        Ok(())
    }

    #[test]
    fn genetic_predict_runtime_reports_backend_details() -> Result<()> {
        let mut expert = GeneticStrategyExpert::default();
        expert.feature_columns = vec!["f1".to_string()];
        expert.backend_mode = GeneticBackendMode::LabelSearch;
        expert.portfolio = vec![sample_gene()];
        expert.best_fitness = 12.0;
        expert.runtime_metadata = Some(sample_runtime_metadata(expert.feature_columns.clone()));

        let df = DataFrame::new(vec![Series::new("f1".into(), vec![0.8_f64, -0.4]).into()])?;
        let predictions = expert.predict_runtime(&df)?;
        assert_eq!(predictions.len(), 2);
        assert_eq!(
            predictions[0].metadata().execution_backend.as_deref(),
            Some("genetic_label_search_cpu")
        );
        assert_eq!(predictions[0].metadata().degraded_reason, None);
        Ok(())
    }

    #[test]
    fn genetic_trait_fit_and_predict_delegate_to_runtime_model() -> Result<()> {
        let features = DataFrame::new(vec![
            Series::new("f1".into(), vec![0.8_f64, -0.2, 1.1, -1.0, 0.3, -0.7]).into(),
            Series::new("f2".into(), vec![0.1_f64, -0.4, 0.9, -0.8, 0.2, -0.5]).into(),
        ])?;
        let labels = Series::new("target".into(), vec![1_i32, -1, 1, -1, 0, 0]);
        let mut expert = GeneticStrategyExpert::new(8, 2, 2)?;

        ExpertModel::fit(&mut expert, &features, &labels)?;
        let probabilities = ExpertModel::predict_proba(&expert, &features)?;

        assert_eq!(probabilities.nrows(), features.height());
        assert_eq!(probabilities.ncols(), 3);
        Ok(())
    }

    #[test]
    fn genetic_label_search_runtime_metadata_tracks_train_val_split() -> Result<()> {
        let features = DataFrame::new(vec![
            Series::new(
                "f1".into(),
                (0..20).map(|idx| idx as f64 * 0.1).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "f2".into(),
                (0..20)
                    .map(|idx| (idx as f64 * 0.1) - 0.5)
                    .collect::<Vec<_>>(),
            )
            .into(),
        ])?;
        let labels = Series::new(
            "target".into(),
            (0..20)
                .map(|idx| match idx % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                })
                .collect::<Vec<_>>(),
        );
        let mut expert = GeneticStrategyExpert::new(12, 2, 2)?;

        expert.fit(&features, &labels, None, None)?;
        let metadata = expert
            .runtime_metadata
            .as_ref()
            .context("genetic runtime metadata should be present after fit")?;
        assert_eq!(metadata.training_summary.dataset_rows, 20);
        assert_eq!(metadata.training_summary.train_rows, 16);
        assert_eq!(metadata.training_summary.val_rows, 4);
        Ok(())
    }

    #[test]
    fn genetic_extract_ohlcv_rejects_null_market_rows() -> Result<()> {
        let df = DataFrame::new(vec![
            Series::new("open".into(), vec![Some(1.0_f64), None]).into(),
            Series::new("high".into(), vec![Some(1.1_f64), Some(1.2)]).into(),
            Series::new("low".into(), vec![Some(0.9_f64), Some(1.0)]).into(),
            Series::new("close".into(), vec![Some(1.05_f64), Some(1.1)]).into(),
        ])?;

        let err = GeneticStrategyExpert::extract_ohlcv(&df)
            .expect_err("null OHLCV rows should fail strict extraction");
        assert!(err.to_string().contains("contains null"));
        Ok(())
    }

    #[test]
    fn genetic_effective_generation_count_obeys_evaluation_budget() {
        assert_eq!(
            GeneticStrategyExpert::effective_generation_count_with_budget(16, 10, 40),
            2
        );
    }

    #[test]
    fn genetic_discovery_candidate_budget_caps_at_default_limit() {
        let expert = GeneticStrategyExpert::new(2_000, 400, 4).expect("construct expert");
        let config = expert.discovery_config();
        assert_eq!(config.candidate_count, DEFAULT_MAX_DISCOVERY_CANDIDATES);
    }
}
