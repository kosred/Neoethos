use crate::base::{dataframe_to_float32_array, feature_columns_from_dataframe};
use anyhow::{Context, Result, bail};
use chrono::{Duration, TimeZone, Utc};
use forex_data::{FeatureFrame, Ohlcv};
use forex_search::genetic::{
    Gene, ParentSelectionPolicy, SeenSignatureMemory, SmcSearchConfig, SurvivorSelectionPolicy,
    crossover, generate_random_genes, mutate, select_parent_index, select_survivor_indices,
    signals_for_gene, unique_candidate_or_retry,
};
use forex_search::{DiscoveryConfig, FilteringConfig, run_discovery_cycle};
use ndarray::Array2;
use polars::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use tracing::info;

const ARTIFACT_FILE_NAME: &str = "genetic_portfolio.json";

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
}

impl GeneticStrategyExpert {
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
            if let Ok(column) = df.column(name) {
                if let Ok(series) = column.as_materialized_series().cast(&DataType::Int64) {
                    if let Ok(values) = series.i64() {
                        return values
                            .into_iter()
                            .enumerate()
                            .map(|(idx, value)| value.unwrap_or(idx as i64))
                            .collect();
                    }
                }
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

    fn numeric_column(df: &DataFrame, names: &[&str]) -> Option<Vec<f64>> {
        for name in names {
            if let Ok(column) = df.column(name) {
                if let Ok(series) = column.as_materialized_series().cast(&DataType::Float64) {
                    if let Ok(values) = series.f64() {
                        let mut last = 0.0_f64;
                        return Some(
                            values
                                .into_iter()
                                .map(|value| {
                                    if let Some(value) = value {
                                        last = value;
                                        value
                                    } else {
                                        last
                                    }
                                })
                                .collect(),
                        );
                    }
                }
            }
        }
        None
    }

    fn extract_ohlcv(df: &DataFrame) -> Option<Ohlcv> {
        let open = Self::numeric_column(df, &["open", "o"])?;
        let high = Self::numeric_column(df, &["high", "h"])?;
        let low = Self::numeric_column(df, &["low", "l"])?;
        let close = Self::numeric_column(df, &["close", "c"])?;
        let len = close.len();
        if open.len() != len || high.len() != len || low.len() != len {
            return None;
        }
        Some(Ohlcv {
            timestamp: Some(Self::timestamps_from_frame(df)),
            open,
            high,
            low,
            close,
            volume: Self::numeric_column(df, &["volume", "vol", "v"]),
        })
    }

    fn discovery_config(&self) -> DiscoveryConfig {
        let candidate_count = self
            .population_size
            .saturating_mul(self.generations.max(1))
            .max(self.population_size)
            .min(25_000);
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

        let result = run_discovery_cycle(features, ohlcv, &self.discovery_config())
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

        let smc_cfg = SmcSearchConfig::from_env();
        let population_size = self.population_size.max(16);
        let mut population = generate_random_genes(
            population_size,
            n_indicators,
            self.max_indicators,
            0,
            &smc_cfg,
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
        let mut rng = rand::rng();

        for generation in 0..self.generations {
            let mut scored = population
                .into_iter()
                .map(|mut gene| {
                    let score = Self::evaluate_gene_against_labels(
                        features, &labels, &mut gene, generation,
                    );
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
                let immigrant = unique_candidate_or_retry(
                    forex_search::genetic::new_random_gene(
                        n_indicators,
                        self.max_indicators,
                        generation + 1,
                        &smc_cfg,
                    ),
                    &mut seen_memory,
                    n_indicators,
                    self.max_indicators,
                    generation + 1,
                    12,
                    &smc_cfg,
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
                let mut child = crossover(parent_a, parent_b, generation + 1);
                if rng.random_bool(0.8) {
                    child = mutate(
                        &child,
                        n_indicators,
                        self.max_indicators,
                        generation + 1,
                        &smc_cfg,
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
                    child = crossover(&child, challenger, generation + 1);
                }
                next_generation.push(unique_candidate_or_retry(
                    child,
                    &mut seen_memory,
                    n_indicators,
                    self.max_indicators,
                    generation + 1,
                    12,
                    &smc_cfg,
                ));
            }

            population = next_generation;
        }

        let mut final_scored = population
            .into_iter()
            .map(|mut gene| {
                let score = Self::evaluate_gene_against_labels(
                    features,
                    &labels,
                    &mut gene,
                    self.generations,
                );
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
        if artifact
            .portfolio
            .iter()
            .any(|gene| !gene.fitness.is_finite())
        {
            bail!("genetic artifact contains a portfolio gene with non-finite fitness");
        }
        Ok(())
    }

    fn write_artifact(path: &Path, artifact: &GeneticArtifact) -> Result<()> {
        Self::validate_artifact(artifact)?;
        std::fs::create_dir_all(path)
            .with_context(|| format!("create genetic artifact directory {}", path.display()))?;
        let artifact_path = Self::artifact_path(path);
        let temp_path = artifact_path.with_extension("tmp");
        let payload = serde_json::to_vec_pretty(artifact)
            .with_context(|| format!("serialize genetic artifact {}", artifact_path.display()))?;
        std::fs::write(&temp_path, payload)
            .with_context(|| format!("write genetic temp artifact {}", temp_path.display()))?;
        std::fs::rename(&temp_path, &artifact_path)
            .with_context(|| format!("rename genetic artifact into {}", artifact_path.display()))?;
        Ok(())
    }

    fn read_artifact(path: &Path) -> Result<GeneticArtifact> {
        let artifact_path = Self::artifact_path(path);
        let payload = std::fs::read(&artifact_path)
            .with_context(|| format!("read genetic artifact {}", artifact_path.display()))?;
        let artifact = serde_json::from_slice(&payload)
            .with_context(|| format!("deserialize genetic artifact {}", artifact_path.display()))?;
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

        let portfolio = if let Some(ohlcv) = metadata
            .and_then(Self::extract_ohlcv)
            .or_else(|| Self::extract_ohlcv(x))
        {
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
            .unwrap_or(0.0);
        self.portfolio = portfolio;

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

    pub fn save(&self, path: &Path) -> Result<()> {
        if self.portfolio.is_empty() {
            bail!("cannot save genetic model before training or loading")
        }

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
        };
        Self::write_artifact(path, &artifact)?;
        info!("Saved genetic expert to: {:?}", path);
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let artifact = Self::read_artifact(path)?;
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

impl Default for GeneticStrategyExpert {
    fn default() -> Self {
        Self::new(50, 10, 8).expect("default genetic strategy expert should initialize")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_model_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be monotonic")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "forex_models_{name}_{}_{}",
            std::process::id(),
            stamp
        ))
    }

    #[test]
    fn genetic_load_rejects_empty_portfolio_artifacts() -> Result<()> {
        let path = temp_model_dir("genetic_empty_portfolio");
        std::fs::create_dir_all(&path)?;

        let artifact = GeneticArtifact {
            portfolio: Vec::new(),
            feature_columns: vec!["f1".to_string()],
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
}
