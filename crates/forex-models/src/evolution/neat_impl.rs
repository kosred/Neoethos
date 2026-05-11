use anyhow::{Context, Result, bail};
use ndarray::Array2;
use polars::prelude::{DataFrame, Series};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoroshiro128PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::path::Path;
use symbios_genetics::Genotype;
use symbios_neat::{Activation, CppnEvaluator, NeatConfig, NeatGenome};

use forex_core::BackendKind;

#[cfg(feature = "neuro-evolution-gpu")]
use super::neat_gpu::{neat_cuda_kernel_enabled, try_population_scores_cuda};
use crate::base::{
    ExpertModel, build_runtime_prediction_with_details, three_class_runtime_confidence,
    try_build_runtime_artifact_metadata,
};
use crate::runtime::artifacts::{
    RuntimeArtifactMetadata, TrainingSummaryMetadata, default_three_class_label_mapping,
};
use crate::runtime::capabilities::{
    CapabilityState, ModelFamily, append_runtime_degraded_reason, normalize_runtime_device_policy,
    runtime_backend_kind_from_label,
};
use crate::runtime::prediction::RuntimePrediction;
use crate::statistical::common::{
    FeatureScaler, METADATA_FILE_NAME, ensure_feature_columns_match, feature_matrix_from_dataframe,
    read_json, remap_three_class_labels, softmax_rows, write_json,
};

const NEAT_ARTIFACT_FILE_NAME: &str = "neat.json";
const NEAT_MODEL_NAME: &str = "neat";
const NEAT_RUNTIME_BACKEND: &str = "symbios_neat_cpu";
#[cfg(feature = "neuro-evolution-gpu")]
const NEAT_CUDA_FITNESS_BACKEND: &str = "symbios_neat_cuda_fitness";
const DEFAULT_NEAT_SPECIES_ELITISM: usize = 0;

fn default_neat_requested_device_policy() -> String {
    "auto".to_string()
}

fn neat_cpu_fallback_reason(policy: &str, runtime_backend: Option<&str>) -> Option<String> {
    let normalized = normalize_runtime_device_policy(policy);
    if normalized == "gpu" || normalized.starts_with("gpu:") {
        if runtime_backend.is_some_and(|backend| backend.contains("cuda")) {
            return None;
        }
        Some(format!(
            "requested device policy `{normalized}`; symbios_neat Rust backend is CPU and does not execute on GPU"
        ))
    } else {
        None
    }
}

fn neat_runtime_backend_kind(runtime_backend: &str) -> BackendKind {
    runtime_backend_kind_from_label(Some(runtime_backend)).unwrap_or(BackendKind::Unavailable)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct NeatArtifact {
    config: NeatConfig,
    generations: usize,
    population_size: usize,
    mutation_rate: f32,
    species_elitism: usize,
    compatibility_threshold: f32,
    immigrant_fraction: f32,
    seed: u64,
    feature_columns: Vec<String>,
    scaler: FeatureScaler,
    best_genome: NeatGenome,
    fitted: bool,
    dataset_rows: usize,
    train_rows: usize,
    val_rows: usize,
    runtime_backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_backend_kind: Option<BackendKind>,
    #[serde(default = "default_neat_requested_device_policy")]
    requested_device_policy: String,
    best_fitness: f32,
    best_loss: f32,
    best_accuracy: f32,
    #[serde(default)]
    runtime_metadata: Option<RuntimeArtifactMetadata>,
}

impl Default for NeatArtifact {
    fn default() -> Self {
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(42);
        let config = NeatConfig::minimal(1, 3);
        Self {
            config: config.clone(),
            generations: 48,
            population_size: 96,
            mutation_rate: 0.85,
            species_elitism: DEFAULT_NEAT_SPECIES_ELITISM,
            compatibility_threshold: 2.5,
            immigrant_fraction: 0.1,
            seed: 42,
            feature_columns: Vec::new(),
            scaler: FeatureScaler {
                means: Vec::new(),
                stds: Vec::new(),
            },
            best_genome: NeatGenome::fully_connected(config, &mut rng),
            fitted: false,
            dataset_rows: 0,
            train_rows: 0,
            val_rows: 0,
            runtime_backend: NEAT_RUNTIME_BACKEND.to_string(),
            runtime_backend_kind: Some(neat_runtime_backend_kind(NEAT_RUNTIME_BACKEND)),
            requested_device_policy: default_neat_requested_device_policy(),
            best_fitness: f32::NEG_INFINITY,
            best_loss: f32::INFINITY,
            best_accuracy: 0.0,
            runtime_metadata: None,
        }
    }
}

#[derive(Clone)]
struct SpeciesBucket {
    representative: usize,
    members: Vec<usize>,
}

#[derive(Clone)]
struct GenomeScore {
    genome: NeatGenome,
    fitness: f32,
    loss: f32,
    accuracy: f32,
    adjusted_fitness: f32,
}

struct NeatDatasetEvaluator<'a> {
    features: &'a Array2<f32>,
    labels: &'a [usize],
}

impl<'a> NeatDatasetEvaluator<'a> {
    fn evaluate(&self, genome: &NeatGenome) -> GenomeScore {
        let probabilities = match evaluate_probabilities(genome, self.features) {
            Ok(probabilities) => probabilities,
            Err(_) => {
                return GenomeScore {
                    genome: genome.clone(),
                    fitness: -1_000_000.0,
                    loss: 1_000_000.0,
                    accuracy: 0.0,
                    adjusted_fitness: -1_000_000.0,
                };
            }
        };

        let mut log_loss = 0.0_f32;
        let mut correct = 0usize;
        let mut confidence_sum = 0.0_f32;
        for row in 0..probabilities.nrows() {
            let expected = self.labels[row];
            let mut best_idx = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            for class_idx in 0..probabilities.ncols() {
                let probability = probabilities[(row, class_idx)].clamp(1e-6, 1.0 - 1e-6);
                if class_idx == expected {
                    log_loss -= probability.ln();
                    confidence_sum += probability;
                }
                if probability > best_value {
                    best_value = probability;
                    best_idx = class_idx;
                }
            }
            if best_idx == expected {
                correct += 1;
            }
        }

        let rows = probabilities.nrows().max(1) as f32;
        let avg_loss = log_loss / rows;
        let accuracy = correct as f32 / rows;
        let confidence = confidence_sum / rows;
        let complexity_penalty = complexity_penalty(genome);
        let fitness = (accuracy * 3.0 + confidence) - avg_loss - complexity_penalty;

        GenomeScore {
            genome: genome.clone(),
            fitness,
            loss: avg_loss,
            accuracy,
            adjusted_fitness: fitness,
        }
    }
}

fn complexity_penalty(genome: &NeatGenome) -> f32 {
    let hidden_nodes = genome.hidden_ids().len() as f32;
    let enabled_connections = genome.num_enabled_connections() as f32;
    hidden_nodes * 0.003 + enabled_connections * 0.0006
}

fn evaluate_probabilities(genome: &NeatGenome, features: &Array2<f32>) -> Result<Array2<f32>> {
    let evaluator = CppnEvaluator::try_new(genome).context("compile NEAT evaluator")?;
    if evaluator.num_outputs() != 3 {
        bail!(
            "NEAT evaluator must expose 3 outputs, got {}",
            evaluator.num_outputs()
        );
    }

    let mut scratch = evaluator.create_scratchpad();
    let mut outputs = [0.0_f32; 3];
    let mut logits = Array2::<f32>::zeros((features.nrows(), 3));

    for row_idx in 0..features.nrows() {
        let inputs = features.row(row_idx).iter().copied().collect::<Vec<_>>();
        evaluator.evaluate_into(&inputs, &mut outputs, &mut scratch);
        for class_idx in 0..3 {
            logits[(row_idx, class_idx)] = outputs[class_idx];
        }
    }

    Ok(softmax_rows(&logits))
}

fn build_neat_config(input_dim: usize) -> NeatConfig {
    NeatConfig {
        num_inputs: input_dim.max(1),
        num_outputs: 3,
        use_bias: true,
        output_activation: Activation::Identity,
        hidden_activations: vec![
            Activation::Tanh,
            Activation::ReLU,
            Activation::LeakyReLU,
            Activation::Sigmoid,
        ],
        add_connection_prob: 0.24,
        add_node_prob: 0.08,
        weight_mutation_prob: 0.9,
        weight_mutation_power: 0.35,
        weight_replace_prob: 0.08,
        weight_range: 1.5,
        toggle_enabled_prob: 0.015,
        activation_mutation_prob: 0.06,
        compatibility_excess_coeff: 1.0,
        compatibility_disjoint_coeff: 1.0,
        compatibility_weight_coeff: 0.45,
    }
}

fn sort_scores_desc(scores: &mut [GenomeScore]) {
    scores.sort_by(|left, right| {
        right
            .fitness
            .partial_cmp(&left.fitness)
            .unwrap_or(Ordering::Equal)
    });
}

fn assign_species(scores: &[GenomeScore], compatibility_threshold: f32) -> Vec<SpeciesBucket> {
    let mut species = Vec::<SpeciesBucket>::new();
    for (idx, score) in scores.iter().enumerate() {
        let mut assigned = false;
        for bucket in &mut species {
            let representative = &scores[bucket.representative].genome;
            if score.genome.compatibility_distance(representative) <= compatibility_threshold {
                bucket.members.push(idx);
                assigned = true;
                break;
            }
        }

        if !assigned {
            species.push(SpeciesBucket {
                representative: idx,
                members: vec![idx],
            });
        }
    }
    species
}

fn adjusted_species_scores(scores: &mut [GenomeScore], species: &[SpeciesBucket]) {
    for bucket in species {
        let divisor = bucket.members.len().max(1) as f32;
        for &idx in &bucket.members {
            scores[idx].adjusted_fitness = scores[idx].fitness / divisor;
        }
    }
}

fn select_parent<'a>(
    members: &'a [usize],
    scores: &'a [GenomeScore],
    rng: &mut Xoroshiro128PlusPlus,
) -> &'a GenomeScore {
    let tournament_size = members.len().clamp(1, 4);
    let mut best_idx = members[rng.random_range(0..members.len())];
    for _ in 1..tournament_size {
        let challenger = members[rng.random_range(0..members.len())];
        let best = &scores[best_idx];
        let candidate = &scores[challenger];
        let candidate_better = candidate.adjusted_fitness > best.adjusted_fitness
            || (candidate.adjusted_fitness == best.adjusted_fitness
                && candidate.fitness > best.fitness);
        if candidate_better {
            best_idx = challenger;
        }
    }
    &scores[best_idx]
}

fn breed_child(
    bucket: &SpeciesBucket,
    scores: &[GenomeScore],
    config: &NeatConfig,
    mutation_rate: f32,
    rng: &mut Xoroshiro128PlusPlus,
) -> NeatGenome {
    let mut child = if bucket.members.len() == 1 {
        scores[bucket.members[0]].genome.clone()
    } else {
        let parent_a = select_parent(&bucket.members, scores, rng);
        let parent_b = select_parent(&bucket.members, scores, rng);
        if (parent_a.fitness - parent_b.fitness).abs() <= f32::EPSILON {
            parent_a
                .genome
                .crossover_equal_fitness(&parent_b.genome, rng)
        } else if parent_a.fitness >= parent_b.fitness {
            parent_a.genome.crossover(&parent_b.genome, rng)
        } else {
            parent_b.genome.crossover(&parent_a.genome, rng)
        }
    };

    child.resize_io(config.num_inputs, config.num_outputs, rng);
    child.mutate(rng, mutation_rate);
    if child.has_cycle() {
        child.break_cycles();
    }
    child.update_depths();
    child
}

fn allocate_species_slots(
    species: &[SpeciesBucket],
    scores: &[GenomeScore],
    available_slots: usize,
) -> Vec<usize> {
    if species.is_empty() {
        return Vec::new();
    }

    let species_scores = species
        .iter()
        .map(|bucket| {
            bucket
                .members
                .iter()
                .map(|&idx| scores[idx].adjusted_fitness.max(0.0))
                .sum::<f32>()
        })
        .collect::<Vec<_>>();

    let score_sum = species_scores.iter().sum::<f32>();
    if score_sum <= f32::EPSILON {
        let base = available_slots / species.len();
        let mut allocation = vec![base; species.len()];
        for slot in allocation.iter_mut().take(available_slots % species.len()) {
            *slot += 1;
        }
        return allocation;
    }

    let mut allocation = vec![0usize; species.len()];
    let mut remainders = Vec::with_capacity(species.len());
    let mut assigned = 0usize;
    for (idx, value) in species_scores.iter().enumerate() {
        let raw = (*value / score_sum) * available_slots as f32;
        let floor = raw.floor() as usize;
        allocation[idx] = floor;
        assigned += floor;
        remainders.push((idx, raw - floor as f32));
    }

    remainders.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    let mut cursor = 0usize;
    while assigned < available_slots {
        allocation[remainders[cursor % remainders.len()].0] += 1;
        assigned += 1;
        cursor += 1;
    }
    allocation
}

fn build_seed_population(
    config: &NeatConfig,
    population_size: usize,
    rng: &mut Xoroshiro128PlusPlus,
) -> Vec<NeatGenome> {
    let mut population = Vec::with_capacity(population_size);
    for idx in 0..population_size {
        let mut genome = if idx % 3 == 0 {
            NeatGenome::minimal(config.clone())
        } else {
            NeatGenome::fully_connected(config.clone(), rng)
        };

        let warmup_mutations = 1 + (idx % 4);
        for _ in 0..warmup_mutations {
            genome.mutate(rng, 0.75);
        }
        if genome.has_cycle() {
            genome.break_cycles();
        }
        genome.update_depths();
        population.push(genome);
    }
    population
}

pub struct NeatExpert {
    config: NeatConfig,
    generations: usize,
    population_size: usize,
    mutation_rate: f32,
    species_elitism: usize,
    compatibility_threshold: f32,
    immigrant_fraction: f32,
    seed: u64,
    feature_columns: Vec<String>,
    scaler: Option<FeatureScaler>,
    best_genome: Option<NeatGenome>,
    fitted: bool,
    dataset_rows: usize,
    train_rows: usize,
    val_rows: usize,
    runtime_backend: String,
    requested_device_policy: String,
    best_fitness: f32,
    best_loss: f32,
    best_accuracy: f32,
}

impl NeatExpert {
    pub fn new(input_dim: usize) -> Self {
        Self::with_config(input_dim, 96, 48)
    }

    pub fn with_config(input_dim: usize, population_size: usize, generations: usize) -> Self {
        Self {
            config: build_neat_config(input_dim),
            generations: generations.max(8),
            population_size: population_size.max(24),
            mutation_rate: 0.85,
            species_elitism: DEFAULT_NEAT_SPECIES_ELITISM,
            compatibility_threshold: 2.5,
            immigrant_fraction: 0.1,
            seed: 42,
            feature_columns: Vec::new(),
            scaler: None,
            best_genome: None,
            fitted: false,
            dataset_rows: 0,
            train_rows: 0,
            val_rows: 0,
            runtime_backend: NEAT_RUNTIME_BACKEND.to_string(),
            requested_device_policy: default_neat_requested_device_policy(),
            best_fitness: f32::NEG_INFINITY,
            best_loss: f32::INFINITY,
            best_accuracy: 0.0,
        }
    }

    pub fn with_search_params(
        mut self,
        mutation_rate: f32,
        species_elitism: usize,
        compatibility_threshold: f32,
        immigrant_fraction: f32,
        seed: u64,
    ) -> Self {
        self.mutation_rate = mutation_rate.clamp(0.05, 1.5);
        self.species_elitism = species_elitism.min(self.population_size);
        self.compatibility_threshold = compatibility_threshold.max(0.25);
        self.immigrant_fraction = immigrant_fraction.clamp(0.0, 0.4);
        self.seed = seed;
        self
    }

    pub fn with_device_policy(mut self, policy: impl AsRef<str>) -> Self {
        self.requested_device_policy = normalize_runtime_device_policy(policy.as_ref());
        self
    }

    fn split_train_val_indices(rows: usize) -> (Vec<usize>, Vec<usize>) {
        if rows <= 4 {
            return ((0..rows).collect(), Vec::new());
        }

        let val_rows = ((rows as f32) * 0.2).round() as usize;
        let val_rows = val_rows.clamp(1, rows.saturating_sub(1));
        let train_rows = rows - val_rows;

        let train = (0..train_rows).collect::<Vec<_>>();
        let val = (train_rows..rows).collect::<Vec<_>>();
        (train, val)
    }

    fn slice_rows(features: &Array2<f32>, indices: &[usize]) -> Array2<f32> {
        let mut sliced = Array2::<f32>::zeros((indices.len(), features.ncols()));
        for (out_row, src_row) in indices.iter().copied().enumerate() {
            for col in 0..features.ncols() {
                sliced[(out_row, col)] = features[(src_row, col)];
            }
        }
        sliced
    }

    fn slice_labels(labels: &[usize], indices: &[usize]) -> Vec<usize> {
        indices
            .iter()
            .filter_map(|idx| labels.get(*idx).copied())
            .collect::<Vec<_>>()
    }

    fn evolve_population(
        &mut self,
        train_features: &Array2<f32>,
        train_labels: &[usize],
        val_features: &Array2<f32>,
        val_labels: &[usize],
    ) -> Result<GenomeScore> {
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(self.seed);
        let mut population = build_seed_population(&self.config, self.population_size, &mut rng);
        let train_evaluator = NeatDatasetEvaluator {
            features: train_features,
            labels: train_labels,
        };
        let val_evaluator = (!val_labels.is_empty()).then_some(NeatDatasetEvaluator {
            features: val_features,
            labels: val_labels,
        });
        let mut best = GenomeScore {
            genome: population
                .first()
                .cloned()
                .context("empty NEAT population")?,
            fitness: f32::NEG_INFINITY,
            loss: f32::INFINITY,
            accuracy: 0.0,
            adjusted_fitness: f32::NEG_INFINITY,
        };
        #[cfg(feature = "neuro-evolution-gpu")]
        let mut used_cuda_fitness = false;
        #[cfg(feature = "neuro-evolution-gpu")]
        let mut cuda_fitness_disabled = false;

        for _generation in 0..self.generations {
            #[cfg(feature = "neuro-evolution-gpu")]
            let mut cuda_scores = None;
            #[cfg(feature = "neuro-evolution-gpu")]
            if !cuda_fitness_disabled && neat_cuda_kernel_enabled(&self.requested_device_policy) {
                match try_population_scores_cuda(
                    &population,
                    train_features,
                    train_labels,
                    val_features,
                    val_labels,
                    &self.requested_device_policy,
                ) {
                    Ok(metrics) if metrics.len() == population.len() => {
                        used_cuda_fitness = true;
                        cuda_scores = Some(
                            population
                                .iter()
                                .cloned()
                                .zip(metrics)
                                .map(|(genome, metrics)| GenomeScore {
                                    genome,
                                    fitness: metrics.fitness,
                                    loss: metrics.loss,
                                    accuracy: metrics.accuracy,
                                    adjusted_fitness: metrics.fitness,
                                })
                                .collect::<Vec<_>>(),
                        );
                    }
                    Ok(metrics) => {
                        cuda_fitness_disabled = true;
                        tracing::warn!(
                            "NEAT cuda fitness kernel returned {} scores for {} genomes; falling back to cpu fitness evaluation",
                            metrics.len(),
                            population.len()
                        );
                    }
                    Err(err) => {
                        cuda_fitness_disabled = true;
                        tracing::warn!(
                            "NEAT cuda fitness kernel unavailable, falling back to cpu fitness evaluation: {err}"
                        );
                    }
                }
            }
            let cpu_scores = || {
                population
                    .par_iter()
                    .map(|genome| {
                        let mut score = train_evaluator.evaluate(genome);
                        if let Some(evaluator) = val_evaluator.as_ref() {
                            let val_score = evaluator.evaluate(genome);
                            score.fitness = 0.65 * score.fitness + 0.35 * val_score.fitness;
                            score.loss = 0.65 * score.loss + 0.35 * val_score.loss;
                            score.accuracy = 0.65 * score.accuracy + 0.35 * val_score.accuracy;
                            score.adjusted_fitness = score.fitness;
                        }
                        score
                    })
                    .collect::<Vec<_>>()
            };
            #[cfg(feature = "neuro-evolution-gpu")]
            let mut scores = cuda_scores.unwrap_or_else(cpu_scores);
            #[cfg(not(feature = "neuro-evolution-gpu"))]
            let mut scores = cpu_scores();
            sort_scores_desc(&mut scores);
            if scores
                .first()
                .is_some_and(|score| score.fitness > best.fitness)
            {
                best = scores[0].clone();
            }

            let species = assign_species(&scores, self.compatibility_threshold);
            adjusted_species_scores(&mut scores, &species);
            sort_scores_desc(&mut scores);

            let immigrant_count =
                ((self.population_size as f32) * self.immigrant_fraction).round() as usize;
            let immigrant_count = immigrant_count.min(self.population_size / 3);

            let mut next_population = Vec::with_capacity(self.population_size);
            for bucket in &species {
                let elite_count = self.species_elitism.min(bucket.members.len());
                if elite_count == 0 {
                    continue;
                }
                let mut ranked = bucket.members.clone();
                ranked.sort_by(|left, right| {
                    scores[*right]
                        .fitness
                        .partial_cmp(&scores[*left].fitness)
                        .unwrap_or(Ordering::Equal)
                });
                for &member_idx in ranked.iter().take(elite_count) {
                    if next_population.len() < self.population_size.saturating_sub(immigrant_count)
                    {
                        next_population.push(scores[member_idx].genome.clone());
                    }
                }
            }

            let species_slots = self
                .population_size
                .saturating_sub(immigrant_count)
                .saturating_sub(next_population.len());
            let allocation = allocate_species_slots(&species, &scores, species_slots);
            for (bucket, &slot_count) in species.iter().zip(allocation.iter()) {
                for _ in 0..slot_count {
                    if next_population.len() >= self.population_size.saturating_sub(immigrant_count)
                    {
                        break;
                    }
                    next_population.push(breed_child(
                        bucket,
                        &scores,
                        &self.config,
                        self.mutation_rate,
                        &mut rng,
                    ));
                }
            }

            while next_population.len() < self.population_size.saturating_sub(immigrant_count) {
                if let Some(bucket) = species.first() {
                    next_population.push(breed_child(
                        bucket,
                        &scores,
                        &self.config,
                        self.mutation_rate,
                        &mut rng,
                    ));
                } else {
                    next_population
                        .push(NeatGenome::fully_connected(self.config.clone(), &mut rng));
                }
            }

            while next_population.len() < self.population_size {
                let mut immigrant = if rng.random::<f32>() < 0.5 {
                    NeatGenome::minimal(self.config.clone())
                } else {
                    NeatGenome::fully_connected(self.config.clone(), &mut rng)
                };
                immigrant.mutate(&mut rng, 1.0);
                immigrant.update_depths();
                next_population.push(immigrant);
            }

            population = next_population;
        }

        #[cfg(feature = "neuro-evolution-gpu")]
        if used_cuda_fitness {
            self.runtime_backend = NEAT_CUDA_FITNESS_BACKEND.to_string();
        }

        Ok(best)
    }

    fn runtime_details(&self) -> (Option<String>, Option<String>) {
        let gpu_cpu_fallback =
            neat_cpu_fallback_reason(&self.requested_device_policy, Some(&self.runtime_backend));
        if !self.fitted {
            return (
                Some("neat_unknown".to_string()),
                append_runtime_degraded_reason(
                    Some("neat_not_fitted".to_string()),
                    gpu_cpu_fallback,
                ),
            );
        }
        if self.feature_columns.is_empty() {
            return (
                Some("neat_unknown".to_string()),
                append_runtime_degraded_reason(
                    Some("neat_feature_schema_missing".to_string()),
                    gpu_cpu_fallback,
                ),
            );
        }
        if self.scaler.is_none() || self.best_genome.is_none() {
            return (
                Some("neat_unknown".to_string()),
                append_runtime_degraded_reason(
                    Some("neat_runtime_state_incomplete".to_string()),
                    gpu_cpu_fallback,
                ),
            );
        }
        if self.train_rows == 0 || self.train_rows + self.val_rows != self.dataset_rows {
            return (
                Some("neat_unknown".to_string()),
                append_runtime_degraded_reason(
                    Some("neat_training_summary_incomplete".to_string()),
                    gpu_cpu_fallback,
                ),
            );
        }
        let backend = if self.runtime_backend.trim().is_empty() {
            "neat_unknown".to_string()
        } else {
            self.runtime_backend.clone()
        };
        let degraded_reason = if backend == "neat_unknown" {
            Some("neat_runtime_backend_missing".to_string())
        } else {
            None
        };
        (
            Some(backend),
            append_runtime_degraded_reason(degraded_reason, gpu_cpu_fallback),
        )
    }

    fn ensure_runtime_state_ready(&self) -> Result<()> {
        if !self.fitted {
            bail!("NEAT expert has not been fitted");
        }
        if self.feature_columns.is_empty() {
            bail!("NEAT feature schema missing");
        }
        if self.best_genome.is_none() {
            bail!("NEAT genome missing");
        }
        if self.scaler.is_none() {
            bail!("NEAT scaler missing");
        }
        if self.dataset_rows == 0 || self.train_rows == 0 {
            bail!("NEAT training summary is incomplete");
        }
        if self.train_rows + self.val_rows != self.dataset_rows {
            bail!("NEAT training summary is inconsistent");
        }
        if self.runtime_backend.trim().is_empty() {
            bail!("NEAT runtime backend missing");
        }
        let scaler = self.scaler.as_ref().context("NEAT scaler missing")?;
        if scaler.means.len() != self.feature_columns.len()
            || scaler.stds.len() != self.feature_columns.len()
        {
            bail!("NEAT scaler does not match feature schema");
        }
        if scaler.means.iter().any(|value| !value.is_finite()) {
            bail!("NEAT scaler contains non-finite means");
        }
        if scaler
            .stds
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            bail!("NEAT scaler contains non-finite or non-positive stds");
        }
        Ok(())
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x)?;
        let (execution_backend, degraded_reason) = self.runtime_details();
        let mut predictions = Vec::with_capacity(probabilities.nrows());
        for row in probabilities.outer_iter() {
            let row_values = [row[0], row[1], row[2]];
            let (confidence, abstain_recommended) = three_class_runtime_confidence(row_values)?;
            predictions.push(build_runtime_prediction_with_details(
                NEAT_MODEL_NAME.to_string(),
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

    fn validate_loaded_metadata(
        metadata: &crate::runtime::artifacts::RuntimeArtifactMetadata,
    ) -> Result<()> {
        if metadata.model_name != NEAT_MODEL_NAME {
            bail!(
                "NEAT artifact model mismatch: expected {NEAT_MODEL_NAME}, got {}",
                metadata.model_name
            );
        }

        if metadata.family != ModelFamily::Evolutionary {
            bail!(
                "NEAT artifact family mismatch: expected {:?}, got {:?}",
                ModelFamily::Evolutionary,
                metadata.family
            );
        }

        if metadata.state != CapabilityState::Implemented {
            bail!(
                "NEAT artifact state mismatch: expected {:?}, got {:?}",
                CapabilityState::Implemented,
                metadata.state
            );
        }

        if metadata.feature_columns.is_empty() {
            bail!("NEAT artifact metadata must contain at least one feature column");
        }

        if metadata.label_mapping != default_three_class_label_mapping() {
            bail!("NEAT artifact label mapping mismatch");
        }

        if metadata.training_summary.dataset_rows
            != metadata.training_summary.train_rows + metadata.training_summary.val_rows
        {
            bail!("NEAT artifact training summary is inconsistent");
        }
        if metadata.training_summary.train_rows == 0 {
            bail!("NEAT artifact training summary must contain positive train_rows");
        }

        Ok(())
    }

    fn validate_loaded_artifact(
        metadata: &crate::runtime::artifacts::RuntimeArtifactMetadata,
        artifact: &NeatArtifact,
    ) -> Result<()> {
        if artifact.feature_columns.is_empty() {
            bail!("NEAT artifact must contain feature columns");
        }

        if artifact.feature_columns.len() != artifact.scaler.means.len()
            || artifact.feature_columns.len() != artifact.scaler.stds.len()
        {
            bail!(
                "NEAT artifact scaler mismatch: feature columns {}, means {}, stds {}",
                artifact.feature_columns.len(),
                artifact.scaler.means.len(),
                artifact.scaler.stds.len()
            );
        }
        if artifact.scaler.means.iter().any(|value| !value.is_finite()) {
            bail!("NEAT artifact scaler contains non-finite means");
        }
        if artifact
            .scaler
            .stds
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            bail!("NEAT artifact scaler contains non-finite or non-positive stds");
        }

        if artifact.population_size < 24 || artifact.generations < 8 {
            bail!(
                "NEAT artifact search topology is invalid: generations={}, population_size={}",
                artifact.generations,
                artifact.population_size
            );
        }
        if artifact.species_elitism > artifact.population_size {
            bail!(
                "NEAT artifact species_elitism ({}) cannot exceed population_size ({})",
                artifact.species_elitism,
                artifact.population_size
            );
        }

        if artifact.dataset_rows == 0 {
            bail!("NEAT artifact dataset rows must be greater than zero");
        }
        if artifact.train_rows == 0 {
            bail!("NEAT artifact train_rows must be greater than zero");
        }
        if artifact.train_rows + artifact.val_rows != artifact.dataset_rows {
            bail!("NEAT artifact train_rows + val_rows must equal dataset_rows");
        }
        if artifact.runtime_backend.trim().is_empty() {
            bail!("NEAT artifact must persist a runtime backend label");
        }
        if let Some(runtime_backend_kind) = artifact.runtime_backend_kind {
            let expected = neat_runtime_backend_kind(&artifact.runtime_backend);
            if runtime_backend_kind != expected {
                bail!(
                    "NEAT artifact runtime_backend_kind {:?} does not match runtime backend label `{}` ({:?})",
                    runtime_backend_kind,
                    artifact.runtime_backend,
                    expected
                );
            }
        }
        if artifact.requested_device_policy.trim().is_empty() {
            bail!("NEAT artifact requested_device_policy must not be blank");
        }

        if artifact.mutation_rate.is_nan()
            || !artifact.mutation_rate.is_finite()
            || !(0.05..=1.5).contains(&artifact.mutation_rate)
        {
            bail!("NEAT artifact mutation rate is out of contract range");
        }

        if artifact.config.num_inputs != artifact.feature_columns.len() {
            bail!(
                "NEAT artifact input dimension mismatch: config expects {}, feature columns {}",
                artifact.config.num_inputs,
                artifact.feature_columns.len()
            );
        }

        if artifact.config.num_outputs != 3 {
            bail!(
                "NEAT artifact output dimension mismatch: expected 3, got {}",
                artifact.config.num_outputs
            );
        }

        if metadata.feature_columns != artifact.feature_columns {
            bail!("NEAT artifact feature columns do not match runtime metadata");
        }

        if metadata.training_summary.dataset_rows != artifact.dataset_rows {
            bail!("NEAT artifact dataset row count does not match metadata");
        }
        if metadata.training_summary.train_rows != artifact.train_rows
            || metadata.training_summary.val_rows != artifact.val_rows
        {
            bail!("NEAT artifact training summary does not match metadata");
        }

        if !artifact.best_fitness.is_finite()
            || !artifact.best_loss.is_finite()
            || !artifact.best_accuracy.is_finite()
        {
            bail!("NEAT artifact metrics are not finite");
        }

        if !(0.0..=1.0).contains(&artifact.best_accuracy) {
            bail!(
                "NEAT artifact accuracy is out of range: {}",
                artifact.best_accuracy
            );
        }

        if !artifact.fitted {
            bail!("NEAT artifact must be marked fitted before loading");
        }

        Ok(())
    }

    fn metadata_from_artifact(artifact: &NeatArtifact) -> Result<RuntimeArtifactMetadata> {
        try_build_runtime_artifact_metadata(
            NEAT_MODEL_NAME,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
            artifact.feature_columns.clone(),
            default_three_class_label_mapping(),
            TrainingSummaryMetadata::new(
                artifact.dataset_rows,
                artifact.train_rows,
                artifact.val_rows,
            ),
        )
    }

    fn validate_metadata_consistency(
        sidecar: &RuntimeArtifactMetadata,
        embedded: &RuntimeArtifactMetadata,
    ) -> Result<()> {
        if sidecar.model_name != embedded.model_name
            || sidecar.family != embedded.family
            || sidecar.state != embedded.state
        {
            bail!("NEAT metadata identity mismatch between sidecar and embedded payload");
        }
        if sidecar.feature_columns != embedded.feature_columns {
            bail!("NEAT metadata feature columns drift between sidecar and embedded");
        }
        if sidecar.label_mapping != embedded.label_mapping {
            bail!("NEAT metadata label mapping drift between sidecar and embedded");
        }
        if sidecar.training_summary != embedded.training_summary {
            bail!("NEAT metadata training summary drift between sidecar and embedded");
        }
        Ok(())
    }

    fn resolve_loaded_metadata(
        path: &Path,
        artifact: &NeatArtifact,
    ) -> Result<RuntimeArtifactMetadata> {
        let metadata_path = path.join(METADATA_FILE_NAME);
        let reconstructed = Self::metadata_from_artifact(artifact)?;
        Self::validate_loaded_metadata(&reconstructed)?;
        match read_json::<RuntimeArtifactMetadata>(&metadata_path) {
            Ok(sidecar) => {
                Self::validate_loaded_metadata(&sidecar)?;
                Self::validate_metadata_consistency(&sidecar, &reconstructed)?;
                if let Some(embedded) = artifact.runtime_metadata.as_ref() {
                    Self::validate_loaded_metadata(embedded)?;
                    Self::validate_metadata_consistency(&sidecar, embedded)?;
                }
                Ok(sidecar)
            }
            Err(error) => {
                if let Some(embedded) = artifact.runtime_metadata.as_ref() {
                    Self::validate_loaded_metadata(embedded)?;
                    Self::validate_metadata_consistency(embedded, &reconstructed)?;
                    tracing::warn!(
                        "NEAT metadata sidecar unavailable at {} ({}); falling back to embedded metadata",
                        metadata_path.display(),
                        error
                    );
                    Ok(embedded.clone())
                } else {
                    tracing::warn!(
                        "NEAT metadata sidecar unavailable at {} ({}); reconstructing metadata from artifact",
                        metadata_path.display(),
                        error
                    );
                    Ok(reconstructed)
                }
            }
        }
    }
}

impl Default for NeatExpert {
    fn default() -> Self {
        Self::new(1)
    }
}

impl ExpertModel for NeatExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        let (features, feature_columns) = feature_matrix_from_dataframe(x)?;
        let scaler = FeatureScaler::fit(&features)?;
        let scaled = scaler.transform(&features)?;
        let labels = remap_three_class_labels(y)?;
        if scaled.nrows() < 32 {
            bail!(
                "NEAT requires at least 32 rows, received {}",
                scaled.nrows()
            );
        }

        self.config = build_neat_config(scaled.ncols());
        self.feature_columns = feature_columns;
        self.scaler = Some(scaler);
        self.dataset_rows = scaled.nrows();
        let (train_indices, val_indices) = Self::split_train_val_indices(scaled.nrows());
        let train_features = Self::slice_rows(&scaled, &train_indices);
        let val_features = Self::slice_rows(&scaled, &val_indices);
        let train_labels = Self::slice_labels(&labels, &train_indices);
        let val_labels = Self::slice_labels(&labels, &val_indices);
        self.train_rows = train_features.nrows();
        self.val_rows = val_features.nrows();
        self.runtime_backend = NEAT_RUNTIME_BACKEND.to_string();

        let best =
            self.evolve_population(&train_features, &train_labels, &val_features, &val_labels)?;
        self.best_fitness = best.fitness;
        self.best_loss = best.loss;
        self.best_accuracy = best.accuracy;
        self.best_genome = Some(best.genome);
        self.fitted = true;
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        self.ensure_runtime_state_ready()?;
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let (features, _) = feature_matrix_from_dataframe(x)?;
        let scaler = self.scaler.as_ref().context("NEAT scaler missing")?;
        let scaled = scaler.transform(&features)?;
        let genome = self.best_genome.as_ref().context("NEAT genome missing")?;
        evaluate_probabilities(genome, &scaled)
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.ensure_runtime_state_ready()?;

        std::fs::create_dir_all(path)
            .with_context(|| format!("create NEAT model directory {}", path.display()))?;
        let runtime_metadata = try_build_runtime_artifact_metadata(
            NEAT_MODEL_NAME,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
            self.feature_columns.clone(),
            default_three_class_label_mapping(),
            TrainingSummaryMetadata::new(self.dataset_rows, self.train_rows, self.val_rows),
        )?;
        write_json(&path.join(METADATA_FILE_NAME), &runtime_metadata)?;
        write_json(
            &path.join(NEAT_ARTIFACT_FILE_NAME),
            &NeatArtifact {
                config: self.config.clone(),
                generations: self.generations,
                population_size: self.population_size,
                mutation_rate: self.mutation_rate,
                species_elitism: self.species_elitism,
                compatibility_threshold: self.compatibility_threshold,
                immigrant_fraction: self.immigrant_fraction,
                seed: self.seed,
                feature_columns: self.feature_columns.clone(),
                scaler: self.scaler.clone().context("NEAT scaler missing")?,
                best_genome: self.best_genome.clone().context("NEAT genome missing")?,
                fitted: self.fitted,
                dataset_rows: self.dataset_rows,
                train_rows: self.train_rows,
                val_rows: self.val_rows,
                runtime_backend: self.runtime_backend.clone(),
                runtime_backend_kind: Some(neat_runtime_backend_kind(&self.runtime_backend)),
                requested_device_policy: self.requested_device_policy.clone(),
                best_fitness: self.best_fitness,
                best_loss: self.best_loss,
                best_accuracy: self.best_accuracy,
                runtime_metadata: Some(runtime_metadata),
            },
        )
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let artifact: NeatArtifact = read_json(&path.join(NEAT_ARTIFACT_FILE_NAME))?;
        let metadata = Self::resolve_loaded_metadata(path, &artifact)?;
        Self::validate_loaded_artifact(&metadata, &artifact)?;

        let next_config = artifact.config;
        let next_generations = artifact.generations.max(8);
        let next_population_size = artifact.population_size.max(24);
        let next_mutation_rate = artifact.mutation_rate;
        let next_species_elitism = artifact.species_elitism.min(next_population_size);
        let next_compatibility_threshold = artifact.compatibility_threshold.max(0.25);
        let next_immigrant_fraction = artifact.immigrant_fraction.clamp(0.0, 0.4);
        let next_seed = artifact.seed;
        let next_feature_columns = artifact.feature_columns;
        let next_scaler = Some(artifact.scaler);
        let next_best_genome = Some(artifact.best_genome);
        let next_fitted = artifact.fitted;
        let next_dataset_rows = artifact.dataset_rows;
        let next_train_rows = artifact.train_rows;
        let next_val_rows = artifact.val_rows;
        let next_runtime_backend = artifact.runtime_backend;
        let next_requested_device_policy =
            normalize_runtime_device_policy(&artifact.requested_device_policy);
        let next_best_fitness = artifact.best_fitness;
        let next_best_loss = artifact.best_loss;
        let next_best_accuracy = artifact.best_accuracy;

        self.config = next_config;
        self.generations = next_generations;
        self.population_size = next_population_size;
        self.mutation_rate = next_mutation_rate;
        self.species_elitism = next_species_elitism;
        self.compatibility_threshold = next_compatibility_threshold;
        self.immigrant_fraction = next_immigrant_fraction;
        self.seed = next_seed;
        self.feature_columns = next_feature_columns;
        self.scaler = next_scaler;
        self.best_genome = next_best_genome;
        self.fitted = next_fitted;
        self.dataset_rows = next_dataset_rows;
        self.train_rows = next_train_rows;
        self.val_rows = next_val_rows;
        self.runtime_backend = next_runtime_backend;
        self.requested_device_policy = next_requested_device_policy;
        self.best_fitness = next_best_fitness;
        self.best_loss = next_best_loss;
        self.best_accuracy = next_best_accuracy;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::build_runtime_artifact_metadata;
    use crate::statistical::common::read_json;
    use polars::prelude::{DataFrame, NamedFrom, Series};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_model_dir(name: &str) -> std::path::PathBuf {
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

    fn training_frame() -> Result<(DataFrame, Series)> {
        let features = DataFrame::new(vec![
            Series::new(
                "f1".into(),
                (0..32).map(|idx| idx as f64).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "f2".into(),
                (0..32).map(|idx| (idx as f64) * 0.5).collect::<Vec<_>>(),
            )
            .into(),
        ])?;
        let labels = Series::new(
            "target".into(),
            (0..32)
                .map(|idx| match idx % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                })
                .collect::<Vec<_>>(),
        );
        Ok((features, labels))
    }

    #[test]
    fn neat_search_params_allow_zero_species_elitism() {
        let expert = NeatExpert::with_config(2, 24, 8).with_search_params(0.85, 0, 2.5, 0.1, 42);
        assert_eq!(expert.species_elitism, 0);
    }

    #[test]
    fn neat_save_records_train_val_rows_and_runtime_backend() -> Result<()> {
        let (features, labels) = training_frame()?;
        let mut expert = NeatExpert::with_config(2, 24, 8);
        expert.fit(&features, &labels)?;

        let path = temp_model_dir("neat");
        expert.save(&path)?;

        let metadata: crate::runtime::artifacts::RuntimeArtifactMetadata =
            read_json(&path.join(METADATA_FILE_NAME))?;
        assert_eq!(metadata.training_summary.dataset_rows, 32);
        assert_eq!(metadata.training_summary.train_rows, 26);
        assert_eq!(metadata.training_summary.val_rows, 6);

        let artifact: NeatArtifact = read_json(&path.join(NEAT_ARTIFACT_FILE_NAME))?;
        assert_eq!(artifact.train_rows, 26);
        assert_eq!(artifact.val_rows, 6);
        assert_eq!(artifact.runtime_backend, NEAT_RUNTIME_BACKEND);
        assert_eq!(
            artifact.runtime_backend_kind,
            Some(forex_core::BackendKind::NativeCpu)
        );

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn neat_predict_runtime_reports_backend_details() -> Result<()> {
        let (features, labels) = training_frame()?;
        let mut expert = NeatExpert::with_config(2, 24, 8);
        expert.fit(&features, &labels)?;

        let predictions = expert.predict_runtime(&features)?;
        assert_eq!(predictions.len(), 32);
        assert_eq!(
            predictions[0].metadata().execution_backend.as_deref(),
            Some(NEAT_RUNTIME_BACKEND)
        );
        assert_eq!(
            predictions[0].metadata().backend_kind,
            Some(forex_core::BackendKind::NativeCpu)
        );
        assert_eq!(
            predictions[0].metadata().runtime_mode,
            Some(forex_core::RuntimeMode::Canonical)
        );
        assert_eq!(predictions[0].metadata().degraded_reason, None);
        Ok(())
    }

    #[test]
    fn neat_load_falls_back_to_embedded_metadata_when_sidecar_missing() -> Result<()> {
        let (features, labels) = training_frame()?;
        let mut expert = NeatExpert::with_config(2, 24, 8);
        expert.fit(&features, &labels)?;
        let path = temp_model_dir("neat_sidecar_missing");
        expert.save(&path)?;
        std::fs::remove_file(path.join(METADATA_FILE_NAME))?;

        let mut loaded = NeatExpert::with_config(2, 24, 8);
        loaded.load(&path)?;
        assert_eq!(loaded.train_rows + loaded.val_rows, loaded.dataset_rows);
        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn neat_load_rejects_sidecar_drift_against_embedded() -> Result<()> {
        let (features, labels) = training_frame()?;
        let mut expert = NeatExpert::with_config(2, 24, 8);
        expert.fit(&features, &labels)?;
        let path = temp_model_dir("neat_sidecar_drift");
        expert.save(&path)?;

        let mut sidecar: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        sidecar.training_summary.train_rows = sidecar.training_summary.train_rows.saturating_sub(1);
        sidecar.training_summary.val_rows += 1;
        write_json(&path.join(METADATA_FILE_NAME), &sidecar)?;

        let mut loaded = NeatExpert::with_config(2, 24, 8);
        let err = loaded
            .load(&path)
            .expect_err("drifted sidecar metadata should be rejected");
        assert!(err.to_string().contains("drift"));
        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn validate_loaded_artifact_rejects_inconsistent_train_val_rows() -> Result<()> {
        let metadata = build_runtime_artifact_metadata(
            NEAT_MODEL_NAME,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
            vec!["f1".to_string()],
            default_three_class_label_mapping(),
            TrainingSummaryMetadata::new(32, 26, 6),
        );
        let artifact = NeatArtifact {
            config: build_neat_config(1),
            generations: 8,
            population_size: 24,
            mutation_rate: 0.85,
            species_elitism: 1,
            compatibility_threshold: 2.5,
            immigrant_fraction: 0.1,
            seed: 42,
            feature_columns: vec!["f1".to_string()],
            scaler: FeatureScaler {
                means: vec![0.0],
                stds: vec![1.0],
            },
            best_genome: {
                let mut rng = Xoroshiro128PlusPlus::seed_from_u64(42);
                NeatGenome::fully_connected(build_neat_config(1), &mut rng)
            },
            fitted: true,
            dataset_rows: 32,
            train_rows: 32,
            val_rows: 1,
            runtime_backend: NEAT_RUNTIME_BACKEND.to_string(),
            runtime_backend_kind: Some(neat_runtime_backend_kind(NEAT_RUNTIME_BACKEND)),
            requested_device_policy: default_neat_requested_device_policy(),
            best_fitness: 0.5,
            best_loss: 1.0,
            best_accuracy: 0.5,
            runtime_metadata: None,
        };

        let err = NeatExpert::validate_loaded_artifact(&metadata, &artifact)
            .expect_err("inconsistent train/val rows should be rejected");
        assert!(err.to_string().contains("train_rows + val_rows"));
        Ok(())
    }

    #[test]
    fn validate_loaded_artifact_rejects_non_positive_scaler_stds() -> Result<()> {
        let metadata = build_runtime_artifact_metadata(
            NEAT_MODEL_NAME,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
            vec!["f1".to_string()],
            default_three_class_label_mapping(),
            TrainingSummaryMetadata::new(32, 26, 6),
        );
        let artifact = NeatArtifact {
            config: build_neat_config(1),
            generations: 8,
            population_size: 24,
            mutation_rate: 0.85,
            species_elitism: 1,
            compatibility_threshold: 2.5,
            immigrant_fraction: 0.1,
            seed: 42,
            feature_columns: vec!["f1".to_string()],
            scaler: FeatureScaler {
                means: vec![0.0],
                stds: vec![0.0],
            },
            best_genome: {
                let mut rng = Xoroshiro128PlusPlus::seed_from_u64(42);
                NeatGenome::fully_connected(build_neat_config(1), &mut rng)
            },
            fitted: true,
            dataset_rows: 32,
            train_rows: 26,
            val_rows: 6,
            runtime_backend: NEAT_RUNTIME_BACKEND.to_string(),
            runtime_backend_kind: Some(neat_runtime_backend_kind(NEAT_RUNTIME_BACKEND)),
            requested_device_policy: default_neat_requested_device_policy(),
            best_fitness: 0.5,
            best_loss: 1.0,
            best_accuracy: 0.5,
            runtime_metadata: None,
        };

        let err = NeatExpert::validate_loaded_artifact(&metadata, &artifact)
            .expect_err("non-positive scaler stds should be rejected");
        assert!(err.to_string().contains("non-positive stds"));
        Ok(())
    }

    #[test]
    fn validate_loaded_artifact_rejects_out_of_range_mutation_rate() -> Result<()> {
        let metadata = build_runtime_artifact_metadata(
            NEAT_MODEL_NAME,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
            vec!["f1".to_string()],
            default_three_class_label_mapping(),
            TrainingSummaryMetadata::new(32, 26, 6),
        );
        let artifact = NeatArtifact {
            config: build_neat_config(1),
            generations: 8,
            population_size: 24,
            mutation_rate: 1.75,
            species_elitism: 1,
            compatibility_threshold: 2.5,
            immigrant_fraction: 0.1,
            seed: 42,
            feature_columns: vec!["f1".to_string()],
            scaler: FeatureScaler {
                means: vec![0.0],
                stds: vec![1.0],
            },
            best_genome: {
                let mut rng = Xoroshiro128PlusPlus::seed_from_u64(42);
                NeatGenome::fully_connected(build_neat_config(1), &mut rng)
            },
            fitted: true,
            dataset_rows: 32,
            train_rows: 26,
            val_rows: 6,
            runtime_backend: NEAT_RUNTIME_BACKEND.to_string(),
            runtime_backend_kind: Some(neat_runtime_backend_kind(NEAT_RUNTIME_BACKEND)),
            requested_device_policy: default_neat_requested_device_policy(),
            best_fitness: 0.5,
            best_loss: 1.0,
            best_accuracy: 0.5,
            runtime_metadata: None,
        };

        let err = NeatExpert::validate_loaded_artifact(&metadata, &artifact)
            .expect_err("out-of-range mutation rate should be rejected");
        assert!(
            err.to_string()
                .contains("mutation rate is out of contract range")
        );
        Ok(())
    }

    #[test]
    fn predict_proba_rejects_missing_runtime_backend() -> Result<()> {
        let (features, labels) = training_frame()?;
        let mut expert = NeatExpert::with_config(2, 24, 8);
        expert.fit(&features, &labels)?;
        expert.runtime_backend.clear();

        let err = expert
            .predict_proba(&features)
            .expect_err("missing runtime backend should fail early");
        assert!(err.to_string().contains("runtime backend"));
        Ok(())
    }
}
