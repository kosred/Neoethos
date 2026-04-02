use anyhow::{Context, Result, bail};
use ndarray::Array2;
use polars::prelude::{DataFrame, Series};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoroshiro128PlusPlus;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::path::Path;
use symbios_genetics::Genotype;
use symbios_neat::{Activation, CppnEvaluator, NeatConfig, NeatGenome};

use crate::base::{ExpertModel, build_runtime_artifact_metadata};
use crate::runtime::artifacts::{TrainingSummaryMetadata, default_three_class_label_mapping};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::statistical::common::{
    FeatureScaler, METADATA_FILE_NAME, ensure_feature_columns_match, feature_matrix_from_dataframe,
    read_json, remap_three_class_labels, softmax_rows, write_json,
};

const NEAT_ARTIFACT_FILE_NAME: &str = "neat.json";

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
    best_fitness: f32,
    best_loss: f32,
    best_accuracy: f32,
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
            species_elitism: 1,
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
            best_fitness: f32::NEG_INFINITY,
            best_loss: f32::INFINITY,
            best_accuracy: 0.0,
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
            species_elitism: 1,
            compatibility_threshold: 2.5,
            immigrant_fraction: 0.1,
            seed: 42,
            feature_columns: Vec::new(),
            scaler: None,
            best_genome: None,
            fitted: false,
            dataset_rows: 0,
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
        self.species_elitism = species_elitism.max(1);
        self.compatibility_threshold = compatibility_threshold.max(0.25);
        self.immigrant_fraction = immigrant_fraction.clamp(0.0, 0.4);
        self.seed = seed;
        self
    }

    fn evolve_population(
        &mut self,
        features: &Array2<f32>,
        labels: &[usize],
    ) -> Result<GenomeScore> {
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(self.seed);
        let mut population = build_seed_population(&self.config, self.population_size, &mut rng);
        let evaluator = NeatDatasetEvaluator { features, labels };
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

        for _generation in 0..self.generations {
            let mut scores = population
                .iter()
                .map(|genome| evaluator.evaluate(genome))
                .collect::<Vec<_>>();
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
                let elite_count = self.species_elitism.min(bucket.members.len()).max(1);
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

        Ok(best)
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

        let best = self.evolve_population(&scaled, &labels)?;
        self.best_fitness = best.fitness;
        self.best_loss = best.loss;
        self.best_accuracy = best.accuracy;
        self.best_genome = Some(best.genome);
        self.fitted = true;
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        if !self.fitted {
            bail!("NEAT expert has not been fitted");
        }
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let (features, _) = feature_matrix_from_dataframe(x)?;
        let scaler = self.scaler.as_ref().context("NEAT scaler missing")?;
        let scaled = scaler.transform(&features)?;
        let genome = self.best_genome.as_ref().context("NEAT genome missing")?;
        evaluate_probabilities(genome, &scaled)
    }

    fn save(&self, path: &Path) -> Result<()> {
        if !self.fitted {
            bail!("NEAT expert cannot be saved before fitting");
        }

        std::fs::create_dir_all(path)
            .with_context(|| format!("create NEAT model directory {}", path.display()))?;
        write_json(
            &path.join(METADATA_FILE_NAME),
            &build_runtime_artifact_metadata(
                "neat",
                ModelFamily::Evolutionary,
                CapabilityState::Implemented,
                self.feature_columns.clone(),
                default_three_class_label_mapping(),
                TrainingSummaryMetadata::new(self.dataset_rows, self.dataset_rows, 0),
            ),
        )?;
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
                best_fitness: self.best_fitness,
                best_loss: self.best_loss,
                best_accuracy: self.best_accuracy,
            },
        )
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let _: crate::runtime::artifacts::RuntimeArtifactMetadata =
            read_json(&path.join(METADATA_FILE_NAME))?;
        let artifact: NeatArtifact = read_json(&path.join(NEAT_ARTIFACT_FILE_NAME))?;
        self.config = artifact.config;
        self.generations = artifact.generations.max(8);
        self.population_size = artifact.population_size.max(24);
        self.mutation_rate = artifact.mutation_rate.clamp(0.05, 1.5);
        self.species_elitism = artifact.species_elitism.max(1);
        self.compatibility_threshold = artifact.compatibility_threshold.max(0.25);
        self.immigrant_fraction = artifact.immigrant_fraction.clamp(0.0, 0.4);
        self.seed = artifact.seed;
        self.feature_columns = artifact.feature_columns;
        self.scaler = Some(artifact.scaler);
        self.best_genome = Some(artifact.best_genome);
        self.fitted = artifact.fitted;
        self.dataset_rows = artifact.dataset_rows;
        self.best_fitness = artifact.best_fitness;
        self.best_loss = artifact.best_loss;
        self.best_accuracy = artifact.best_accuracy;
        Ok(())
    }
}
