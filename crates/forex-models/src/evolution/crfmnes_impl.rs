use anyhow::{Context, Result, bail};
use ndarray::Array2;
use polars::prelude::{DataFrame, Series};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoroshiro128PlusPlus;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{cmp::Ordering, f64::consts::PI};

use crate::base::{ExpertModel, build_runtime_artifact_metadata};
use crate::runtime::artifacts::{TrainingSummaryMetadata, default_three_class_label_mapping};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::statistical::common::{
    FeatureScaler, METADATA_FILE_NAME, ensure_feature_columns_match, feature_matrix_from_dataframe,
    read_json, remap_three_class_labels, softmax_rows, write_json,
};

const NEURO_EVO_ARTIFACT_FILE_NAME: &str = "neuro_evo.json";

type NeuroEvoParams = (Array2<f32>, Vec<f32>, Array2<f32>, Vec<f32>);

struct SimpleEvolutionState {
    mean: Vec<f64>,
    sigma: f64,
    population: usize,
    last_candidates: Vec<Vec<f64>>,
    best_weights: Vec<f64>,
    best_fitness: f64,
    rng: Xoroshiro128PlusPlus,
}

impl SimpleEvolutionState {
    fn new(dim: usize, sigma: f64, population: usize) -> Self {
        let mut seeder = rand::rng();
        let mut rng = Xoroshiro128PlusPlus::from_rng(&mut seeder);
        let mean = (0..dim)
            .map(|_| gaussian_sample(&mut rng) * sigma.min(0.25))
            .collect::<Vec<_>>();
        Self {
            mean: mean.clone(),
            sigma: sigma.max(1e-4),
            population: population.max(8),
            last_candidates: Vec::new(),
            best_weights: mean,
            best_fitness: f64::NEG_INFINITY,
            rng,
        }
    }

    fn ask(&mut self) -> Vec<Vec<f64>> {
        let mut candidates = Vec::with_capacity(self.population);
        for _ in 0..self.population {
            let candidate = self
                .mean
                .iter()
                .map(|mean| mean + gaussian_sample(&mut self.rng) * self.sigma)
                .collect::<Vec<_>>();
            candidates.push(candidate);
        }
        self.last_candidates = candidates.clone();
        candidates
    }

    fn tell(&mut self, fitness_values: Vec<f64>) -> Result<()> {
        if fitness_values.len() != self.last_candidates.len() {
            bail!(
                "neuro-evo fallback fitness/candidate mismatch: {} fitness values for {} candidates",
                fitness_values.len(),
                self.last_candidates.len()
            );
        }

        let mut ranked = self
            .last_candidates
            .iter()
            .cloned()
            .zip(fitness_values)
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));

        if let Some((weights, fitness)) = ranked.first() {
            if *fitness > self.best_fitness {
                self.best_fitness = *fitness;
                self.best_weights = weights.clone();
                self.sigma *= 0.96;
            } else {
                self.sigma *= 1.04;
            }
        }

        let elite_count = (self.population / 4).max(2).min(ranked.len());
        let mut next_mean = vec![0.0; self.mean.len()];
        let mut weight_sum = 0.0_f64;
        for (rank, (weights, _)) in ranked.iter().take(elite_count).enumerate() {
            let weight = (elite_count - rank) as f64;
            weight_sum += weight;
            for (idx, value) in weights.iter().enumerate() {
                next_mean[idx] += value * weight;
            }
        }
        if weight_sum > 0.0 {
            for value in &mut next_mean {
                *value /= weight_sum;
            }
            for (idx, value) in self.mean.iter_mut().enumerate() {
                *value = *value * 0.25 + next_mean[idx] * 0.75;
            }
        }

        self.sigma = self.sigma.clamp(1e-4, 2.5);
        Ok(())
    }
}

fn gaussian_sample(rng: &mut Xoroshiro128PlusPlus) -> f64 {
    let u1 = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

enum NeuroEvoBackend {
    Fallback(SimpleEvolutionState),
}

pub struct NeuroEvoOptimizer {
    backend: NeuroEvoBackend,
    pub dim: usize,
}

impl NeuroEvoOptimizer {
    pub fn new(dim: usize, sigma: f64, population: usize) -> Self {
        Self {
            backend: NeuroEvoBackend::Fallback(SimpleEvolutionState::new(dim, sigma, population)),
            dim,
        }
    }

    pub fn ask(&mut self) -> Result<Vec<Vec<f64>>> {
        match &mut self.backend {
            NeuroEvoBackend::Fallback(state) => Ok(state.ask()),
        }
    }

    pub fn tell(&mut self, fitness_values: Vec<f64>) -> Result<()> {
        match &mut self.backend {
            NeuroEvoBackend::Fallback(state) => state.tell(fitness_values),
        }
    }

    pub fn best_weights(&self) -> Result<Vec<f64>> {
        match &self.backend {
            NeuroEvoBackend::Fallback(state) => Ok(state.best_weights.clone()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct NeuroEvoArtifact {
    input_dim: usize,
    hidden_dim: usize,
    sigma: f64,
    generations: usize,
    population: usize,
    islands: usize,
    dataset_rows: usize,
    feature_columns: Vec<String>,
    scaler: FeatureScaler,
    params: Vec<f32>,
    fitted: bool,
}

impl Default for NeuroEvoArtifact {
    fn default() -> Self {
        Self {
            input_dim: 1,
            hidden_dim: 32,
            sigma: 0.25,
            generations: 24,
            population: 16,
            islands: 1,
            dataset_rows: 0,
            feature_columns: Vec::new(),
            scaler: FeatureScaler {
                means: Vec::new(),
                stds: Vec::new(),
            },
            params: Vec::new(),
            fitted: false,
        }
    }
}

pub struct NeuroEvoExpert {
    input_dim: usize,
    hidden_dim: usize,
    sigma: f64,
    generations: usize,
    population: usize,
    islands: usize,
    dataset_rows: usize,
    feature_columns: Vec<String>,
    scaler: Option<FeatureScaler>,
    params: Vec<f32>,
    fitted: bool,
}

impl NeuroEvoExpert {
    pub fn new(input_dim: usize) -> Self {
        Self::with_config(input_dim, 32, 0.25, 24)
    }

    pub fn with_config(
        input_dim: usize,
        hidden_dim: usize,
        sigma: f64,
        generations: usize,
    ) -> Self {
        let safe_input_dim = input_dim.max(1);
        let safe_hidden_dim = hidden_dim.max(8);
        let param_dim = Self::parameter_dim(safe_input_dim, safe_hidden_dim);
        Self {
            input_dim: safe_input_dim,
            hidden_dim: safe_hidden_dim,
            sigma: sigma.max(1e-4),
            generations: generations.max(4),
            population: 16,
            islands: 1,
            dataset_rows: 0,
            feature_columns: Vec::new(),
            scaler: None,
            params: vec![0.0; param_dim],
            fitted: false,
        }
    }

    pub fn with_search_topology(mut self, population: usize, islands: usize) -> Self {
        self.population = population.max(4);
        self.islands = islands.max(1);
        self
    }

    fn parameter_dim(input_dim: usize, hidden_dim: usize) -> usize {
        input_dim * hidden_dim + hidden_dim + hidden_dim * 3 + 3
    }

    fn decode_params(&self, params: &[f32]) -> Result<NeuroEvoParams> {
        let expected = Self::parameter_dim(self.input_dim, self.hidden_dim);
        if params.len() != expected {
            bail!(
                "neuro-evo parameter mismatch: expected {}, received {}",
                expected,
                params.len()
            );
        }

        let mut offset = 0usize;
        let w1_len = self.input_dim * self.hidden_dim;
        let w1 = Array2::from_shape_vec(
            (self.input_dim, self.hidden_dim),
            params[offset..offset + w1_len].to_vec(),
        )
        .context("shape neuro-evo input weights")?;
        offset += w1_len;

        let b1 = params[offset..offset + self.hidden_dim].to_vec();
        offset += self.hidden_dim;

        let w2_len = self.hidden_dim * 3;
        let w2 = Array2::from_shape_vec(
            (self.hidden_dim, 3),
            params[offset..offset + w2_len].to_vec(),
        )
        .context("shape neuro-evo output weights")?;
        offset += w2_len;

        let b2 = params[offset..offset + 3].to_vec();
        Ok((w1, b1, w2, b2))
    }

    fn hidden_activations(features: &Array2<f32>, w1: &Array2<f32>, b1: &[f32]) -> Array2<f32> {
        let mut hidden = Array2::<f32>::zeros((features.nrows(), b1.len()));
        for row in 0..features.nrows() {
            for hidden_idx in 0..b1.len() {
                let mut sum = b1[hidden_idx];
                for feature_idx in 0..features.ncols() {
                    sum += features[(row, feature_idx)] * w1[(feature_idx, hidden_idx)];
                }
                hidden[(row, hidden_idx)] = sum.tanh();
            }
        }
        hidden
    }

    fn logits_from_params(&self, params: &[f32], features: &Array2<f32>) -> Result<Array2<f32>> {
        let (w1, b1, w2, b2) = self.decode_params(params)?;
        let hidden = Self::hidden_activations(features, &w1, &b1);
        let mut logits = Array2::<f32>::zeros((features.nrows(), 3));
        for row in 0..hidden.nrows() {
            for class_idx in 0..3 {
                let mut sum = b2[class_idx];
                for hidden_idx in 0..self.hidden_dim {
                    sum += hidden[(row, hidden_idx)] * w2[(hidden_idx, class_idx)];
                }
                logits[(row, class_idx)] = sum;
            }
        }
        Ok(logits)
    }

    fn loss_for_params(
        &self,
        params: &[f32],
        features: &Array2<f32>,
        labels: &[usize],
    ) -> Result<f64> {
        let probabilities = softmax_rows(&self.logits_from_params(params, features)?);
        let mut loss = 0.0_f64;
        for row in 0..probabilities.nrows() {
            let class_idx = labels[row];
            let probability = probabilities[(row, class_idx)].clamp(1e-6, 1.0 - 1e-6) as f64;
            loss -= probability.ln();
        }
        loss /= probabilities.nrows().max(1) as f64;

        let l2 = params
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            / params.len().max(1) as f64;
        Ok(loss + 1e-4 * l2)
    }
}

impl Default for NeuroEvoExpert {
    fn default() -> Self {
        Self::with_config(1, 32, 0.25, 24)
    }
}

impl ExpertModel for NeuroEvoExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        let (features, feature_columns) = feature_matrix_from_dataframe(x)?;
        let scaler = FeatureScaler::fit(&features)?;
        let scaled = scaler.transform(&features)?;
        let labels = remap_three_class_labels(y)?;
        if scaled.nrows() < 32 {
            bail!(
                "neuro-evo requires at least 32 rows, received {}",
                scaled.nrows()
            );
        }

        self.input_dim = scaled.ncols().max(1);
        self.feature_columns = feature_columns;
        self.scaler = Some(scaler.clone());
        self.dataset_rows = scaled.nrows();
        let param_dim = Self::parameter_dim(self.input_dim, self.hidden_dim);
        let mut best_params = vec![0.0_f32; param_dim];
        let mut best_loss = f64::INFINITY;

        for _ in 0..self.islands.max(1) {
            let mut optimizer = NeuroEvoOptimizer::new(param_dim, self.sigma, self.population);
            for _ in 0..self.generations {
                let candidates = optimizer.ask()?;
                let mut fitness = Vec::with_capacity(candidates.len());
                for candidate in &candidates {
                    let candidate_f32 = candidate
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<_>>();
                    let loss = self.loss_for_params(&candidate_f32, &scaled, &labels)?;
                    if loss < best_loss {
                        best_loss = loss;
                        best_params = candidate_f32;
                    }
                    fitness.push(-loss);
                }
                optimizer.tell(fitness)?;
            }

            if best_loss.is_infinite() {
                let best = optimizer.best_weights()?;
                best_params = best.iter().map(|value| *value as f32).collect();
            }
        }

        self.params = best_params;
        self.fitted = true;
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        if !self.fitted {
            bail!("neuro-evo expert has not been fitted");
        }
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let (features, _) = feature_matrix_from_dataframe(x)?;
        let scaler = self.scaler.as_ref().context("neuro-evo scaler missing")?;
        let scaled = scaler.transform(&features)?;
        Ok(softmax_rows(
            &self.logits_from_params(&self.params, &scaled)?,
        ))
    }

    fn save(&self, path: &Path) -> Result<()> {
        if !self.fitted {
            bail!("neuro-evo expert cannot be saved before fitting");
        }
        std::fs::create_dir_all(path)
            .with_context(|| format!("create neuro-evo directory {}", path.display()))?;
        write_json(
            &path.join(METADATA_FILE_NAME),
            &build_runtime_artifact_metadata(
                "neuro_evo",
                ModelFamily::Evolutionary,
                CapabilityState::Implemented,
                self.feature_columns.clone(),
                default_three_class_label_mapping(),
                TrainingSummaryMetadata::new(self.dataset_rows, self.dataset_rows, 0),
            ),
        )?;
        write_json(
            &path.join(NEURO_EVO_ARTIFACT_FILE_NAME),
            &NeuroEvoArtifact {
                input_dim: self.input_dim,
                hidden_dim: self.hidden_dim,
                sigma: self.sigma,
                generations: self.generations,
                population: self.population,
                islands: self.islands,
                dataset_rows: self.dataset_rows,
                feature_columns: self.feature_columns.clone(),
                scaler: self.scaler.clone().context("neuro-evo scaler missing")?,
                params: self.params.clone(),
                fitted: self.fitted,
            },
        )
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let _: crate::runtime::artifacts::RuntimeArtifactMetadata =
            read_json(&path.join(METADATA_FILE_NAME))?;
        let artifact: NeuroEvoArtifact = read_json(&path.join(NEURO_EVO_ARTIFACT_FILE_NAME))?;
        self.input_dim = artifact.input_dim;
        self.hidden_dim = artifact.hidden_dim;
        self.sigma = artifact.sigma;
        self.generations = artifact.generations;
        self.population = artifact.population.max(4);
        self.islands = artifact.islands.max(1);
        self.dataset_rows = artifact.dataset_rows;
        self.feature_columns = artifact.feature_columns;
        self.scaler = Some(artifact.scaler);
        self.params = artifact.params;
        self.fitted = artifact.fitted;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn neuro_evo_save_records_training_rows() -> Result<()> {
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

        let mut expert = NeuroEvoExpert::with_config(2, 8, 0.25, 4).with_search_topology(4, 1);
        expert.fit(&features, &labels)?;

        let path = temp_model_dir("neuro_evo");
        expert.save(&path)?;

        let metadata: crate::runtime::artifacts::RuntimeArtifactMetadata =
            read_json(&path.join(METADATA_FILE_NAME))?;
        assert_eq!(metadata.training_summary.dataset_rows, 32);
        assert_eq!(metadata.training_summary.train_rows, 32);

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn neuro_evo_predicts_with_saved_params() -> Result<()> {
        let scaler = FeatureScaler {
            means: vec![0.0, 0.0],
            stds: vec![1.0, 1.0],
        };
        let mut expert = NeuroEvoExpert::with_config(2, 8, 0.25, 4);
        expert.feature_columns = vec!["f1".to_string(), "f2".to_string()];
        expert.scaler = Some(scaler);
        expert.params = vec![0.0; NeuroEvoExpert::parameter_dim(2, 8)];
        expert.fitted = true;

        let df = DataFrame::new(vec![
            Series::new("f1".into(), vec![1.0_f64, 2.0]).into(),
            Series::new("f2".into(), vec![0.5_f64, 1.5]).into(),
        ])?;

        let proba = expert.predict_proba(&df)?;
        assert_eq!(proba.nrows(), 2);
        assert_eq!(proba.ncols(), 3);
        Ok(())
    }
}
