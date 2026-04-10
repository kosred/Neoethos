use anyhow::{bail, Context, Result};
use ndarray::Array2;
use polars::prelude::{DataFrame, Series};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoroshiro128PlusPlus;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{cmp::Ordering, f64::consts::PI};

use crate::base::{
    build_runtime_artifact_metadata, build_runtime_prediction_with_details,
    three_class_runtime_confidence, ExpertModel,
};
use crate::runtime::artifacts::{
    default_three_class_label_mapping, RuntimeArtifactMetadata, TrainingSummaryMetadata,
};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;
use crate::statistical::common::{
    ensure_feature_columns_match, feature_matrix_from_dataframe, read_json,
    remap_three_class_labels, softmax_rows, write_json, FeatureScaler, METADATA_FILE_NAME,
};

const NEURO_EVO_ARTIFACT_FILE_NAME: &str = "neuro_evo.json";
const NEURO_EVO_MODEL_NAME: &str = "neuro_evo";
const FALLBACK_BACKEND_NAME: &str = "simple_es_restart_cpu";
const FALLBACK_DEGRADED_REASON: &str = "crfmnes_backend_degraded_to_simple_es";

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
    train_rows: usize,
    val_rows: usize,
    feature_columns: Vec<String>,
    scaler: FeatureScaler,
    params: Vec<f32>,
    search_backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_degraded_reason: Option<String>,
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
            train_rows: 0,
            val_rows: 0,
            feature_columns: Vec::new(),
            scaler: FeatureScaler {
                means: Vec::new(),
                stds: Vec::new(),
            },
            params: Vec::new(),
            search_backend: FALLBACK_BACKEND_NAME.to_string(),
            runtime_degraded_reason: Some(FALLBACK_DEGRADED_REASON.to_string()),
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
    train_rows: usize,
    val_rows: usize,
    feature_columns: Vec<String>,
    scaler: Option<FeatureScaler>,
    params: Vec<f32>,
    search_backend: String,
    runtime_degraded_reason: Option<String>,
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
            train_rows: 0,
            val_rows: 0,
            feature_columns: Vec::new(),
            scaler: None,
            params: vec![0.0; param_dim],
            search_backend: FALLBACK_BACKEND_NAME.to_string(),
            runtime_degraded_reason: Some(FALLBACK_DEGRADED_REASON.to_string()),
            fitted: false,
        }
    }

    pub fn with_search_topology(mut self, population: usize, islands: usize) -> Self {
        self.population = population.max(4);
        self.islands = islands.max(1);
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

    fn validate_loaded_metadata(metadata: &RuntimeArtifactMetadata) -> Result<()> {
        if metadata.model_name != NEURO_EVO_MODEL_NAME {
            bail!(
                "neuro-evo artifact model mismatch: expected {NEURO_EVO_MODEL_NAME}, got {}",
                metadata.model_name
            );
        }

        if metadata.family != ModelFamily::Evolutionary {
            bail!(
                "neuro-evo artifact family mismatch: expected {:?}, got {:?}",
                ModelFamily::Evolutionary,
                metadata.family
            );
        }

        if metadata.state != CapabilityState::Implemented {
            bail!(
                "neuro-evo artifact state mismatch: expected {:?}, got {:?}",
                CapabilityState::Implemented,
                metadata.state
            );
        }

        if metadata.feature_columns.is_empty() {
            bail!("neuro-evo artifact metadata must contain at least one feature column");
        }

        if metadata.label_mapping != default_three_class_label_mapping() {
            bail!("neuro-evo artifact label mapping mismatch");
        }

        if metadata.training_summary.dataset_rows
            != metadata.training_summary.train_rows + metadata.training_summary.val_rows
        {
            bail!("neuro-evo artifact training summary is inconsistent");
        }
        if metadata.training_summary.train_rows == 0 {
            bail!("neuro-evo artifact training summary must contain positive train_rows");
        }

        Ok(())
    }

    fn validate_loaded_artifact(
        metadata: &RuntimeArtifactMetadata,
        artifact: &NeuroEvoArtifact,
    ) -> Result<()> {
        if !artifact.fitted {
            bail!("neuro-evo artifact must be marked fitted before loading");
        }

        if artifact.input_dim == 0 || artifact.hidden_dim == 0 {
            bail!("neuro-evo artifact has invalid network dimensions");
        }

        if !artifact.sigma.is_finite() || artifact.sigma <= 0.0 {
            bail!("neuro-evo artifact sigma is invalid");
        }

        let expected_params = Self::parameter_dim(artifact.input_dim, artifact.hidden_dim);
        if artifact.params.len() != expected_params {
            bail!(
                "neuro-evo artifact parameter mismatch: expected {}, got {}",
                expected_params,
                artifact.params.len()
            );
        }
        if artifact.params.iter().any(|value| !value.is_finite()) {
            bail!("neuro-evo artifact parameters contain non-finite values");
        }

        if artifact.feature_columns.is_empty() {
            bail!("neuro-evo artifact must contain feature columns");
        }

        if artifact.feature_columns.len() != artifact.scaler.means.len()
            || artifact.feature_columns.len() != artifact.scaler.stds.len()
        {
            bail!(
                "neuro-evo artifact scaler mismatch: feature columns {}, means {}, stds {}",
                artifact.feature_columns.len(),
                artifact.scaler.means.len(),
                artifact.scaler.stds.len()
            );
        }
        if artifact.scaler.means.iter().any(|value| !value.is_finite()) {
            bail!("neuro-evo artifact scaler contains non-finite means");
        }
        if artifact
            .scaler
            .stds
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            bail!("neuro-evo artifact scaler contains non-finite or non-positive stds");
        }

        if artifact.population < 4 || artifact.islands == 0 || artifact.generations < 4 {
            bail!(
                "neuro-evo artifact search topology is invalid: generations={}, population={}, islands={}",
                artifact.generations,
                artifact.population,
                artifact.islands
            );
        }

        if artifact.dataset_rows == 0 {
            bail!("neuro-evo artifact dataset rows must be greater than zero");
        }
        if artifact.train_rows == 0 {
            bail!("neuro-evo artifact train_rows must be greater than zero");
        }
        if artifact.train_rows + artifact.val_rows != artifact.dataset_rows {
            bail!("neuro-evo artifact train_rows + val_rows must equal dataset_rows");
        }
        if artifact.search_backend.trim().is_empty() {
            bail!("neuro-evo artifact must persist a runtime backend label");
        }

        if metadata.feature_columns != artifact.feature_columns {
            bail!("neuro-evo artifact feature columns do not match runtime metadata");
        }

        if metadata.training_summary.dataset_rows != artifact.dataset_rows {
            bail!("neuro-evo artifact dataset row count does not match metadata");
        }
        if metadata.training_summary.train_rows != artifact.train_rows
            || metadata.training_summary.val_rows != artifact.val_rows
        {
            bail!("neuro-evo artifact training summary does not match metadata");
        }
        if artifact.search_backend == FALLBACK_BACKEND_NAME
            && artifact.runtime_degraded_reason.as_deref() != Some(FALLBACK_DEGRADED_REASON)
        {
            bail!("neuro-evo artifact fallback backend must persist the fallback degraded reason");
        }
        if artifact.search_backend != FALLBACK_BACKEND_NAME
            && artifact.runtime_degraded_reason.is_some()
        {
            bail!("neuro-evo artifact non-fallback backend may not persist a degraded reason");
        }

        Ok(())
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

    fn selection_loss_for_params(
        &self,
        params: &[f32],
        train_features: &Array2<f32>,
        train_labels: &[usize],
        val_features: &Array2<f32>,
        val_labels: &[usize],
    ) -> Result<(f64, f64, f64)> {
        let train_loss = self.loss_for_params(params, train_features, train_labels)?;
        let val_loss = if val_labels.is_empty() {
            train_loss
        } else {
            self.loss_for_params(params, val_features, val_labels)?
        };
        let selection_loss = if val_labels.is_empty() {
            train_loss
        } else {
            0.65 * train_loss + 0.35 * val_loss
        };
        Ok((selection_loss, train_loss, val_loss))
    }

    fn runtime_details(&self) -> (Option<String>, Option<String>) {
        if !self.fitted {
            return (
                Some("neuro_evo_unknown".to_string()),
                Some("neuro_evo_not_fitted".to_string()),
            );
        }
        if self.feature_columns.is_empty() {
            return (
                Some("neuro_evo_unknown".to_string()),
                Some("neuro_evo_feature_schema_missing".to_string()),
            );
        }
        if self.scaler.is_none() || self.params.is_empty() {
            return (
                Some("neuro_evo_unknown".to_string()),
                Some("neuro_evo_runtime_state_incomplete".to_string()),
            );
        }
        let backend = if self.search_backend.trim().is_empty() {
            "neuro_evo_unknown".to_string()
        } else {
            self.search_backend.clone()
        };
        let degraded_reason = if backend == FALLBACK_BACKEND_NAME {
            Some(FALLBACK_DEGRADED_REASON.to_string())
        } else {
            self.runtime_degraded_reason.clone()
        };
        (Some(backend), degraded_reason)
    }

    fn ensure_runtime_state_ready(&self) -> Result<()> {
        if !self.fitted {
            bail!("neuro-evo expert has not been fitted");
        }
        if self.feature_columns.is_empty() {
            bail!("neuro-evo feature schema missing");
        }
        if self.scaler.is_none() {
            bail!("neuro-evo scaler missing");
        }
        if self.params.is_empty() {
            bail!("neuro-evo parameters missing");
        }
        if self.dataset_rows == 0 || self.train_rows == 0 {
            bail!("neuro-evo training summary is incomplete");
        }
        if self.train_rows + self.val_rows != self.dataset_rows {
            bail!("neuro-evo training summary is inconsistent");
        }
        let scaler = self.scaler.as_ref().context("neuro-evo scaler missing")?;
        if scaler.means.len() != self.feature_columns.len()
            || scaler.stds.len() != self.feature_columns.len()
        {
            bail!("neuro-evo scaler does not match feature schema");
        }
        if scaler.means.iter().any(|value| !value.is_finite()) {
            bail!("neuro-evo scaler contains non-finite means");
        }
        if scaler
            .stds
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            bail!("neuro-evo scaler contains non-finite or non-positive stds");
        }
        if self.params.iter().any(|value| !value.is_finite()) {
            bail!("neuro-evo parameters contain non-finite values");
        }
        Ok(())
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        self.ensure_runtime_state_ready()?;
        let probabilities = self.predict_proba(x)?;
        let (execution_backend, degraded_reason) = self.runtime_details();
        let mut predictions = Vec::with_capacity(probabilities.nrows());
        for row in probabilities.outer_iter() {
            let row_values = [row[0], row[1], row[2]];
            let (confidence, abstain_recommended) = three_class_runtime_confidence(row_values)?;
            predictions.push(build_runtime_prediction_with_details(
                NEURO_EVO_MODEL_NAME.to_string(),
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
        let (train_indices, val_indices) = Self::split_train_val_indices(scaled.nrows());
        let train_features = Self::slice_rows(&scaled, &train_indices);
        let val_features = Self::slice_rows(&scaled, &val_indices);
        let train_labels = Self::slice_labels(&labels, &train_indices);
        let val_labels = Self::slice_labels(&labels, &val_indices);
        self.train_rows = train_features.nrows();
        self.val_rows = val_features.nrows();
        self.search_backend = FALLBACK_BACKEND_NAME.to_string();
        self.runtime_degraded_reason = Some(FALLBACK_DEGRADED_REASON.to_string());
        let param_dim = Self::parameter_dim(self.input_dim, self.hidden_dim);
        let mut best_params = vec![0.0_f32; param_dim];
        let mut best_selection_loss = f64::INFINITY;

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
                    let (selection_loss, _train_loss, _val_loss) = self.selection_loss_for_params(
                        &candidate_f32,
                        &train_features,
                        &train_labels,
                        &val_features,
                        &val_labels,
                    )?;
                    if selection_loss < best_selection_loss {
                        best_selection_loss = selection_loss;
                        best_params = candidate_f32;
                    }
                    fitness.push(-selection_loss);
                }
                optimizer.tell(fitness)?;
            }

            if best_selection_loss.is_infinite() {
                let best = optimizer.best_weights()?;
                best_params = best.iter().map(|value| *value as f32).collect();
            }
        }

        self.params = best_params;
        self.fitted = true;
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        self.ensure_runtime_state_ready()?;
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let (features, _) = feature_matrix_from_dataframe(x)?;
        let scaler = self.scaler.as_ref().context("neuro-evo scaler missing")?;
        let scaled = scaler.transform(&features)?;
        Ok(softmax_rows(
            &self.logits_from_params(&self.params, &scaled)?,
        ))
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.ensure_runtime_state_ready()?;
        std::fs::create_dir_all(path)
            .with_context(|| format!("create neuro-evo directory {}", path.display()))?;
        write_json(
            &path.join(METADATA_FILE_NAME),
            &build_runtime_artifact_metadata(
                NEURO_EVO_MODEL_NAME,
                ModelFamily::Evolutionary,
                CapabilityState::Implemented,
                self.feature_columns.clone(),
                default_three_class_label_mapping(),
                TrainingSummaryMetadata::new(self.dataset_rows, self.train_rows, self.val_rows),
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
                train_rows: self.train_rows,
                val_rows: self.val_rows,
                feature_columns: self.feature_columns.clone(),
                scaler: self.scaler.clone().context("neuro-evo scaler missing")?,
                params: self.params.clone(),
                search_backend: self.search_backend.clone(),
                runtime_degraded_reason: self.runtime_degraded_reason.clone(),
                fitted: self.fitted,
            },
        )
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        let artifact: NeuroEvoArtifact = read_json(&path.join(NEURO_EVO_ARTIFACT_FILE_NAME))?;
        Self::validate_loaded_metadata(&metadata)?;
        Self::validate_loaded_artifact(&metadata, &artifact)?;

        let next_input_dim = artifact.input_dim;
        let next_hidden_dim = artifact.hidden_dim;
        let next_sigma = artifact.sigma;
        let next_generations = artifact.generations.max(4);
        let next_population = artifact.population.max(4);
        let next_islands = artifact.islands.max(1);
        let next_dataset_rows = artifact.dataset_rows;
        let next_train_rows = artifact.train_rows;
        let next_val_rows = artifact.val_rows;
        let next_feature_columns = artifact.feature_columns;
        let next_scaler = Some(artifact.scaler);
        let next_params = artifact.params;
        let next_search_backend = artifact.search_backend;
        let next_runtime_degraded_reason = artifact.runtime_degraded_reason;
        let next_fitted = artifact.fitted;

        self.input_dim = next_input_dim;
        self.hidden_dim = next_hidden_dim;
        self.sigma = next_sigma;
        self.generations = next_generations;
        self.population = next_population;
        self.islands = next_islands;
        self.dataset_rows = next_dataset_rows;
        self.train_rows = next_train_rows;
        self.val_rows = next_val_rows;
        self.feature_columns = next_feature_columns;
        self.scaler = next_scaler;
        self.params = next_params;
        self.search_backend = next_search_backend;
        self.runtime_degraded_reason = next_runtime_degraded_reason;
        self.fitted = next_fitted;
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
        assert_eq!(metadata.training_summary.train_rows, 26);
        assert_eq!(metadata.training_summary.val_rows, 6);

        let artifact: NeuroEvoArtifact = read_json(&path.join(NEURO_EVO_ARTIFACT_FILE_NAME))?;
        assert_eq!(artifact.train_rows, 26);
        assert_eq!(artifact.val_rows, 6);
        assert_eq!(artifact.search_backend, FALLBACK_BACKEND_NAME);
        assert_eq!(
            artifact.runtime_degraded_reason.as_deref(),
            Some(FALLBACK_DEGRADED_REASON)
        );

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

    #[test]
    fn neuro_evo_predict_runtime_reports_truthful_fallback_backend() -> Result<()> {
        let scaler = FeatureScaler {
            means: vec![0.0, 0.0],
            stds: vec![1.0, 1.0],
        };
        let mut expert = NeuroEvoExpert::with_config(2, 8, 0.25, 4);
        expert.feature_columns = vec!["f1".to_string(), "f2".to_string()];
        expert.scaler = Some(scaler);
        expert.params = vec![0.0; NeuroEvoExpert::parameter_dim(2, 8)];
        expert.dataset_rows = 32;
        expert.train_rows = 26;
        expert.val_rows = 6;
        expert.search_backend = FALLBACK_BACKEND_NAME.to_string();
        expert.runtime_degraded_reason = Some(FALLBACK_DEGRADED_REASON.to_string());
        expert.fitted = true;

        let df = DataFrame::new(vec![
            Series::new("f1".into(), vec![1.0_f64, 2.0]).into(),
            Series::new("f2".into(), vec![0.5_f64, 1.5]).into(),
        ])?;

        let predictions = expert.predict_runtime(&df)?;
        assert_eq!(predictions.len(), 2);
        assert_eq!(
            predictions[0].metadata().execution_backend.as_deref(),
            Some(FALLBACK_BACKEND_NAME)
        );
        assert_eq!(
            predictions[0].metadata().degraded_reason.as_deref(),
            Some(FALLBACK_DEGRADED_REASON)
        );
        Ok(())
    }

    #[test]
    fn validate_loaded_artifact_rejects_inconsistent_train_val_rows() -> Result<()> {
        let metadata = build_runtime_artifact_metadata(
            NEURO_EVO_MODEL_NAME,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
            vec!["f1".to_string()],
            default_three_class_label_mapping(),
            TrainingSummaryMetadata::new(32, 26, 6),
        );
        let artifact = NeuroEvoArtifact {
            input_dim: 1,
            hidden_dim: 8,
            sigma: 0.25,
            generations: 4,
            population: 4,
            islands: 1,
            dataset_rows: 32,
            train_rows: 32,
            val_rows: 1,
            feature_columns: vec!["f1".to_string()],
            scaler: FeatureScaler {
                means: vec![0.0],
                stds: vec![1.0],
            },
            params: vec![0.0; NeuroEvoExpert::parameter_dim(1, 8)],
            search_backend: FALLBACK_BACKEND_NAME.to_string(),
            runtime_degraded_reason: Some(FALLBACK_DEGRADED_REASON.to_string()),
            fitted: true,
        };

        let err = NeuroEvoExpert::validate_loaded_artifact(&metadata, &artifact)
            .expect_err("inconsistent train/val rows should be rejected");
        assert!(err.to_string().contains("train_rows + val_rows"));
        Ok(())
    }

    #[test]
    fn validate_loaded_artifact_rejects_fallback_backend_without_reason() -> Result<()> {
        let metadata = build_runtime_artifact_metadata(
            NEURO_EVO_MODEL_NAME,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
            vec!["f1".to_string()],
            default_three_class_label_mapping(),
            TrainingSummaryMetadata::new(32, 26, 6),
        );
        let artifact = NeuroEvoArtifact {
            input_dim: 1,
            hidden_dim: 8,
            sigma: 0.25,
            generations: 4,
            population: 4,
            islands: 1,
            dataset_rows: 32,
            train_rows: 26,
            val_rows: 6,
            feature_columns: vec!["f1".to_string()],
            scaler: FeatureScaler {
                means: vec![0.0],
                stds: vec![1.0],
            },
            params: vec![0.0; NeuroEvoExpert::parameter_dim(1, 8)],
            search_backend: FALLBACK_BACKEND_NAME.to_string(),
            runtime_degraded_reason: None,
            fitted: true,
        };

        let err = NeuroEvoExpert::validate_loaded_artifact(&metadata, &artifact)
            .expect_err("fallback backend without degraded reason should be rejected");
        assert!(err.to_string().contains("fallback degraded reason"));
        Ok(())
    }

    #[test]
    fn validate_loaded_artifact_rejects_non_fallback_backend_with_degraded_reason() -> Result<()> {
        let metadata = build_runtime_artifact_metadata(
            NEURO_EVO_MODEL_NAME,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
            vec!["f1".to_string()],
            default_three_class_label_mapping(),
            TrainingSummaryMetadata::new(32, 26, 6),
        );
        let artifact = NeuroEvoArtifact {
            input_dim: 1,
            hidden_dim: 8,
            sigma: 0.25,
            generations: 4,
            population: 4,
            islands: 1,
            dataset_rows: 32,
            train_rows: 26,
            val_rows: 6,
            feature_columns: vec!["f1".to_string()],
            scaler: FeatureScaler {
                means: vec![0.0],
                stds: vec![1.0],
            },
            params: vec![0.0; NeuroEvoExpert::parameter_dim(1, 8)],
            search_backend: "crfmnes_cpu".to_string(),
            runtime_degraded_reason: Some("stale_degraded_reason".to_string()),
            fitted: true,
        };

        let err = NeuroEvoExpert::validate_loaded_artifact(&metadata, &artifact)
            .expect_err("non-fallback backend with degraded reason should be rejected");
        assert!(err
            .to_string()
            .contains("non-fallback backend may not persist a degraded reason"));
        Ok(())
    }

    #[test]
    fn predict_proba_rejects_incomplete_runtime_state() {
        let expert = NeuroEvoExpert::with_config(2, 8, 0.25, 4);
        let df = DataFrame::new(vec![
            Series::new("f1".into(), vec![1.0_f64, 2.0]).into(),
            Series::new("f2".into(), vec![0.5_f64, 1.5]).into(),
        ])
        .expect("valid dataframe");

        let err = expert
            .predict_proba(&df)
            .expect_err("incomplete runtime state should fail early");
        assert!(err.to_string().contains("not been fitted"));
    }
}
