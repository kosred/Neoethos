//! [`super::ExpertModel`] adapters for the Evolutionary family.
//!
//! Phase D1.2.6 — partial. Covers the three straightforward
//! evolutionary experts that produce Classification3 output:
//! - **genetic** — [`GeneticStrategyExpert`]
//! - **neuro_evo** — [`NeuroEvoExpert`] (CR-FM-NES neuroevolution)
//! - **neat** — [`NeatExpert`] (NeuroEvolution of Augmenting Topologies)
//!
//! All three are pure Rust → `Send + Sync` auto-derived.
//!
//! ## Deferred to D1.2.7
//!
//! The remaining two names on the original D1.2.6 list have
//! non-standard inference APIs and need extra trait/taxonomy
//! support that warrants a focused commit:
//!
//! - **exit_agent** — [`crate::exit_agent::ExitAgent`] outputs a
//!   2-action Q-value vector (hold / exit), not the 3-class
//!   buy/neutral/sell taxonomy. It also exposes
//!   `predict_runtime() -> Vec<RuntimePrediction>` rather than
//!   `predict_proba(&DataFrame)`, and its `load` is a static
//!   constructor `pub fn load(path) -> Result<Self>` (not
//!   `&mut self`). Adapter needs either a new
//!   `ExpertOutputKind::Classification2` / `ExitDecision` variant
//!   in the taxonomy, OR a domain-specific decision (binary
//!   exit-or-hold doesn't aggregate with direction signals the
//!   way SoftVoting expects).
//!
//! - **dqn** — [`crate::rl::TradingReinforcementLearner`] outputs
//!   3-action Q-values via `predict_q_values(&[f32]) -> Vec<f32>`
//!   — per-state, not per-DataFrame. To plug into the ensemble
//!   trait we need (a) row-iteration glue that extracts each row
//!   of the input DataFrame as `&[f32]`, and (b) the decision
//!   between exposing raw `ActionValues3` to the aggregator or
//!   softmax-converting to `Classification3` at the adapter
//!   boundary. Both are real design decisions, not pure mechanics.
//!
//! Both adapters land in D1.2.7 along with the trait extensions
//! they need. Until then, the bot loses 2/33 experts from any
//! ensemble that loads via [`super::ExpertRegistry::load_with_partial`]
//! — those names get reported in the `degraded` list with the
//! "no loader registered" reason from
//! [`super::ExpertLoadError::InvalidArtifact`], which is the
//! correct + honest behaviour.

use std::path::Path;

use anyhow::{Context, Result};
use polars::prelude::DataFrame;

use super::tree_adapters::classification3_per_row;
use super::{ExpertLoader, ExpertModel, ExpertOutputKind, ExpertPrediction};
use crate::base::ExpertModel as BaseExpertModel;
use crate::evolution::{NeatExpert, NeuroEvoExpert};
use crate::genetic::GeneticStrategyExpert;
use crate::runtime::capabilities::ModelFamily;

// ---------------------------------------------------------------------------
// genetic
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`GeneticStrategyExpert`].
pub struct GeneticAdapter {
    inner: GeneticStrategyExpert,
}

impl GeneticAdapter {
    pub fn new(inner: GeneticStrategyExpert) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &GeneticStrategyExpert {
        &self.inner
    }
}

impl ExpertModel for GeneticAdapter {
    fn name(&self) -> &str {
        "genetic"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Evolutionary
    }
    fn output_kind(&self) -> ExpertOutputKind {
        ExpertOutputKind::Classification3
    }
    fn feature_columns(&self) -> &[String] {
        self.inner.feature_columns()
    }
    fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
        // GeneticStrategyExpert has an inherent `predict_proba(&self, df, metadata, symbol)`
        // method that shadows the trait method when called normally;
        // we call the trait method explicitly via UFCS.
        let probs = <GeneticStrategyExpert as BaseExpertModel>::predict_proba(&self.inner, df)
            .with_context(|| "genetic predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`GeneticAdapter`].
pub struct GeneticLoader;
impl ExpertLoader for GeneticLoader {
    fn name(&self) -> &str {
        "genetic"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = GeneticStrategyExpert::new(32, 4, 8)
            .with_context(|| "GeneticStrategyExpert::new failed")?;
        inner.load(artifact_dir).with_context(|| {
            format!(
                "GeneticStrategyExpert::load({}) failed",
                artifact_dir.display()
            )
        })?;
        Ok(Box::new(GeneticAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// neuro_evo
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`NeuroEvoExpert`].
pub struct NeuroEvoAdapter {
    inner: NeuroEvoExpert,
}

impl NeuroEvoAdapter {
    pub fn new(inner: NeuroEvoExpert) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &NeuroEvoExpert {
        &self.inner
    }
}

impl ExpertModel for NeuroEvoAdapter {
    fn name(&self) -> &str {
        "neuro_evo"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Evolutionary
    }
    fn output_kind(&self) -> ExpertOutputKind {
        ExpertOutputKind::Classification3
    }
    fn feature_columns(&self) -> &[String] {
        self.inner.feature_columns()
    }
    fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
        let probs = self
            .inner
            .predict_proba(df)
            .with_context(|| "neuro_evo predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`NeuroEvoAdapter`].
pub struct NeuroEvoLoader;
impl ExpertLoader for NeuroEvoLoader {
    fn name(&self) -> &str {
        "neuro_evo"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        // input_dim is irrelevant — load() rebuilds from disk.
        let mut inner = NeuroEvoExpert::new(1);
        inner
            .load(artifact_dir)
            .with_context(|| format!("NeuroEvoExpert::load({}) failed", artifact_dir.display()))?;
        Ok(Box::new(NeuroEvoAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// neat
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`NeatExpert`].
pub struct NeatAdapter {
    inner: NeatExpert,
}

impl NeatAdapter {
    pub fn new(inner: NeatExpert) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &NeatExpert {
        &self.inner
    }
}

impl ExpertModel for NeatAdapter {
    fn name(&self) -> &str {
        "neat"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Evolutionary
    }
    fn output_kind(&self) -> ExpertOutputKind {
        ExpertOutputKind::Classification3
    }
    fn feature_columns(&self) -> &[String] {
        self.inner.feature_columns()
    }
    fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
        let probs = self
            .inner
            .predict_proba(df)
            .with_context(|| "neat predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`NeatAdapter`].
pub struct NeatLoader;
impl ExpertLoader for NeatLoader {
    fn name(&self) -> &str {
        "neat"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = NeatExpert::new(1);
        inner
            .load(artifact_dir)
            .with_context(|| format!("NeatExpert::load({}) failed", artifact_dir.display()))?;
        Ok(Box::new(NeatAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// Convenience: register-all-evolutionary-loaders
// ---------------------------------------------------------------------------

/// Register the three Classification3 evolutionary loaders.
pub fn register_evolutionary_loaders(registry: &mut super::ExpertRegistry) -> Result<()> {
    registry.register(Box::new(GeneticLoader))?;
    registry.register(Box::new(NeuroEvoLoader))?;
    registry.register(Box::new(NeatLoader))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble_inference::ExpertRegistry;

    #[test]
    fn genetic_adapter_round_trip() {
        let inner = GeneticStrategyExpert::new(32, 4, 8).expect("new");
        let adapter = GeneticAdapter::new(inner);
        assert_eq!(adapter.name(), "genetic");
        assert_eq!(adapter.family(), ModelFamily::Evolutionary);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn neuro_evo_adapter_round_trip() {
        let inner = NeuroEvoExpert::new(1);
        let adapter = NeuroEvoAdapter::new(inner);
        assert_eq!(adapter.name(), "neuro_evo");
        assert_eq!(adapter.family(), ModelFamily::Evolutionary);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn neat_adapter_round_trip() {
        let inner = NeatExpert::new(1);
        let adapter = NeatAdapter::new(inner);
        assert_eq!(adapter.name(), "neat");
        assert_eq!(adapter.family(), ModelFamily::Evolutionary);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn loader_names_match_adapter_names() {
        assert_eq!(GeneticLoader.name(), "genetic");
        assert_eq!(NeuroEvoLoader.name(), "neuro_evo");
        assert_eq!(NeatLoader.name(), "neat");
    }

    #[test]
    fn register_evolutionary_loaders_installs_three_names() {
        let mut reg = ExpertRegistry::new();
        register_evolutionary_loaders(&mut reg).expect("register");
        let mut names = reg.registered_names();
        names.sort_unstable();
        assert_eq!(names, vec!["genetic", "neat", "neuro_evo"]);
    }

    #[test]
    fn full_30_loaders_coexist() {
        let mut reg = ExpertRegistry::new();
        super::super::tree_adapters::register_tree_loaders(&mut reg).expect("trees");
        super::super::deep_classification_adapters::register_deep_classification_loaders(&mut reg)
            .expect("deep-cls");
        super::super::deep_timeseries_adapters::register_deep_timeseries_loaders(&mut reg)
            .expect("deep-ts");
        super::super::meta_adapters::register_meta_loaders(&mut reg).expect("meta");
        super::super::mixed_adapters::register_mixed_loaders(&mut reg).expect("mixed");
        register_evolutionary_loaders(&mut reg).expect("evo");
        // 7 tree + 3 deep-cls + 7 deep-ts + 7 meta + 3 mixed + 3 evo = 30
        assert_eq!(reg.registered_names().len(), 30);
    }
}
