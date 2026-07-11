//! [`super::ExpertModel`] adapters for the EVOLUTIONARY expert family:
//! **NEAT** and **NeuroEvo** (CR-FM-NES).
//!
//! ## Why these vote (F-319 revision, operator directive 2026-07-11)
//!
//! F-319 originally excluded `neat` / `neuro_evo` from the voting
//! ensemble as "strategy discoverers that live in neoethos-search".
//! That description was stale: BOTH are trained by the shared expert
//! training path (`training_orchestrator::uses_shared_expert_dispatch`)
//! on the SAME 3-class labels as every other classifier, and both
//! implement the base `ExpertModel` contract end-to-end —
//! [`NeatExpert`] hard-validates `num_outputs == 3` at load, and
//! [`NeuroEvoExpert`] saves `default_three_class_label_mapping()` and
//! emits `softmax_rows` probabilities. They are genuine directional
//! classifiers whose artifacts were being trained and then never
//! loaded — hours of operator compute thrown away every run. The
//! operator's rule is: **every trained model votes unless its job is
//! search, not voting** (only `genetic` qualifies for that exemption).
//!
//! ## Shape
//!
//! Same wrap-the-training-struct pattern as the other adapter
//! modules (tree / deep / meta): the adapter owns the loaded inner
//! expert, delegates `predict_proba`, and validates each row through
//! the shared [`classification3_per_row`] helper. Unlike the
//! Burn-backed deep adapters there is NO `unsafe impl Sync` here:
//! both inner experts are plain ndarray/serde state (no `OnceCell`,
//! no `Rc`), so `Sync` is automatic.

use std::path::Path;

use anyhow::{Context, Result};
use polars::prelude::DataFrame;

use super::tree_adapters::classification3_per_row;
use super::{ExpertLoader, ExpertModel, ExpertOutputKind, ExpertPrediction};
use crate::base::ExpertModel as BaseExpertModel;
use crate::evolution::{NeatExpert, NeuroEvoExpert};
use crate::runtime::capabilities::ModelFamily;

// ---------------------------------------------------------------------------
// neat — NeuroEvolution of Augmenting Topologies, 3-class head
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`NeatExpert`] (NEAT genome network,
/// 3-class softmax head — output count validated at artifact load).
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
            .context("neat predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`NeatAdapter`].
pub struct NeatAdapterLoader;

impl ExpertLoader for NeatAdapterLoader {
    fn name(&self) -> &str {
        "neat"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        // input_dim is a placeholder — `load` restores every dimension,
        // the scaler, the genome and the feature columns from the artifact.
        let mut inner = NeatExpert::new(0);
        inner.load(artifact_dir).with_context(|| {
            format!("NeatExpert::load({}) failed", artifact_dir.display())
        })?;
        Ok(Box::new(NeatAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// neuro_evo — CR-FM-NES-evolved MLP, 3-class softmax head
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`NeuroEvoExpert`] (CR-FM-NES-evolved
/// MLP with a 3-class softmax head and its own feature scaler).
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
            .context("neuro_evo predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`NeuroEvoAdapter`].
pub struct NeuroEvoAdapterLoader;

impl ExpertLoader for NeuroEvoAdapterLoader {
    fn name(&self) -> &str {
        "neuro_evo"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = NeuroEvoExpert::new(0);
        inner.load(artifact_dir).with_context(|| {
            format!("NeuroEvoExpert::load({}) failed", artifact_dir.display())
        })?;
        Ok(Box::new(NeuroEvoAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register both evolutionary voters. Called by
/// [`super::bootstrap::build_default_registry`].
pub fn register_evolution_loaders(registry: &mut super::ExpertRegistry) -> Result<()> {
    registry.register(Box::new(NeatAdapterLoader))?;
    registry.register(Box::new(NeuroEvoAdapterLoader))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_names_families_kinds() {
        let neat = NeatAdapter::new(NeatExpert::new(0));
        assert_eq!(neat.name(), "neat");
        assert_eq!(neat.family(), ModelFamily::Evolutionary);
        assert_eq!(neat.output_kind(), ExpertOutputKind::Classification3);

        let nevo = NeuroEvoAdapter::new(NeuroEvoExpert::new(0));
        assert_eq!(nevo.name(), "neuro_evo");
        assert_eq!(nevo.family(), ModelFamily::Evolutionary);
        assert_eq!(nevo.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn loader_names_match_adapter_names() {
        assert_eq!(NeatAdapterLoader.name(), "neat");
        assert_eq!(NeuroEvoAdapterLoader.name(), "neuro_evo");
    }

    #[test]
    fn loaders_fail_loud_on_missing_artifacts() {
        let dir = std::env::temp_dir().join("neoethos_evo_adapter_missing_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(NeatAdapterLoader.load(&dir).is_err());
        assert!(NeuroEvoAdapterLoader.load(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
