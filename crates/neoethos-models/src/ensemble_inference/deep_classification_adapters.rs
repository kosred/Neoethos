//! [`super::ExpertModel`] adapters for the deep CLASSIFICATION
//! expert families.
//!
//! Phase D1.2.2. This module covers the three "classifier-head"
//! deep experts that produce 3-class softmax probabilities over
//! `[neutral, buy, sell]` (canonical order — see `base.rs` lines
//! 128-135): **MLP**, **KAN** (Kolmogorov-Arnold
//! Networks), and **TabNet**. The forecaster-head deep experts
//! (NBEATS, TiDE, Transformer, PatchTST, TimesNet, NBeatsx_NF,
//! TiDE_NF) emit continuous values and land in D1.2.3 with their
//! own `ExpertOutputKind::Forecast1` adapter family.
//!
//! ## Shared backing struct
//!
//! All three classifiers are macro-generated newtype wrappers
//! around [`crate::deep_models::BurnDeepExpert`] (see
//! `define_deep_expert!` in `deep_models.rs`). They differ only in
//! the `DeepModelKind` they carry — every other surface
//! (predict_proba, feature_columns, save, load) is identical.
//! Their adapters could in principle be macro-collapsed but kept
//! distinct here for grep-ability and so each adapter's canonical
//! name is a literal `&'static str`.
//!
//! ## Send + Sync
//!
//! The deep experts are pure-Rust via the Burn framework's ndarray
//! CPU backend (no FFI / C pointers, unlike the tree experts).
//! However, internally Burn caches tensors in `OnceCell` for
//! lazy-init, and `OnceCell<T>` is `Send` but not `Sync`. The
//! adapter therefore needs an explicit `unsafe impl Sync` with the
//! following SAFETY contract:
//!
//! 1. The inner expert is fully loaded (via [`MlpAdapter::new`] /
//!    [`MlpLoader::load`] etc.) BEFORE any thread calls
//!    [`ExpertModel::predict`] on it.
//! 2. `predict` (the only method the producer calls) is read-only
//!    from the operator-observable surface. After the OnceCell
//!    is initialized on first predict, subsequent reads are
//!    atomic.
//! 3. If the OnceCell triggers a lazy-init DURING a concurrent
//!    second predict, Burn's internal initialization is
//!    idempotent at the value level (same model graph + same
//!    device → same tensor); the worst-case race is a duplicated
//!    init, never a corrupted state.
//!
//! In the producer's actual usage (one predict per closed bar
//! from a single thread), Sync is never exercised. The trait
//! constraint exists for the future MoE aggregator that may
//! parallelise predict across experts.
//!
//! ## Output shape
//!
//! Every classifier head is configured with `n_classes = 3` at
//! construction (see the `with_n_classes(3)` calls in
//! `deep_models.rs::default_*_config`). `predict_proba` therefore
//! returns a `(n_rows, 3)` matrix that maps directly to
//! [`super::ExpertOutputKind::Classification3`] without reordering.

use std::path::Path;

use anyhow::{Context, Result};
use polars::prelude::DataFrame;

use super::{ExpertLoader, ExpertModel, ExpertOutputKind, ExpertPrediction};
use crate::base::ExpertModel as BaseExpertModel;
use crate::deep_models::{KANExpert, MLPExpert, TabNetExpert};
use crate::runtime::capabilities::ModelFamily;
// Shared row-validator: we deliberately re-use the same helper
// the tree adapters use — the predict_proba → ExpertPrediction
// conversion is identical for every Classification3 family.
use super::tree_adapters::classification3_per_row;

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`MLPExpert`].
///
/// Plain feed-forward MLP with `n_classes = 3` softmax head.
pub struct MlpAdapter {
    inner: MLPExpert,
}

// SAFETY: see module-level Sync contract — Burn's OnceCell tensor
// cache is initialized on load and read atomically thereafter.
unsafe impl Sync for MlpAdapter {}

impl MlpAdapter {
    pub fn new(inner: MLPExpert) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &MLPExpert {
        &self.inner
    }
}

impl ExpertModel for MlpAdapter {
    fn name(&self) -> &str {
        "mlp"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Deep
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
            .with_context(|| "mlp predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`MlpAdapter`].
pub struct MlpLoader;

impl ExpertLoader for MlpLoader {
    fn name(&self) -> &str {
        "mlp"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = MLPExpert::new(42, None);
        inner
            .load(artifact_dir)
            .with_context(|| format!("MLPExpert::load({}) failed", artifact_dir.display()))?;
        Ok(Box::new(MlpAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// KAN (Kolmogorov-Arnold Networks)
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`KANExpert`].
///
/// Kolmogorov-Arnold Network classifier with learnable activation
/// functions on edges. 3-class softmax head.
pub struct KanAdapter {
    inner: KANExpert,
}

// SAFETY: see module-level Sync contract.
unsafe impl Sync for KanAdapter {}

impl KanAdapter {
    pub fn new(inner: KANExpert) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &KANExpert {
        &self.inner
    }
}

impl ExpertModel for KanAdapter {
    fn name(&self) -> &str {
        "kan"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Deep
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
            .with_context(|| "kan predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`KanAdapter`].
pub struct KanLoader;

impl ExpertLoader for KanLoader {
    fn name(&self) -> &str {
        "kan"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = KANExpert::new(42, None);
        inner
            .load(artifact_dir)
            .with_context(|| format!("KANExpert::load({}) failed", artifact_dir.display()))?;
        Ok(Box::new(KanAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// TabNet
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`TabNetExpert`].
///
/// Attentive feature-selection deep classifier (Google Brain
/// TabNet architecture). 3-class softmax head.
pub struct TabNetAdapter {
    inner: TabNetExpert,
}

// SAFETY: see module-level Sync contract.
unsafe impl Sync for TabNetAdapter {}

impl TabNetAdapter {
    pub fn new(inner: TabNetExpert) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &TabNetExpert {
        &self.inner
    }
}

impl ExpertModel for TabNetAdapter {
    fn name(&self) -> &str {
        "tabnet"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Deep
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
            .with_context(|| "tabnet predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`TabNetAdapter`].
pub struct TabNetLoader;

impl ExpertLoader for TabNetLoader {
    fn name(&self) -> &str {
        "tabnet"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = TabNetExpert::new(42, None);
        inner
            .load(artifact_dir)
            .with_context(|| format!("TabNetExpert::load({}) failed", artifact_dir.display()))?;
        Ok(Box::new(TabNetAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// Convenience: register-all-deep-classifier-loaders
// ---------------------------------------------------------------------------

/// Register every deep-classifier loader (mlp, kan, tabnet) on the
/// supplied registry. Mirrors the
/// [`super::tree_adapters::register_tree_loaders`] convenience helper
/// — the neoethos-app bootstrap will call this once per session
/// alongside its tree-family counterpart.
pub fn register_deep_classification_loaders(registry: &mut super::ExpertRegistry) -> Result<()> {
    registry.register(Box::new(MlpLoader))?;
    registry.register(Box::new(KanLoader))?;
    registry.register(Box::new(TabNetLoader))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble_inference::ExpertRegistry;

    // --- Adapter construction + naming invariants -----------------------

    #[test]
    fn mlp_adapter_name_family_kind() {
        let inner = MLPExpert::new(42, None);
        let adapter = MlpAdapter::new(inner);
        assert_eq!(adapter.name(), "mlp");
        assert_eq!(adapter.family(), ModelFamily::Deep);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn kan_adapter_name_family_kind() {
        let inner = KANExpert::new(42, None);
        let adapter = KanAdapter::new(inner);
        assert_eq!(adapter.name(), "kan");
        assert_eq!(adapter.family(), ModelFamily::Deep);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn tabnet_adapter_name_family_kind() {
        let inner = TabNetExpert::new(42, None);
        let adapter = TabNetAdapter::new(inner);
        assert_eq!(adapter.name(), "tabnet");
        assert_eq!(adapter.family(), ModelFamily::Deep);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    // --- Loader names match adapter names -------------------------------

    #[test]
    fn loader_names_match_their_adapter_names() {
        assert_eq!(MlpLoader.name(), "mlp");
        assert_eq!(KanLoader.name(), "kan");
        assert_eq!(TabNetLoader.name(), "tabnet");
    }

    // --- register_deep_classification_loaders: all 3 land --------------

    #[test]
    fn register_deep_classification_loaders_installs_three_names() {
        let mut reg = ExpertRegistry::new();
        register_deep_classification_loaders(&mut reg).expect("register");
        let mut names = reg.registered_names();
        names.sort_unstable();
        assert_eq!(names, vec!["kan", "mlp", "tabnet"]);
    }

    #[test]
    fn register_deep_classification_loaders_rejects_double_registration() {
        let mut reg = ExpertRegistry::new();
        register_deep_classification_loaders(&mut reg).expect("first call");
        let err =
            register_deep_classification_loaders(&mut reg).expect_err("second call must error");
        assert!(err.to_string().contains("already registered"));
    }

    // --- Coexistence with tree adapters ---------------------------------
    //
    // The registry must accept tree-family + deep-classification
    // loaders side-by-side without collision. Pin that property
    // so a future name-collision (e.g. someone adding a "tree" prefix
    // to a deep adapter) gets caught here.

    #[test]
    fn deep_and_tree_loaders_coexist_in_one_registry() {
        let mut reg = ExpertRegistry::new();
        super::super::tree_adapters::register_tree_loaders(&mut reg).expect("trees");
        register_deep_classification_loaders(&mut reg).expect("deep");
        let names = reg.registered_names();
        // 7 tree + 3 deep = 10 names total
        assert_eq!(names.len(), 10);
        for required in [
            "lightgbm",
            "xgboost",
            "xgboost_rf",
            "xgboost_dart",
            "catboost",
            "catboost_alt",
            "sklears_tree",
            "mlp",
            "kan",
            "tabnet",
        ] {
            assert!(
                reg.has_loader(required),
                "registry missing loader for '{required}'"
            );
        }
    }

    // --- feature_columns proxies through to inner ----------------------

    #[test]
    fn mlp_adapter_feature_columns_proxies_inner_state() {
        // A freshly constructed MLPExpert has empty feature_columns
        // until fit/load runs. Adapter should report the same.
        let inner = MLPExpert::new(42, None);
        let adapter = MlpAdapter::new(inner);
        assert!(adapter.feature_columns().is_empty());
    }
}
