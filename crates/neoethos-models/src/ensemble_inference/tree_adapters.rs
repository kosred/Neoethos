//! [`super::ExpertModel`] adapters for the 7 tree expert families.
//!
//! Phase D1.2.1. This module bridges the existing tree experts
//! ([`crate::tree_models::LightGBMExpert`],
//! [`crate::tree_models::XGBoostExpert`],
//! [`crate::tree_models::CatBoostExpert`],
//! [`crate::tree_models::SklearsTreeExpert`]) to the uniform
//! [`super::ExpertModel`] trait so the [`super::ExpertRegistry`]
//! and (in D1.3) `SoftVotingEnsemble` can treat them
//! interchangeably with the deep / meta / RL families.
//!
//! ## Why thin adapters instead of `impl ExpertModel for LightGBMExpert`
//!
//! Three of the seven canonical names (`xgboost`, `xgboost_rf`,
//! `xgboost_dart`) share the same backing struct
//! ([`crate::tree_models::XGBoostExpert`]) — they differ only in
//! the booster `variant` config the trainer passed at fit time. Two
//! more (`catboost`, `catboost_alt`) likewise share
//! [`crate::tree_models::CatBoostExpert`]. Adapters carry the
//! **canonical name** as a field so the registry can route each
//! variant to its own artifact directory without the underlying
//! struct having to know which variant it became.
//!
//! ## Output shape
//!
//! Every tree expert's `predict_proba` returns a `(n_rows, 3)`
//! `Array2<f32>` of class probabilities — the training pipeline's
//! `objective: "multi:softprob"` / `num_class=3` config pins this
//! shape across the family. Adapters therefore unconditionally
//! return [`super::ExpertOutputKind::Classification3`] predictions.
//!
//! The column ordering inside each 3-vector matches whatever the
//! training pipeline encoded (the trainer + the adapter agree
//! transitively via the saved model artifact). The aggregator
//! [`super::EnsemblePredictor`] is responsible for mapping these
//! probability columns to trade sides; that mapping is a separate
//! concern from this adapter's per-row pass-through.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array2;
use polars::prelude::DataFrame;

use super::{ExpertLoader, ExpertModel, ExpertOutputKind, ExpertPrediction};
// The training-side trait — provides `predict_proba`, `fit`, `load`,
// `save` on every existing expert struct. We rename to disambiguate
// from our inference-side `super::ExpertModel` trait (same simple
// name, different concern).
use crate::base::ExpertModel as BaseExpertModel;
use crate::runtime::capabilities::ModelFamily;
use crate::tree_models::{CatBoostExpert, LightGBMExpert, SklearsTreeExpert, XGBoostExpert};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a `(n_rows, 3)` probability matrix into one
/// [`ExpertPrediction`] per row, validating each row's invariants
/// (sum ≈ 1, every value finite + in `[0, 1]`).
///
/// Rejects the matrix on the first row whose validate() fails — a
/// caller seeing such an error knows the underlying expert has
/// drifted out of the Classification3 contract (which is a real bug,
/// not a recoverable runtime condition).
///
/// Visible to sibling adapter submodules
/// ([`super::deep_classification_adapters`], etc.) so every
/// Classification3 family shares the same row-validator.
pub(super) fn classification3_per_row(probs: &Array2<f32>) -> Result<Vec<ExpertPrediction>> {
    if probs.ncols() != 3 {
        anyhow::bail!(
            "tree expert predict_proba returned {} columns; ExpertOutputKind::Classification3 \
             requires exactly 3 columns",
            probs.ncols()
        );
    }
    let mut out = Vec::with_capacity(probs.nrows());
    for row in probs.outer_iter() {
        let pred = ExpertPrediction {
            kind: ExpertOutputKind::Classification3,
            values: row.to_vec(),
        };
        pred.validate()?;
        out.push(pred);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// LightGBM
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`LightGBMExpert`].
///
/// SAFETY: [`LightGBMExpert`] holds a `lightgbm3::Booster` which
/// contains a `*mut c_void` to the native LightGBM C handle. Rust
/// auto-derives `!Send + !Sync` for that pointer. We assert `Send
/// + Sync` manually because the LightGBM C API documents
/// `LGBM_BoosterPredictForMat` (the only mutation-free call we ever
/// make from the adapter — see [`Self::predict`]) as thread-safe
/// for concurrent inference against the same booster. We never
/// invoke `fit` / `update` on the inner booster from the adapter,
/// so the only un-Send/un-Sync usage path is closed.
pub struct LightGbmAdapter {
    inner: LightGBMExpert,
}

// SAFETY: see the LightGbmAdapter doc comment. The wrapped
// LightGBM booster is only read (predict_proba) from the adapter,
// and predict is documented thread-safe.
unsafe impl Send for LightGbmAdapter {}
unsafe impl Sync for LightGbmAdapter {}

impl LightGbmAdapter {
    /// Wrap a pre-trained or pre-loaded `LightGBMExpert`. The
    /// adapter takes ownership; the inner expert is reachable via
    /// [`Self::inner`] for direct API access (diagnostics / tests).
    pub fn new(inner: LightGBMExpert) -> Self {
        Self { inner }
    }

    /// Read-only view of the wrapped expert.
    pub fn inner(&self) -> &LightGBMExpert {
        &self.inner
    }
}

impl ExpertModel for LightGbmAdapter {
    fn name(&self) -> &str {
        "lightgbm"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Tree
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
            .with_context(|| "lightgbm predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader that materialises a [`LightGbmAdapter`] from an artifact
/// directory ((`.../lightgbm/model.txt`, `.../lightgbm/runtime.json`,
/// `.../lightgbm/metadata.json`, `.../lightgbm/lightgbm_local_fallback.json`).
pub struct LightGbmLoader;

impl ExpertLoader for LightGbmLoader {
    fn name(&self) -> &str {
        "lightgbm"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = LightGBMExpert::new(0, None);
        inner
            .load(artifact_dir)
            .with_context(|| format!("LightGBMExpert::load({}) failed", artifact_dir.display()))?;
        Ok(Box::new(LightGbmAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// XGBoost (3 canonical names, 1 backing struct)
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`XGBoostExpert`].
///
/// One struct backs three canonical names (`xgboost`,
/// `xgboost_rf`, `xgboost_dart`) — they differ only in the booster
/// `variant` config the trainer applied. The adapter carries the
/// canonical name as a field so each variant routes to its own
/// artifact directory and surfaces to the chrome banner under its
/// own label.
///
/// SAFETY: same contract as [`LightGbmAdapter`] — the inner
/// `xgboost::Booster` holds a `*mut c_void` that auto-derives
/// `!Send + !Sync`. We assert manually because XGBoost's
/// `XGBoosterPredict` C API is documented thread-safe for
/// inference against the same booster, and we never mutate the
/// booster from the adapter.
pub struct XgboostAdapter {
    inner: XGBoostExpert,
    canonical_name: &'static str,
}

unsafe impl Send for XgboostAdapter {}
unsafe impl Sync for XgboostAdapter {}

impl XgboostAdapter {
    /// Wrap a pre-loaded `XGBoostExpert` with the canonical name
    /// it was trained as (one of `xgboost`, `xgboost_rf`,
    /// `xgboost_dart`). Returns an error for any other name —
    /// the registry already routes per name, so a wrong name here
    /// would be a programmer bug.
    pub fn new(inner: XGBoostExpert, canonical_name: &'static str) -> Result<Self> {
        if !matches!(canonical_name, "xgboost" | "xgboost_rf" | "xgboost_dart") {
            anyhow::bail!(
                "XgboostAdapter requires canonical_name in {{xgboost, xgboost_rf, xgboost_dart}}, got '{}'",
                canonical_name
            );
        }
        Ok(Self {
            inner,
            canonical_name,
        })
    }

    pub fn inner(&self) -> &XGBoostExpert {
        &self.inner
    }
}

impl ExpertModel for XgboostAdapter {
    fn name(&self) -> &str {
        self.canonical_name
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Tree
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
            .with_context(|| format!("{} predict_proba failed", self.canonical_name))?;
        classification3_per_row(&probs)
    }
}

/// Loader for one of the three XGBoost canonical variants. The
/// variant is fixed at construction:
/// - [`Self::gbtree`] → `xgboost`
/// - [`Self::random_forest`] → `xgboost_rf`
/// - [`Self::dart`] → `xgboost_dart`
pub struct XgboostLoader {
    canonical_name: &'static str,
}

impl XgboostLoader {
    /// Standard gradient-boosted-tree variant — canonical name
    /// `xgboost`.
    pub fn gbtree() -> Self {
        Self {
            canonical_name: "xgboost",
        }
    }
    /// Random-forest variant — canonical name `xgboost_rf`.
    pub fn random_forest() -> Self {
        Self {
            canonical_name: "xgboost_rf",
        }
    }
    /// DART (dropout) variant — canonical name `xgboost_dart`.
    pub fn dart() -> Self {
        Self {
            canonical_name: "xgboost_dart",
        }
    }
}

impl ExpertLoader for XgboostLoader {
    fn name(&self) -> &str {
        self.canonical_name
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = XGBoostExpert::new(0, None);
        inner.load(artifact_dir).with_context(|| {
            format!(
                "XGBoostExpert::load({}) failed for canonical name {}",
                artifact_dir.display(),
                self.canonical_name
            )
        })?;
        Ok(Box::new(XgboostAdapter::new(inner, self.canonical_name)?))
    }
}

// ---------------------------------------------------------------------------
// CatBoost (2 canonical names, 1 backing struct)
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`CatBoostExpert`].
///
/// One struct backs two canonical names (`catboost`, `catboost_alt`)
/// — they differ only in the hyperparam profile the trainer
/// applied. The adapter carries the canonical name as a field for
/// the same reason as [`XgboostAdapter`].
///
/// SAFETY: same contract as the other tree adapters — the inner
/// CatBoost model handle holds a `*mut c_void`; its predict path is
/// thread-safe and we never mutate from the adapter.
pub struct CatboostAdapter {
    inner: CatBoostExpert,
    canonical_name: &'static str,
}

unsafe impl Send for CatboostAdapter {}
unsafe impl Sync for CatboostAdapter {}

impl CatboostAdapter {
    /// Wrap a pre-loaded `CatBoostExpert` with the canonical name
    /// it was trained as (one of `catboost`, `catboost_alt`).
    pub fn new(inner: CatBoostExpert, canonical_name: &'static str) -> Result<Self> {
        if !matches!(canonical_name, "catboost" | "catboost_alt") {
            anyhow::bail!(
                "CatboostAdapter requires canonical_name in {{catboost, catboost_alt}}, got '{}'",
                canonical_name
            );
        }
        Ok(Self {
            inner,
            canonical_name,
        })
    }

    pub fn inner(&self) -> &CatBoostExpert {
        &self.inner
    }
}

impl ExpertModel for CatboostAdapter {
    fn name(&self) -> &str {
        self.canonical_name
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Tree
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
            .with_context(|| format!("{} predict_proba failed", self.canonical_name))?;
        classification3_per_row(&probs)
    }
}

/// Loader for one of the two CatBoost canonical variants.
pub struct CatboostLoader {
    canonical_name: &'static str,
}

impl CatboostLoader {
    /// Standard CatBoost variant — canonical name `catboost`.
    pub fn standard() -> Self {
        Self {
            canonical_name: "catboost",
        }
    }
    /// Alternative CatBoost variant (different hyperparam profile)
    /// — canonical name `catboost_alt`.
    pub fn alt() -> Self {
        Self {
            canonical_name: "catboost_alt",
        }
    }
}

impl ExpertLoader for CatboostLoader {
    fn name(&self) -> &str {
        self.canonical_name
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = CatBoostExpert::new(0);
        inner.load(artifact_dir).with_context(|| {
            format!(
                "CatBoostExpert::load({}) failed for canonical name {}",
                artifact_dir.display(),
                self.canonical_name
            )
        })?;
        Ok(Box::new(CatboostAdapter::new(inner, self.canonical_name)?))
    }
}

// ---------------------------------------------------------------------------
// Sklears (Rust-native sklearn-style decision tree)
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`SklearsTreeExpert`].
pub struct SklearsTreeAdapter {
    inner: SklearsTreeExpert,
}

impl SklearsTreeAdapter {
    pub fn new(inner: SklearsTreeExpert) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &SklearsTreeExpert {
        &self.inner
    }
}

impl ExpertModel for SklearsTreeAdapter {
    fn name(&self) -> &str {
        "sklears_tree"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Tree
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
            .with_context(|| "sklears_tree predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`SklearsTreeAdapter`].
pub struct SklearsTreeLoader;

impl ExpertLoader for SklearsTreeLoader {
    fn name(&self) -> &str {
        "sklears_tree"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = SklearsTreeExpert::new();
        inner.load(artifact_dir).with_context(|| {
            format!("SklearsTreeExpert::load({}) failed", artifact_dir.display())
        })?;
        Ok(Box::new(SklearsTreeAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// Convenience: register-all-tree-loaders
// ---------------------------------------------------------------------------

/// Register every tree-family loader (all 7 canonical names) on the
/// supplied registry.
///
/// Returns the registry for chaining. Errors on duplicate
/// registration of any one of the names — the caller is responsible
/// for clearing existing tree-family loaders if it wants to swap
/// them out.
///
/// This is the entry point the neoethos-app bootstrap will call once
/// per session to wire the tree family into the ensemble registry.
/// Future families (deep, meta, evolutionary, RL) get their own
/// equivalent `register_*_loaders` helper in their own adapter
/// module.
pub fn register_tree_loaders(registry: &mut super::ExpertRegistry) -> Result<()> {
    registry.register(Box::new(LightGbmLoader))?;
    registry.register(Box::new(XgboostLoader::gbtree()))?;
    registry.register(Box::new(XgboostLoader::random_forest()))?;
    registry.register(Box::new(XgboostLoader::dart()))?;
    registry.register(Box::new(CatboostLoader::standard()))?;
    registry.register(Box::new(CatboostLoader::alt()))?;
    registry.register(Box::new(SklearsTreeLoader))?;
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
    fn xgboost_adapter_rejects_unknown_canonical_name() {
        let inner = XGBoostExpert::new(0, None);
        // Can't `expect_err` because XgboostAdapter doesn't implement
        // Debug (the inner XGBoostExpert holds an opaque C pointer
        // that doesn't either, and Debug isn't required by the
        // ExpertModel trait). Match on the result variants instead.
        match XgboostAdapter::new(inner, "lightgbm") {
            Ok(_) => panic!("XgboostAdapter must reject 'lightgbm' as canonical name"),
            Err(err) => {
                assert!(
                    err.to_string().contains("xgboost"),
                    "wrong error message: {err}"
                );
            }
        }
    }

    #[test]
    fn xgboost_adapter_accepts_three_canonical_names() {
        for name in ["xgboost", "xgboost_rf", "xgboost_dart"] {
            let inner = XGBoostExpert::new(0, None);
            let adapter = XgboostAdapter::new(inner, name).expect("accept");
            assert_eq!(adapter.name(), name);
            assert_eq!(adapter.family(), ModelFamily::Tree);
            assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
        }
    }

    #[test]
    fn catboost_adapter_rejects_unknown_canonical_name() {
        let inner = CatBoostExpert::new(0);
        assert!(CatboostAdapter::new(inner, "catboost_xxx").is_err());
    }

    #[test]
    fn catboost_adapter_accepts_two_canonical_names() {
        for name in ["catboost", "catboost_alt"] {
            let inner = CatBoostExpert::new(0);
            let adapter = CatboostAdapter::new(inner, name).expect("accept");
            assert_eq!(adapter.name(), name);
            assert_eq!(adapter.family(), ModelFamily::Tree);
        }
    }

    #[test]
    fn lightgbm_adapter_name_and_family() {
        let inner = LightGBMExpert::new(0, None);
        let adapter = LightGbmAdapter::new(inner);
        assert_eq!(adapter.name(), "lightgbm");
        assert_eq!(adapter.family(), ModelFamily::Tree);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn sklears_tree_adapter_name_and_family() {
        let inner = SklearsTreeExpert::new();
        let adapter = SklearsTreeAdapter::new(inner);
        assert_eq!(adapter.name(), "sklears_tree");
        assert_eq!(adapter.family(), ModelFamily::Tree);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    // --- Loader names match adapter names -------------------------------

    #[test]
    fn loader_names_match_their_adapter_names() {
        assert_eq!(LightGbmLoader.name(), "lightgbm");
        assert_eq!(XgboostLoader::gbtree().name(), "xgboost");
        assert_eq!(XgboostLoader::random_forest().name(), "xgboost_rf");
        assert_eq!(XgboostLoader::dart().name(), "xgboost_dart");
        assert_eq!(CatboostLoader::standard().name(), "catboost");
        assert_eq!(CatboostLoader::alt().name(), "catboost_alt");
        assert_eq!(SklearsTreeLoader.name(), "sklears_tree");
    }

    // --- register_tree_loaders: all 7 land in the registry --------------

    #[test]
    fn register_tree_loaders_installs_all_seven_canonical_names() {
        let mut reg = ExpertRegistry::new();
        register_tree_loaders(&mut reg).expect("register");
        let mut names = reg.registered_names();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "catboost",
                "catboost_alt",
                "lightgbm",
                "sklears_tree",
                "xgboost",
                "xgboost_dart",
                "xgboost_rf",
            ]
        );
    }

    #[test]
    fn register_tree_loaders_rejects_double_registration() {
        let mut reg = ExpertRegistry::new();
        register_tree_loaders(&mut reg).expect("first call");
        let err = register_tree_loaders(&mut reg).expect_err("second call must error");
        assert!(err.to_string().contains("already registered"));
    }

    // --- classification3_per_row helper invariants ----------------------

    #[test]
    fn classification3_per_row_rejects_wrong_column_count() {
        // Build a (2, 4) matrix — must reject as not-3-columns.
        let probs = Array2::<f32>::zeros((2, 4));
        assert!(classification3_per_row(&probs).is_err());
    }

    #[test]
    fn classification3_per_row_rejects_row_with_invalid_sum() {
        let probs = ndarray::array![[0.5_f32, 0.5, 0.5]]; // sums to 1.5
        assert!(classification3_per_row(&probs).is_err());
    }

    #[test]
    fn classification3_per_row_accepts_well_formed_matrix() {
        let probs = ndarray::array![[0.2_f32, 0.5, 0.3], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]];
        let preds = classification3_per_row(&probs).expect("ok");
        assert_eq!(preds.len(), 3);
        assert_eq!(preds[0].values, vec![0.2, 0.5, 0.3]);
        assert_eq!(preds[1].values, vec![0.7, 0.2, 0.1]);
        assert_eq!(preds[2].values, vec![0.1, 0.1, 0.8]);
        for p in &preds {
            assert_eq!(p.kind, ExpertOutputKind::Classification3);
        }
    }
}
