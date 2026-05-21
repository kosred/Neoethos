//! [`super::ExpertModel`] adapters for the 7 META family experts.
//!
//! Phase D1.2.4. Covers:
//! - **elasticnet** ([`ElasticNetExpert`]) — sparse linear classifier
//! - **logistic** ([`LogisticExpert`]) — plain logistic regression
//! - **bayes_logit** ([`BayesianLogitExpert`]) — Bayesian logistic
//! - **meta_blender** ([`MetaBlender`]) — XGBoost meta-head
//! - **probability_calibrator** ([`ProbabilityCalibrationExpert`])
//! - **conformal_gate** ([`ConformalPredictionExpert`])
//! - **meta_stack** ([`MetaDecisionStack`]) — full meta pipeline
//!
//! All seven emit 3-class probabilities → [`super::ExpertOutputKind::Classification3`].
//!
//! ## Send + Sync
//!
//! - The three statistical experts (`elasticnet`, `logistic`,
//!   `bayes_logit`) are pure Rust → auto Send + Sync.
//! - The four meta experts that wrap [`MetaBlender`] internally
//!   (`meta_blender`, `probability_calibrator`, `conformal_gate`,
//!   `meta_stack`) inherit the [`XGBoostExpert`]'s C-FFI handle —
//!   same `*mut c_void` Send/Sync issue as the tree adapters
//!   (D1.2.1). They each declare `unsafe impl Send + Sync` with
//!   the SAFETY contract documented at the tree adapter level
//!   (predict-time read-only access against a thread-safe C API).

use std::path::Path;

use anyhow::{Context, Result};
use polars::prelude::DataFrame;

use super::tree_adapters::classification3_per_row;
use super::{ExpertLoader, ExpertModel, ExpertOutputKind, ExpertPrediction};
use crate::base::ExpertModel as BaseExpertModel;
use crate::ensemble::{
    CalibrationMethod, ConformalPredictionExpert, MetaBlender, MetaDecisionStack,
    ProbabilityCalibrationExpert,
};
use crate::runtime::capabilities::ModelFamily;
use crate::statistical::{BayesianLogitExpert, ElasticNetExpert, LogisticExpert};

// ---------------------------------------------------------------------------
// Shared default calibration method for the meta wrappers
// ---------------------------------------------------------------------------
//
// At load() time we have to construct an empty expert first and
// then `load(path)` rebuilds its state from disk. The calibration
// method passed to `new` is irrelevant — `load` overrides it from
// the persisted artifact. We pick `Platt` as the default-default
// to match the operator config `models.calibration_method: platt`
// in config.yaml.
const DEFAULT_LOAD_CALIBRATION_METHOD: CalibrationMethod = CalibrationMethod::Platt;
/// Default conformal alpha at load() time — same rationale: `load`
/// overrides this from disk. 0.10 matches the operator config.
const DEFAULT_LOAD_CONFORMAL_ALPHA: f32 = 0.10;

// ---------------------------------------------------------------------------
// elasticnet
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`ElasticNetExpert`].
pub struct ElasticNetAdapter {
    inner: ElasticNetExpert,
}

impl ElasticNetAdapter {
    pub fn new(inner: ElasticNetExpert) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &ElasticNetExpert {
        &self.inner
    }
}

impl ExpertModel for ElasticNetAdapter {
    fn name(&self) -> &str {
        "elasticnet"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Meta
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
            .with_context(|| "elasticnet predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`ElasticNetAdapter`].
pub struct ElasticNetLoader;
impl ExpertLoader for ElasticNetLoader {
    fn name(&self) -> &str {
        "elasticnet"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        // ElasticNetExpert::new takes (alpha, l1_ratio) — load()
        // overrides these from disk so any positive defaults work.
        let mut inner = ElasticNetExpert::new(0.01, 0.5);
        inner.load(artifact_dir).with_context(|| {
            format!("ElasticNetExpert::load({}) failed", artifact_dir.display())
        })?;
        Ok(Box::new(ElasticNetAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// logistic
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`LogisticExpert`].
pub struct LogisticAdapter {
    inner: LogisticExpert,
}

impl LogisticAdapter {
    pub fn new(inner: LogisticExpert) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &LogisticExpert {
        &self.inner
    }
}

impl ExpertModel for LogisticAdapter {
    fn name(&self) -> &str {
        "logistic"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Meta
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
            .with_context(|| "logistic predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`LogisticAdapter`].
pub struct LogisticLoader;
impl ExpertLoader for LogisticLoader {
    fn name(&self) -> &str {
        "logistic"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = LogisticExpert::new();
        inner
            .load(artifact_dir)
            .with_context(|| format!("LogisticExpert::load({}) failed", artifact_dir.display()))?;
        Ok(Box::new(LogisticAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// bayes_logit
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`BayesianLogitExpert`].
pub struct BayesLogitAdapter {
    inner: BayesianLogitExpert,
}

impl BayesLogitAdapter {
    pub fn new(inner: BayesianLogitExpert) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &BayesianLogitExpert {
        &self.inner
    }
}

impl ExpertModel for BayesLogitAdapter {
    fn name(&self) -> &str {
        "bayes_logit"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Meta
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
            .with_context(|| "bayes_logit predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`BayesLogitAdapter`].
pub struct BayesLogitLoader;
impl ExpertLoader for BayesLogitLoader {
    fn name(&self) -> &str {
        "bayes_logit"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = BayesianLogitExpert::new();
        inner.load(artifact_dir).with_context(|| {
            format!(
                "BayesianLogitExpert::load({}) failed",
                artifact_dir.display()
            )
        })?;
        Ok(Box::new(BayesLogitAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// meta_blender — wraps XGBoostExpert (C FFI → unsafe impl Send/Sync)
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`MetaBlender`].
///
/// SAFETY: inherits the C-FFI Send/Sync contract from
/// [`crate::tree_models::XGBoostExpert`] (the meta blender's
/// backing struct). See [`super::tree_adapters::XgboostAdapter`]
/// for the full SAFETY rationale (predict-only inference, no
/// mutation from the adapter).
pub struct MetaBlenderAdapter {
    inner: MetaBlender,
}

unsafe impl Send for MetaBlenderAdapter {}
unsafe impl Sync for MetaBlenderAdapter {}

impl MetaBlenderAdapter {
    pub fn new(inner: MetaBlender) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &MetaBlender {
        &self.inner
    }
}

impl ExpertModel for MetaBlenderAdapter {
    fn name(&self) -> &str {
        "meta_blender"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Meta
    }
    fn output_kind(&self) -> ExpertOutputKind {
        ExpertOutputKind::Classification3
    }
    fn feature_columns(&self) -> &[String] {
        &self.inner.feature_columns
    }
    fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
        let probs = self
            .inner
            .predict_proba(df)
            .with_context(|| "meta_blender predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`MetaBlenderAdapter`].
pub struct MetaBlenderLoader;
impl ExpertLoader for MetaBlenderLoader {
    fn name(&self) -> &str {
        "meta_blender"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = MetaBlender::new();
        inner
            .load(artifact_dir)
            .with_context(|| format!("MetaBlender::load({}) failed", artifact_dir.display()))?;
        Ok(Box::new(MetaBlenderAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// probability_calibrator
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`ProbabilityCalibrationExpert`].
///
/// SAFETY: inherits C-FFI Send/Sync contract via internal
/// [`MetaBlender`] (which holds XGBoost). See
/// [`MetaBlenderAdapter`].
pub struct ProbabilityCalibratorAdapter {
    inner: ProbabilityCalibrationExpert,
}

unsafe impl Send for ProbabilityCalibratorAdapter {}
unsafe impl Sync for ProbabilityCalibratorAdapter {}

impl ProbabilityCalibratorAdapter {
    pub fn new(inner: ProbabilityCalibrationExpert) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &ProbabilityCalibrationExpert {
        &self.inner
    }
}

impl ExpertModel for ProbabilityCalibratorAdapter {
    fn name(&self) -> &str {
        "probability_calibrator"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Meta
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
            .with_context(|| "probability_calibrator predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`ProbabilityCalibratorAdapter`].
pub struct ProbabilityCalibratorLoader;
impl ExpertLoader for ProbabilityCalibratorLoader {
    fn name(&self) -> &str {
        "probability_calibrator"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = ProbabilityCalibrationExpert::new(DEFAULT_LOAD_CALIBRATION_METHOD);
        inner.load(artifact_dir).with_context(|| {
            format!(
                "ProbabilityCalibrationExpert::load({}) failed",
                artifact_dir.display()
            )
        })?;
        Ok(Box::new(ProbabilityCalibratorAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// conformal_gate
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`ConformalPredictionExpert`].
///
/// SAFETY: inherits C-FFI Send/Sync contract via internal MetaBlender.
pub struct ConformalGateAdapter {
    inner: ConformalPredictionExpert,
}

unsafe impl Send for ConformalGateAdapter {}
unsafe impl Sync for ConformalGateAdapter {}

impl ConformalGateAdapter {
    pub fn new(inner: ConformalPredictionExpert) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &ConformalPredictionExpert {
        &self.inner
    }
}

impl ExpertModel for ConformalGateAdapter {
    fn name(&self) -> &str {
        "conformal_gate"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Meta
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
            .with_context(|| "conformal_gate predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`ConformalGateAdapter`].
pub struct ConformalGateLoader;
impl ExpertLoader for ConformalGateLoader {
    fn name(&self) -> &str {
        "conformal_gate"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = ConformalPredictionExpert::new(
            DEFAULT_LOAD_CALIBRATION_METHOD,
            DEFAULT_LOAD_CONFORMAL_ALPHA,
        );
        inner.load(artifact_dir).with_context(|| {
            format!(
                "ConformalPredictionExpert::load({}) failed",
                artifact_dir.display()
            )
        })?;
        Ok(Box::new(ConformalGateAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// meta_stack
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`MetaDecisionStack`].
///
/// SAFETY: inherits C-FFI Send/Sync contract via internal MetaBlender.
pub struct MetaStackAdapter {
    inner: MetaDecisionStack,
}

unsafe impl Send for MetaStackAdapter {}
unsafe impl Sync for MetaStackAdapter {}

impl MetaStackAdapter {
    pub fn new(inner: MetaDecisionStack) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &MetaDecisionStack {
        &self.inner
    }
}

impl ExpertModel for MetaStackAdapter {
    fn name(&self) -> &str {
        "meta_stack"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Meta
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
            .with_context(|| "meta_stack predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`MetaStackAdapter`].
pub struct MetaStackLoader;
impl ExpertLoader for MetaStackLoader {
    fn name(&self) -> &str {
        "meta_stack"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = MetaDecisionStack::new(
            DEFAULT_LOAD_CALIBRATION_METHOD,
            DEFAULT_LOAD_CONFORMAL_ALPHA,
        );
        inner.load(artifact_dir).with_context(|| {
            format!("MetaDecisionStack::load({}) failed", artifact_dir.display())
        })?;
        Ok(Box::new(MetaStackAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// Convenience: register-all-meta-loaders
// ---------------------------------------------------------------------------

/// Register every meta-family loader (7 canonical names).
pub fn register_meta_loaders(registry: &mut super::ExpertRegistry) -> Result<()> {
    registry.register(Box::new(ElasticNetLoader))?;
    registry.register(Box::new(LogisticLoader))?;
    registry.register(Box::new(BayesLogitLoader))?;
    registry.register(Box::new(MetaBlenderLoader))?;
    registry.register(Box::new(ProbabilityCalibratorLoader))?;
    registry.register(Box::new(ConformalGateLoader))?;
    registry.register(Box::new(MetaStackLoader))?;
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
    fn all_meta_adapters_report_correct_name_family_kind() {
        let cases: Vec<(Box<dyn ExpertModel>, &str)> = vec![
            (
                Box::new(ElasticNetAdapter::new(ElasticNetExpert::new(0.01, 0.5))),
                "elasticnet",
            ),
            (
                Box::new(LogisticAdapter::new(LogisticExpert::new())),
                "logistic",
            ),
            (
                Box::new(BayesLogitAdapter::new(BayesianLogitExpert::new())),
                "bayes_logit",
            ),
            (
                Box::new(MetaBlenderAdapter::new(MetaBlender::new())),
                "meta_blender",
            ),
            (
                Box::new(ProbabilityCalibratorAdapter::new(
                    ProbabilityCalibrationExpert::new(DEFAULT_LOAD_CALIBRATION_METHOD),
                )),
                "probability_calibrator",
            ),
            (
                Box::new(ConformalGateAdapter::new(ConformalPredictionExpert::new(
                    DEFAULT_LOAD_CALIBRATION_METHOD,
                    DEFAULT_LOAD_CONFORMAL_ALPHA,
                ))),
                "conformal_gate",
            ),
            (
                Box::new(MetaStackAdapter::new(MetaDecisionStack::new(
                    DEFAULT_LOAD_CALIBRATION_METHOD,
                    DEFAULT_LOAD_CONFORMAL_ALPHA,
                ))),
                "meta_stack",
            ),
        ];
        for (adapter, name) in cases {
            assert_eq!(adapter.name(), name);
            assert_eq!(adapter.family(), ModelFamily::Meta);
            assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
        }
    }

    #[test]
    fn loader_names_match_their_adapter_names() {
        assert_eq!(ElasticNetLoader.name(), "elasticnet");
        assert_eq!(LogisticLoader.name(), "logistic");
        assert_eq!(BayesLogitLoader.name(), "bayes_logit");
        assert_eq!(MetaBlenderLoader.name(), "meta_blender");
        assert_eq!(ProbabilityCalibratorLoader.name(), "probability_calibrator");
        assert_eq!(ConformalGateLoader.name(), "conformal_gate");
        assert_eq!(MetaStackLoader.name(), "meta_stack");
    }

    #[test]
    fn register_meta_loaders_installs_seven_names() {
        let mut reg = ExpertRegistry::new();
        register_meta_loaders(&mut reg).expect("register");
        let mut names = reg.registered_names();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "bayes_logit",
                "conformal_gate",
                "elasticnet",
                "logistic",
                "meta_blender",
                "meta_stack",
                "probability_calibrator",
            ]
        );
    }

    #[test]
    fn register_meta_loaders_rejects_double_registration() {
        let mut reg = ExpertRegistry::new();
        register_meta_loaders(&mut reg).expect("first call");
        assert!(register_meta_loaders(&mut reg).is_err());
    }

    #[test]
    fn full_24_tree_deep_meta_loaders_coexist() {
        let mut reg = ExpertRegistry::new();
        super::super::tree_adapters::register_tree_loaders(&mut reg).expect("trees");
        super::super::deep_classification_adapters::register_deep_classification_loaders(&mut reg)
            .expect("deep-cls");
        super::super::deep_timeseries_adapters::register_deep_timeseries_loaders(&mut reg)
            .expect("deep-ts");
        register_meta_loaders(&mut reg).expect("meta");
        // 7 tree + 3 deep-cls + 7 deep-ts + 7 meta = 24
        assert_eq!(reg.registered_names().len(), 24);
    }
}
