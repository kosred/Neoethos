//! [`super::ExpertModel`] adapters for the streaming-adaptive +
//! anomaly families (3 canonical names).
//!
//! Phase D1.2.5. Covers:
//! - **online_pa** — [`OnlinePassiveAggressiveExpert`] (Adaptive)
//! - **online_hoeffding** — [`OnlineHoeffdingExpert`] (Adaptive)
//! - **isolation_forest** — [`IsolationForestExpert`] (Anomaly)
//!
//! All three emit Classification3 (`(n_rows, 3)` softmax). For
//! `isolation_forest` the 3-class shape is a deliberate adaptation:
//! the underlying anomaly score is mapped through
//! `canonical_three_class_label_mapping` so the ensemble can treat
//! it uniformly with the other classifiers — the meta gate (D1.5
//! MoE) is the right layer to ALSO consume the raw anomaly score
//! as a side-channel feature, which will require an additional
//! `AnomalyScore`-output adapter shipping in a focused follow-up
//! commit if the operator wants it.
//!
//! ## Send + Sync
//!
//! All three are pure Rust:
//! - online_pa: f32 weight vector + SGD updates, no FFI.
//! - online_hoeffding: pure-Rust `irithyll::SGBT` SGBT backend.
//! - isolation_forest: Rust `extended_isolation_forest` (or
//!   internal diagonal-profile fallback). No FFI / OnceCell.
//!
//! So `Send + Sync` are auto-derived — no `unsafe impl` needed.
//!
//! ## Deferred: SwarmForecaster
//!
//! The seventh "forecasting" canonical name (`swarm_forecaster`)
//! has a DIFFERENT interface — `forecast(&mut self, horizon: usize)
//! -> Result<SwarmForecastResult>` instead of `predict_proba(&self,
//! &DataFrame) -> Result<Array2<f32>>`. It produces continuous
//! horizon-length forecast vectors with confidence intervals, not
//! 3-class probabilities. Wrapping it in the current `ExpertModel`
//! trait would require either (a) trait extension to support
//! `&mut self` predictors, or (b) a stateless adapter that maps
//! a single-step forecast back into a Classification3 prediction
//! via signed-return-to-class binning. Either path is a focused
//! decision that lands in a separate commit (D1.2.5b) — keeping
//! it out of this commit so the three straightforward adapters
//! ship cleanly today.

use std::path::Path;

use anyhow::{Context, Result};
use polars::prelude::DataFrame;

use super::{ExpertLoader, ExpertModel, ExpertOutputKind, ExpertPrediction};
use super::tree_adapters::classification3_per_row;
use crate::anomaly::IsolationForestExpert;
use crate::base::ExpertModel as BaseExpertModel;
use crate::runtime::capabilities::ModelFamily;
use crate::streaming::{OnlineHoeffdingExpert, OnlinePassiveAggressiveExpert};

// ---------------------------------------------------------------------------
// online_pa
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`OnlinePassiveAggressiveExpert`].
pub struct OnlinePaAdapter {
    inner: OnlinePassiveAggressiveExpert,
}

impl OnlinePaAdapter {
    pub fn new(inner: OnlinePassiveAggressiveExpert) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &OnlinePassiveAggressiveExpert {
        &self.inner
    }
}

impl ExpertModel for OnlinePaAdapter {
    fn name(&self) -> &str {
        "online_pa"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Adaptive
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
            .with_context(|| "online_pa predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`OnlinePaAdapter`].
pub struct OnlinePaLoader;
impl ExpertLoader for OnlinePaLoader {
    fn name(&self) -> &str {
        "online_pa"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        // Defaults are irrelevant — load() rebuilds state from disk.
        let mut inner = OnlinePassiveAggressiveExpert::new(0.5, 25);
        inner.load(artifact_dir).with_context(|| {
            format!(
                "OnlinePassiveAggressiveExpert::load({}) failed",
                artifact_dir.display()
            )
        })?;
        Ok(Box::new(OnlinePaAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// online_hoeffding
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`OnlineHoeffdingExpert`].
pub struct OnlineHoeffdingAdapter {
    inner: OnlineHoeffdingExpert,
}

impl OnlineHoeffdingAdapter {
    pub fn new(inner: OnlineHoeffdingExpert) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &OnlineHoeffdingExpert {
        &self.inner
    }
}

impl ExpertModel for OnlineHoeffdingAdapter {
    fn name(&self) -> &str {
        "online_hoeffding"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Adaptive
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
            .with_context(|| "online_hoeffding predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`OnlineHoeffdingAdapter`].
pub struct OnlineHoeffdingLoader;
impl ExpertLoader for OnlineHoeffdingLoader {
    fn name(&self) -> &str {
        "online_hoeffding"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = OnlineHoeffdingExpert::new(None);
        inner.load(artifact_dir).with_context(|| {
            format!(
                "OnlineHoeffdingExpert::load({}) failed",
                artifact_dir.display()
            )
        })?;
        Ok(Box::new(OnlineHoeffdingAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// isolation_forest
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`IsolationForestExpert`].
///
/// The isolation forest's NATIVE output is a 1-D anomaly score in
/// `[0.0, 1.0]`. The struct's [`crate::base::ExpertModel`]::
/// `predict_proba` adapts that into a Classification3 layout via
/// `canonical_three_class_label_mapping` so the existing ensemble
/// machinery treats it uniformly. We propagate that
/// Classification3 view here.
///
/// If the future MoE gate wants the raw anomaly score as a
/// SIDE-CHANNEL feature (rather than the adapted Classification3),
/// a parallel `IsolationForestAnomalyAdapter` with
/// `ExpertOutputKind::AnomalyScore` ships in a focused follow-up.
pub struct IsolationForestAdapter {
    inner: IsolationForestExpert,
}

impl IsolationForestAdapter {
    pub fn new(inner: IsolationForestExpert) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &IsolationForestExpert {
        &self.inner
    }
}

impl ExpertModel for IsolationForestAdapter {
    fn name(&self) -> &str {
        "isolation_forest"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Anomaly
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
            .with_context(|| "isolation_forest predict_proba failed")?;
        classification3_per_row(&probs)
    }
}

/// Loader for [`IsolationForestAdapter`].
pub struct IsolationForestLoader;
impl ExpertLoader for IsolationForestLoader {
    fn name(&self) -> &str {
        "isolation_forest"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut inner = IsolationForestExpert::new(100, 256);
        inner.load(artifact_dir).with_context(|| {
            format!(
                "IsolationForestExpert::load({}) failed",
                artifact_dir.display()
            )
        })?;
        Ok(Box::new(IsolationForestAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// Convenience: register-all-mixed-loaders
// ---------------------------------------------------------------------------

/// Register every adaptive + anomaly loader (online_pa,
/// online_hoeffding, isolation_forest — 3 canonical names).
pub fn register_mixed_loaders(registry: &mut super::ExpertRegistry) -> Result<()> {
    registry.register(Box::new(OnlinePaLoader))?;
    registry.register(Box::new(OnlineHoeffdingLoader))?;
    registry.register(Box::new(IsolationForestLoader))?;
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
    fn online_pa_adapter_round_trip() {
        let inner = OnlinePassiveAggressiveExpert::new(0.5, 25);
        let adapter = OnlinePaAdapter::new(inner);
        assert_eq!(adapter.name(), "online_pa");
        assert_eq!(adapter.family(), ModelFamily::Adaptive);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn online_hoeffding_adapter_round_trip() {
        let inner = OnlineHoeffdingExpert::new(None);
        let adapter = OnlineHoeffdingAdapter::new(inner);
        assert_eq!(adapter.name(), "online_hoeffding");
        assert_eq!(adapter.family(), ModelFamily::Adaptive);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn isolation_forest_adapter_round_trip() {
        let inner = IsolationForestExpert::new(100, 256);
        let adapter = IsolationForestAdapter::new(inner);
        assert_eq!(adapter.name(), "isolation_forest");
        assert_eq!(adapter.family(), ModelFamily::Anomaly);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn loader_names_match_adapter_names() {
        assert_eq!(OnlinePaLoader.name(), "online_pa");
        assert_eq!(OnlineHoeffdingLoader.name(), "online_hoeffding");
        assert_eq!(IsolationForestLoader.name(), "isolation_forest");
    }

    #[test]
    fn register_mixed_loaders_installs_three_names() {
        let mut reg = ExpertRegistry::new();
        register_mixed_loaders(&mut reg).expect("register");
        let mut names = reg.registered_names();
        names.sort_unstable();
        assert_eq!(names, vec!["isolation_forest", "online_hoeffding", "online_pa"]);
    }

    #[test]
    fn full_27_loaders_coexist() {
        let mut reg = ExpertRegistry::new();
        super::super::tree_adapters::register_tree_loaders(&mut reg).expect("trees");
        super::super::deep_classification_adapters::register_deep_classification_loaders(
            &mut reg,
        )
        .expect("deep-cls");
        super::super::deep_timeseries_adapters::register_deep_timeseries_loaders(&mut reg)
            .expect("deep-ts");
        super::super::meta_adapters::register_meta_loaders(&mut reg).expect("meta");
        register_mixed_loaders(&mut reg).expect("mixed");
        // 7 tree + 3 deep-cls + 7 deep-ts + 7 meta + 3 mixed = 27
        assert_eq!(reg.registered_names().len(), 27);
    }
}
