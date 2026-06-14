//! [`super::ExpertModel`] adapters for the 2 non-standard families
//! that don't fit the predict_proba(&DataFrame) -> Array2<f32>
//! mould.
//!
//! Phase D1.2.7. Covers:
//! - **dqn** — [`crate::rl::TradingReinforcementLearner`]
//!   3-action Q-network. The native API is per-STATE
//!   (`predict_q_values(&[f32]) -> Vec<f32>`), not per-DataFrame.
//!   The adapter row-iterates the input DataFrame, extracts each
//!   row as a feature vector, predicts Q-values, and softmaxes
//!   into a [`super::ExpertOutputKind::Classification3`]
//!   `[p_neutral, p_buy, p_sell]` distribution (canonical order;
//!   see `base.rs` lines 128-135) that the aggregator can vote on.
//! - **exit_agent** — [`crate::exit_agent::ExitAgent`]
//!   exit-side decision network. Returns
//!   `RuntimePrediction` with `class_probabilities: [hold,
//!   neutral, close]`. Tagged with
//!   [`super::ExpertOutputKind::ExitDecision3`] so the
//!   trade-direction aggregators (SoftVoting, MoE classifier head)
//!   SKIP it — the exit-side pipeline (close existing positions
//!   on signal) is its own future consumer.
//!
//! ## Deferred to D1.2.8: SwarmForecaster
//!
//! [`crate::forecasting::SwarmForecaster`] has a fundamentally
//! different lifecycle: `fit_series(values, timestamps)` then
//! `forecast(&mut self, horizon) -> SwarmForecastResult` —
//! univariate-time-series-in, continuous-vector-out, requires
//! mutable state, doesn't take a DataFrame. To plug into the
//! ExpertModel trait we'd need either (a) a separate trait
//! extension for stateful univariate forecasters, or (b) a
//! Mutex-wrapped adapter that extracts a "close" column from the
//! input DataFrame and re-fits on each predict — both decisions
//! that warrant their own focused commit (D1.2.8). The producer
//! reports `swarm_forecaster` as `missing` in the load outcome
//! until that work lands; the SoftVoting layer is unaffected
//! because Forecast1 outputs are already skipped.
//!
//! ## Load contract
//!
//! Unlike the other adapter families, both `dqn` and `exit_agent`
//! use a STATIC `load` method: `pub fn load(path) -> Result<Self>`
//! (not `&mut self -> Result<()>`). The loaders wrap that pattern
//! directly without the `new(...)` + `load(&mut)` two-step the
//! tree/meta adapters use.

use std::path::Path;

use anyhow::{Context, Result};
use polars::prelude::DataFrame;

use super::{ExpertLoader, ExpertModel, ExpertOutputKind, ExpertPrediction};
use crate::exit_agent::ExitAgent;
use crate::rl::TradingReinforcementLearner;
use crate::runtime::capabilities::ModelFamily;
use crate::soft_actor_critic::SoftActorCritic;

// ---------------------------------------------------------------------------
// dqn
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`TradingReinforcementLearner`] (dqn).
///
/// Calls `predict_q_values` once per DataFrame row, then softmaxes
/// the 3 Q-values into a Classification3 distribution so the
/// aggregator's vote layer treats the RL agent's preferred action
/// like any other classifier's argmax.
pub struct DqnAdapter {
    inner: TradingReinforcementLearner,
}

// SAFETY: same contract as the deep-classifier adapters
// (D1.2.2 module doc): Burn's OnceCell tensor cache is
// initialized on load and read atomically thereafter. The
// adapter's `predict` is the only entry point the producer ever
// reaches, and it doesn't mutate the inner agent.
unsafe impl Sync for DqnAdapter {}

impl DqnAdapter {
    pub fn new(inner: TradingReinforcementLearner) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &TradingReinforcementLearner {
        &self.inner
    }
}

impl ExpertModel for DqnAdapter {
    fn name(&self) -> &str {
        "dqn"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Rl
    }
    fn output_kind(&self) -> ExpertOutputKind {
        // We softmax the Q-values internally so the aggregator can
        // treat us as a regular Classification3 voter.
        ExpertOutputKind::Classification3
    }
    fn feature_columns(&self) -> &[String] {
        self.inner.feature_columns()
    }
    fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
        let n_rows = df.height();
        if n_rows == 0 {
            return Ok(Vec::new());
        }
        // Build a row-major (n_rows × n_cols) view by collecting
        // each numeric column into f32. polars columns expose
        // `f64()` for cast-to-f64; we down-cast to f32 at the
        // boundary (matches the DQN trainer's input contract).
        let n_cols = df.width();
        let columns = df.get_columns();
        // Per-row state vector — collected once per row to keep
        // the Q-network's input shape `(state_dim,)`.
        let mut out = Vec::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            let mut state: Vec<f32> = Vec::with_capacity(n_cols);
            for col in columns {
                // Try f64() then i64() then f32() — the feature
                // builder typically emits f32 / f64; integer
                // columns (timestamps etc.) get coerced.
                let value = if let Ok(series) = col.f64() {
                    series.get(row_idx).unwrap_or(0.0) as f32
                } else if let Ok(series) = col.i64() {
                    series.get(row_idx).unwrap_or(0) as f32
                } else if let Ok(series) = col.f32() {
                    series.get(row_idx).unwrap_or(0.0)
                } else {
                    // Unknown dtype — substitute 0.0 so the
                    // predict_q_values call doesn't panic on
                    // length mismatch.
                    0.0
                };
                state.push(if value.is_finite() { value } else { 0.0 });
            }
            let q_values = self
                .inner
                .predict_q_values(&state)
                .with_context(|| format!("dqn predict_q_values failed at row {row_idx}"))?;
            if q_values.len() != 3 {
                anyhow::bail!(
                    "dqn predict_q_values returned {} values, expected 3 (sell/hold/buy)",
                    q_values.len()
                );
            }
            // Softmax → 3-class probability distribution.
            let max_q = q_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_q: Vec<f32> = q_values.iter().map(|q| (q - max_q).exp()).collect();
            let sum: f32 = exp_q.iter().sum();
            let probs: Vec<f32> = if sum > 0.0 {
                exp_q.iter().map(|e| e / sum).collect()
            } else {
                vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            };
            let pred = ExpertPrediction {
                kind: ExpertOutputKind::Classification3,
                values: probs,
            };
            pred.validate()?;
            out.push(pred);
        }
        Ok(out)
    }
}

/// Loader for [`DqnAdapter`].
pub struct DqnLoader;

impl ExpertLoader for DqnLoader {
    fn name(&self) -> &str {
        "dqn"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let inner = TradingReinforcementLearner::load(artifact_dir).with_context(|| {
            format!(
                "TradingReinforcementLearner::load({}) failed",
                artifact_dir.display()
            )
        })?;
        Ok(Box::new(DqnAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// exit_agent
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`ExitAgent`].
///
/// Exposes the agent's `[hold, neutral, close]` exit-decision
/// distribution tagged with
/// [`ExpertOutputKind::ExitDecision3`]. The trade-direction
/// aggregator (SoftVoting / MoE classifier head) silently skips
/// ExitDecision3 outputs; the exit-side pipeline (which closes
/// open positions on signal) is the consumer.
pub struct ExitAgentAdapter {
    inner: ExitAgent,
}

// SAFETY: see DqnAdapter — same Burn OnceCell contract.
unsafe impl Sync for ExitAgentAdapter {}

impl ExitAgentAdapter {
    pub fn new(inner: ExitAgent) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &ExitAgent {
        &self.inner
    }
}

impl ExpertModel for ExitAgentAdapter {
    fn name(&self) -> &str {
        "exit_agent"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Exit
    }
    fn output_kind(&self) -> ExpertOutputKind {
        ExpertOutputKind::ExitDecision3
    }
    fn feature_columns(&self) -> &[String] {
        self.inner.feature_columns()
    }
    fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
        let runtime_preds = self
            .inner
            .predict_runtime(df)
            .with_context(|| "exit_agent predict_runtime failed")?;
        let mut out = Vec::with_capacity(runtime_preds.len());
        for rp in runtime_preds {
            let probs = rp.class_probabilities();
            let values: Vec<f32> = probs.to_vec();
            let pred = ExpertPrediction {
                kind: ExpertOutputKind::ExitDecision3,
                values,
            };
            pred.validate()?;
            out.push(pred);
        }
        Ok(out)
    }
}

/// Loader for [`ExitAgentAdapter`].
pub struct ExitAgentLoader;

impl ExpertLoader for ExitAgentLoader {
    fn name(&self) -> &str {
        "exit_agent"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let inner = ExitAgent::load(artifact_dir)
            .with_context(|| format!("ExitAgent::load({}) failed", artifact_dir.display()))?;
        Ok(Box::new(ExitAgentAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// sac (Soft Actor-Critic, discrete) — ENTRY voter (like dqn)
// ---------------------------------------------------------------------------

/// [`ExpertModel`] adapter for [`SoftActorCritic`] (sac).
///
/// SAC is an **entry / direction** policy. Its softmax policy emits a
/// canonical 3-class `[neutral, buy, sell]` distribution tagged
/// [`super::ExpertOutputKind::Classification3`] so the trade-direction
/// aggregators (SoftVoting / MoE classifier head) treat it as a regular
/// soft voter — exactly like [`DqnAdapter`], and unlike the exit agent
/// whose `ExitDecision3` outputs are filtered out.
pub struct SacAgentAdapter {
    inner: SoftActorCritic,
}

// SAFETY: see DqnAdapter — same Burn OnceCell contract. `predict` is the
// only entry point and does not mutate the inner agent.
unsafe impl Sync for SacAgentAdapter {}

impl SacAgentAdapter {
    pub fn new(inner: SoftActorCritic) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &SoftActorCritic {
        &self.inner
    }
}

impl ExpertModel for SacAgentAdapter {
    fn name(&self) -> &str {
        "sac"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Rl
    }
    fn output_kind(&self) -> ExpertOutputKind {
        // SAC's softmax policy IS a Classification3 distribution — it
        // votes on trade direction like the DQN entry agent.
        ExpertOutputKind::Classification3
    }
    fn feature_columns(&self) -> &[String] {
        self.inner.feature_columns()
    }
    fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
        let runtime_preds = self
            .inner
            .predict_runtime(df)
            .with_context(|| "sac predict_runtime failed")?;
        let mut out = Vec::with_capacity(runtime_preds.len());
        for rp in runtime_preds {
            let pred = ExpertPrediction {
                kind: ExpertOutputKind::Classification3,
                values: rp.class_probabilities().to_vec(),
            };
            pred.validate()?;
            out.push(pred);
        }
        Ok(out)
    }
}

/// Loader for [`SacAgentAdapter`].
pub struct SacAgentLoader;

impl ExpertLoader for SacAgentLoader {
    fn name(&self) -> &str {
        "sac"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let inner = SoftActorCritic::load(artifact_dir)
            .with_context(|| format!("SoftActorCritic::load({}) failed", artifact_dir.display()))?;
        Ok(Box::new(SacAgentAdapter::new(inner)))
    }
}

// ---------------------------------------------------------------------------
// Convenience: register-all-rl-exit-loaders
// ---------------------------------------------------------------------------

/// Register the RL entry-voter loaders: `dqn` + `sac` (2 names).
///
/// **F-318 (2026-05-29)**: the exit-side `exit_agent` is NOT wired in —
/// its `ExitDecision3` outputs are filtered out by `SoftVotingEnsemble`
/// (Classification3 only) and no production exit-side pipeline reads
/// them. `ExitAgentLoader` is kept in the module for future revival.
///
/// **SAC (Christodoulou 2019)**: a real discrete Soft Actor-Critic entry
/// policy. Unlike the exit agent it emits a `Classification3`
/// `[neutral, buy, sell]` distribution and soft-votes like `dqn`, so it
/// IS registered here.
pub fn register_rl_exit_loaders(registry: &mut super::ExpertRegistry) -> Result<()> {
    registry.register(Box::new(DqnLoader))?;
    registry.register(Box::new(SacAgentLoader))?;
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
    fn dqn_loader_name() {
        assert_eq!(DqnLoader.name(), "dqn");
    }

    #[test]
    fn exit_agent_loader_name() {
        assert_eq!(ExitAgentLoader.name(), "exit_agent");
    }

    #[test]
    fn register_rl_exit_loaders_installs_two_names() {
        let mut reg = ExpertRegistry::new();
        register_rl_exit_loaders(&mut reg).expect("register");
        let mut names = reg.registered_names();
        names.sort_unstable();
        // dqn + sac entry voters (exit_agent stays unwired — F-318).
        assert_eq!(names, vec!["dqn", "sac"]);
    }

    #[test]
    fn register_rl_exit_loaders_rejects_double_registration() {
        let mut reg = ExpertRegistry::new();
        register_rl_exit_loaders(&mut reg).expect("first call");
        assert!(register_rl_exit_loaders(&mut reg).is_err());
    }

    #[test]
    fn output_kinds_are_distinct() {
        // DQN softmaxes to Classification3 (votes), exit_agent
        // tags ExitDecision3 (doesn't vote). Pin that they are
        // intentionally different so the SoftVoting aggregator
        // skip-logic stays correct.
        let inner_dqn = TradingReinforcementLearner::new();
        let dqn = DqnAdapter::new(inner_dqn);
        assert_eq!(dqn.output_kind(), ExpertOutputKind::Classification3);

        let inner_exit = ExitAgent::new(8);
        let exit = ExitAgentAdapter::new(inner_exit);
        assert_eq!(exit.output_kind(), ExpertOutputKind::ExitDecision3);
        assert_eq!(exit.family(), ModelFamily::Exit);

        // SAC is an entry voter: Classification3 (like dqn), NOT filtered.
        let inner_sac = SoftActorCritic::new(8);
        let sac = SacAgentAdapter::new(inner_sac);
        assert_eq!(sac.output_kind(), ExpertOutputKind::Classification3);
        assert_eq!(sac.family(), ModelFamily::Rl);
    }

    #[test]
    fn full_30_loaders_coexist() {
        // **F-319 (2026-05-29)**: genetic/neuro_evo/neat removed — they
        // are strategy discoverers in `neoethos-search`, not voters.
        // **F-318 (2026-05-29)**: exit_agent removed — `ExitDecision3`
        // outputs are filtered out by SoftVotingEnsemble and no
        // production exit-side pipeline consumes them.
        let mut reg = ExpertRegistry::new();
        super::super::tree_adapters::register_tree_loaders(&mut reg).expect("trees");
        super::super::deep_classification_adapters::register_deep_classification_loaders(&mut reg)
            .expect("deep-cls");
        super::super::deep_timeseries_adapters::register_deep_timeseries_loaders(&mut reg)
            .expect("deep-ts");
        super::super::meta_adapters::register_meta_loaders(&mut reg).expect("meta");
        super::super::mixed_adapters::register_mixed_loaders(&mut reg).expect("mixed");
        register_rl_exit_loaders(&mut reg).expect("rl");
        // 7 tree + 3 deep-cls + 7 deep-ts + 8 meta (incl. hmm_regime) +
        // 3 mixed + 2 rl (dqn + sac) = 30
        // (swarm_forecaster deferred to D1.2.8;
        //  3 evolutionary removed in F-319; exit_agent removed in F-318)
        assert_eq!(reg.registered_names().len(), 30);
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
            "nbeats",
            "nbeatsx_nf",
            "tide",
            "tide_nf",
            "transformer",
            "patchtst",
            "timesnet",
            "elasticnet",
            "logistic",
            "bayes_logit",
            "meta_blender",
            "probability_calibrator",
            "conformal_gate",
            "meta_stack",
            "hmm_regime",
            "online_pa",
            "online_hoeffding",
            "isolation_forest",
            "dqn",
            "sac",
        ] {
            assert!(
                reg.has_loader(required),
                "registry missing loader for '{required}'"
            );
        }
        for absent in ["genetic", "neuro_evo", "neat", "exit_agent"] {
            assert!(
                !reg.has_loader(absent),
                "registry should not have loader for '{absent}' (F-318/F-319)"
            );
        }
        assert!(
            !reg.has_loader("swarm_forecaster"),
            "swarm_forecaster is intentionally deferred to D1.2.8"
        );
    }
}
