//! Inference-half foundation for the 34-model ensemble.
//!
//! ## Why this module exists
//!
//! The training side of neoethos-models already trains all 34 expert
//! families ([`crate::runtime::capabilities::KNOWN_MODEL_NAMES`]) in
//! parallel via [`crate::training_orchestrator::TrainingOrchestrator`]
//! and saves each one's artifacts to disk (34th added 2026-05-25:
//! `hmm_regime`, a 3-state HMM regime classifier). But until this
//! module landed, the **inference side did not exist**: no runtime
//! code loaded the trained experts back from disk, no orchestrator
//! ran `predict_proba` on each, and no aggregator combined their
//! outputs. The 34-model "ensemble" was 34 INDEPENDENT models with
//! no consumer.
//!
//! Phase D1.2 (this module) lays the **foundation traits** so the
//! follow-up phases can build progressively:
//!
//! - **D1.2 (this)** — [`ExpertModel`] trait + [`ExpertRegistry`] +
//!   [`EnsemblePredictor`] trait + tests with mock experts.
//! - **D1.2.x** — per-family adapters: each existing expert struct
//!   (LightGBM, XGBoost, MLP, Transformer, …) gets an
//!   [`ExpertModel`] impl that exposes its existing predict
//!   behaviour through the uniform trait.
//! - **D1.3** — `SoftVotingEnsemble` — the aggregator, reached through
//!   [`SoftVotingEnsemble::predict_with_roles`] (weighted average of loaded
//!   experts' 3-class probabilities, role-aware). Useable from day one
//!   against existing trained artifacts.
//!
//!   **2026-08-09 (batch D4):** [`EnsemblePredictor`] is NO LONGER a
//!   prediction trait — its `predict` method (a role-blind flat average with
//!   no production caller) was deleted; what remains is the load-reporting
//!   contract (`load_outcome` + `experts`). Inference is
//!   `predict_with_roles`. If the MoE gating network (D1.6) arrives, give it
//!   `predict_with_roles` — do NOT resurrect a flat average alongside it.
//! - **D1.4** — diversity enforcement during training (random
//!   seeds + regime feature; NOT feature subsets — operator
//!   directive 2026-05-17 rejected the random-subspace approach
//!   because the modern MoE answer creates diversity through
//!   joint training rather than artificial restrictions).
//! - **D1.5** — MoE gating network design + training pipeline.
//! - **D1.6** — `MoeEnsemble` as the production
//!   [`EnsemblePredictor`] (replaces SoftVotingEnsemble when a
//!   trained MoE artifact is available).
//!
//! ## Partial-load contract (operator directive 2026-05-17 option β)
//!
//! [`ExpertRegistry::load_with_partial`] does NOT fail when an
//! expert's artifact is missing or invalid. Instead it returns an
//! [`ExpertLoadOutcome`] that names each of the three categories:
//!
//! - `loaded`: experts that came back from disk healthy.
//! - `missing`: experts whose artifact dir doesn't exist on disk
//!   (training never ran or was aborted).
//! - `degraded`: experts whose artifact dir exists but didn't load
//!   cleanly (corruption, version skew, missing native deps).
//!
//! The [`EnsemblePredictor`] surfaces the load outcome through
//! [`EnsemblePredictor::load_outcome`] so the operator chrome can
//! render "Running ensemble: 25/33 experts active — 8 degraded
//! (see system log)". This is the **tracked degradation** the
//! operator explicitly asked for. (33 = `DEFAULT_BOOTSTRAP_EXPERT_NAMES.len()`
//! — KNOWN_MODEL_NAMES.len() minus the deferred `swarm_forecaster`.)
//!
//! ## Heterogeneous expert outputs
//!
//! Not every expert produces 3-class probabilities. The 33 names
//! include classification heads (tree experts → buy/neutral/sell
//! probs), single-value forecasters (nbeats, tide, transformer →
//! continuous next-bar forecast), anomaly scorers (isolation
//! forest → 1-D outlier score), and RL agents (dqn → 3-action
//! Q-values). The trait normalises on [`ExpertPrediction`] which
//! carries an [`ExpertOutputKind`] tag plus the native values; the
//! aggregating [`EnsemblePredictor`] decides how to combine them:
//! - `SoftVotingEnsemble` (D1.3) only averages
//!   `Classification3` and `ActionValues3` outputs (the others
//!   sit unused for naive voting).
//! - `MoeEnsemble` (D1.6) feeds the heterogeneous outputs to its
//!   gating network as features and combines them learnt-fashion.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::Result;
use polars::prelude::DataFrame;
use serde::{Deserialize, Serialize};

use crate::runtime::capabilities::ModelFamily;

// Per-family adapter submodules. Each one bridges the existing
// concrete expert structs to the uniform `ExpertModel` trait
// defined below. D1.2.x phases add one family per focused commit:
//   .1 tree (this commit)
//   .2 deep classification
//   .3 deep forecasting
//   .4 meta
//   .5 forecasting/adaptive/anomaly
//   .6 evolutionary + exit + RL
pub mod bootstrap;
pub mod deep_classification_adapters;
pub mod deep_timeseries_adapters;
// F-319 REVISED (2026-07-11, operator directive). The 2026-05-29
// cleanup removed the evolutionary adapters as "strategy discoverers,
// not inference experts" — but it only removed the CONSUMER: the
// training orchestrator kept training `neat` + `neuro_evo` through the
// shared expert path (both implement the base ExpertModel contract
// with genuine 3-class heads), so every run burned hours producing
// artifacts nothing ever read. The operator's standing rule is now:
// every model that trains, votes — the search-only exemption applies
// to `genetic` alone (it discovers strategies; it is not a
// classifier). `evolution_adapters` restores the neat/neuro_evo
// voters, loading the SAME artifacts training already produces.
pub mod evolution_adapters;
pub mod meta_adapters;
pub mod mixed_adapters;
pub mod rl_exit_adapters;
pub mod soft_voting;
// D1.2.8 (2026-07-11): the swarm forecaster finally votes — last-row-only
// (live-gate) semantics; see the module doc for the honesty constraints.
pub mod swarm_adapter;
pub mod tree_adapters;

pub use bootstrap::{
    DEFAULT_BOOTSTRAP_EXPERT_NAMES, build_default_registry, build_ensemble_for_symbol,
    load_experts_for_symbol,
};
pub use deep_classification_adapters::{
    KanAdapter, KanLoader, MlpAdapter, MlpLoader, TabNetAdapter, TabNetLoader,
    register_deep_classification_loaders,
};
pub use deep_timeseries_adapters::{
    NBeatsAdapter, NBeatsLoader, NBeatsxNfAdapter, NBeatsxNfLoader, PatchTstAdapter,
    PatchTstLoader, TiDEAdapter, TiDELoader, TiDENfAdapter, TiDENfLoader, TimesNetAdapter,
    TimesNetLoader, TransformerAdapter, TransformerLoader, register_deep_timeseries_loaders,
};
// F-319: `evolutionary_adapters` re-export removed — those adapters
// were the misclassified-as-experts wrappers around GA/NeuroEvo/NEAT
// search algorithms. See the comment block above the module list.
pub use meta_adapters::{
    BayesLogitAdapter, BayesLogitLoader, ConformalGateAdapter, ConformalGateLoader,
    ElasticNetAdapter, ElasticNetLoader, HmmRegimeAdapter, HmmRegimeLoader, LogisticAdapter,
    LogisticLoader, MetaBlenderAdapter, MetaBlenderLoader, MetaStackAdapter, MetaStackLoader,
    ProbabilityCalibratorAdapter, ProbabilityCalibratorLoader, register_meta_loaders,
};
pub use mixed_adapters::{
    IsolationForestAdapter, IsolationForestLoader, OnlineHoeffdingAdapter, OnlineHoeffdingLoader,
    OnlinePaAdapter, OnlinePaLoader, register_mixed_loaders,
};
pub use rl_exit_adapters::{
    DqnAdapter, DqnLoader, ExitAgentAdapter, ExitAgentLoader, SacAgentAdapter, SacAgentLoader,
    register_rl_exit_loaders,
};
pub use soft_voting::{SoftVotingEnsemble, SoftVotingEnsembleConfig};
pub use tree_adapters::{
    CatboostAdapter, CatboostLoader, LightGbmAdapter, LightGbmLoader, SklearsTreeAdapter,
    SklearsTreeLoader, XgboostAdapter, XgboostLoader, register_tree_loaders,
};

// ---------------------------------------------------------------------------
// Expert output taxonomy
// ---------------------------------------------------------------------------

/// What kind of native output an [`ExpertModel`] produces per input
/// row. The [`EnsemblePredictor`] consults this when deciding how
/// to combine an expert's predictions with the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertOutputKind {
    /// 3-class probability vector `[p_neutral, p_buy, p_sell]`,
    /// rows sum to 1.0. Produced by classification-head experts
    /// (LightGBM, XGBoost, MLP-classifier, transformer-classifier,
    /// etc.). This is the most directly aggregable output for a
    /// trading decision. Canonical mapping (see `base.rs` lines
    /// 128-135 + `runtime/artifacts.rs::default_three_class_label_mapping`):
    /// col 0 = neutral (label 0), col 1 = buy (label 1),
    /// col 2 = sell (label -1).
    Classification3,
    /// 3-action Q-values for `[hold, buy, sell]` from an RL agent
    /// (dqn). Not a probability distribution — values are arbitrary
    /// reals; the action with the highest Q is the recommended
    /// action. Matches `dqn_impl.rs::TradingAction::as_index`: Hold=0,
    /// Buy=1, Sell=2 (same axis order as Classification3).
    ///
    /// NOTE: the live `dqn` ADAPTER (`rl_exit_adapters.rs`) does NOT emit this
    /// variant — it softmaxes the Q-values into a `Classification3` vector and
    /// reports `Classification3`, so the soft-voting / role-aware combiners
    /// treat dqn as a (confirm-only) directional voter. This variant remains
    /// for any future expert that wants to expose raw Q-values directly.
    ActionValues3,
    /// Single continuous forecast value — e.g. predicted next-bar
    /// close-to-close return. Produced by time-series forecasters
    /// (nbeats, tide, patchtst, timesnet, transformer-forecaster).
    /// Not directly comparable to classification probs; the MoE
    /// gating consumes this as a feature for its own decision.
    Forecast1,
    /// `[0.0, 1.0]` anomaly score (higher = more anomalous). From
    /// isolation forest. Acts as a regime / outlier indicator
    /// rather than a direct trading signal — high anomaly scores
    /// suggest the other experts may be unreliable.
    AnomalyScore,
    /// 3-class probability vector `[p_hold, p_neutral, p_close]`
    /// from the exit-side decision agent ([`crate::exit_agent::ExitAgent`]).
    /// SHAPE-COMPATIBLE with [`Self::Classification3`] but
    /// SEMANTICALLY DIFFERENT: this is "should I close my open
    /// position?" not "should I open a new long/short?".
    /// Aggregators that vote on trade DIRECTION (SoftVoting, MoE
    /// classifier-head) ignore it; the exit-side pipeline (which
    /// closes existing positions on signal) consumes it directly.
    ExitDecision3,
}

impl fmt::Display for ExpertOutputKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Classification3 => "classification_3",
            Self::ActionValues3 => "action_values_3",
            Self::Forecast1 => "forecast_1",
            Self::AnomalyScore => "anomaly_score",
            Self::ExitDecision3 => "exit_decision_3",
        };
        f.write_str(s)
    }
}

impl ExpertOutputKind {
    /// Expected length of [`ExpertPrediction::values`] for this
    /// output kind.
    pub fn expected_length(&self) -> usize {
        match self {
            Self::Classification3 => 3,
            Self::ActionValues3 => 3,
            Self::Forecast1 => 1,
            Self::AnomalyScore => 1,
            Self::ExitDecision3 => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// v0.5 ML-integration Stage 2 — role-aware ensemble decision.
// ---------------------------------------------------------------------------

/// The role an expert plays in the role-aware combiner
/// ([`super::soft_voting::SoftVotingEnsemble::predict_with_roles`]).
///
/// CORRECTION TO THE ORIGINAL DESIGN (verified in code 2026-06-07): almost
/// every expert reports [`ExpertOutputKind::Classification3`] and therefore
/// ALREADY votes — but several do so with the WRONG semantics, polluting the
/// flat directional average. The fix is to give each expert its correct ROLE,
/// not to "resurrect" a discarded one. `hmm_regime` is a regime classifier
/// (its trend posterior should GATE size, not vote on direction);
/// `isolation_forest` is an anomaly detector (its score should SCALE/VETO size,
/// not drag every bar toward neutral); `dqn` is a confirm-only directional
/// contributor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRole {
    /// Genuine directional classifier — contributes to the direction vote.
    Direction,
    /// Directional, but confirm-only at the blend (can add weight to a side,
    /// never originate/override direction). Functionally joins the direction
    /// pool here; the gene-dominant invariant is enforced downstream (Stage 3).
    DirectionalConfirm,
    /// Regime gate (`hmm_regime`): multiplies the final confidence by a
    /// bounded [0,1] factor; never votes on direction.
    RegimeGate,
    /// Anomaly scale/veto (`isolation_forest`): multiplies the final confidence
    /// by a bounded [0,1] factor (hard veto when extreme); never votes.
    AnomalyScale,
}

/// Canonical expert-name -> role map. Declared in ONE place so a future new
/// expert cannot silently fall into the wrong role — the role-aware combiner
/// FAILS LOUD on an unmapped loaded expert. (Partitioning on the coarse
/// [`ModelFamily`] would be wrong: `hmm_regime` is `ModelFamily::Meta`, the same
/// family as the directional `meta_blender`/`meta_stack`.)
///
/// Returns `None` for an unmapped name (the caller decides whether to bail).
pub fn expert_role(name: &str) -> Option<ExpertRole> {
    // Replica ensemble members (`transformer_01`, `transformer_02`, …,
    // produced when `models.num_transformers > 1`) inherit the canonical
    // model's role — each replica is an independent voter but plays the
    // same role as its base architecture.
    let canonical = match name.strip_prefix("transformer_") {
        Some(suffix) if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) => {
            "transformer"
        }
        _ => name,
    };
    match canonical {
        // Regime gate — trend posterior gates size, does not vote.
        "hmm_regime" => Some(ExpertRole::RegimeGate),
        // Anomaly scale/veto — outlier score scales size, does not vote.
        "isolation_forest" => Some(ExpertRole::AnomalyScale),
        // Confirm-only directional contributors (RL policies). `dqn`
        // softmaxes its Q-values; `sac` emits a softmax policy directly.
        // Both are entry/direction RL voters that confirm — never
        // originate/override — the genes' direction (Stage 3 invariant).
        "dqn" | "sac" => Some(ExpertRole::DirectionalConfirm),
        // Genuine directional classifiers. `neat` + `neuro_evo` joined
        // 2026-07-11 (F-319 revision — see the `evolution_adapters`
        // module doc): both train 3-class heads on the same labels as
        // every other classifier here. `swarm_forecaster` joined the
        // same day (D1.2.8): a modest forecast-lean voter on the LAST
        // row only, neutral abstain elsewhere (see `swarm_adapter`).
        "lightgbm" | "xgboost" | "xgboost_rf" | "xgboost_dart" | "catboost" | "catboost_alt"
        | "sklears_tree" | "mlp" | "kan" | "tabnet" | "nbeats" | "nbeatsx_nf" | "tide"
        | "tide_nf" | "transformer" | "patchtst" | "timesnet" | "elasticnet" | "logistic"
        | "bayes_logit" | "meta_blender" | "probability_calibrator" | "conformal_gate"
        | "meta_stack" | "online_pa" | "online_hoeffding" | "neat" | "neuro_evo"
        | "swarm_forecaster" => Some(ExpertRole::Direction),
        _ => None,
    }
}

/// Per-row output of the role-aware combiner: a direction vote plus two bounded
/// [0,1] confidence factors. The blend (Stage 3) and the OOS re-validation
/// (Stage 4) consume `dir_probs` for the ML agreement on the gene's side and
/// multiply the confidence by `regime_gate * anomaly_scale`. ML can therefore
/// only SHRINK conviction or veto — never flip direction or manufacture a trade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnsembleDecision {
    /// `[p_neutral, p_buy, p_sell]` from the directional voters only
    /// (hmm_regime / isolation_forest removed from this vote).
    pub dir_probs: [f32; 3],
    /// Regime gate g ∈ [0,1] from `hmm_regime` (1.0 when absent — never a
    /// veto-by-accident on missing data).
    pub regime_gate: f32,
    /// Anomaly scale s ∈ [0,1] from `isolation_forest` (1.0 when absent;
    /// 0.0 = hard veto on an extreme anomaly).
    pub anomaly_scale: f32,
}

impl EnsembleDecision {
    /// Neutral decision (no directional lean, no gate, no veto) — used for
    /// warmup/NaN rows where the ensemble abstains.
    pub fn neutral() -> Self {
        Self {
            dir_probs: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            regime_gate: 1.0,
            anomaly_scale: 1.0,
        }
    }
}

/// Pure regime-gate math (testable without a loaded HMM): given the directional
/// vote and the HMM posterior `[p_range, p_buy, p_sell]`, return
/// `g = (p_buy + p_sell) * posterior_mass_on_the_voted_side` ∈ [0,1].
/// A range regime (p_range→1) or a posterior disagreeing with the vote → g→0
/// (shrink/veto); an agreeing trend → g→1.
pub fn regime_gate_from_posterior(dir_probs: [f32; 3], posterior: [f32; 3]) -> f32 {
    // Direction the vote picked: buy (col1) vs sell (col2); neutral lean -> no
    // gate amplification basis, fall back to (1 - p_range).
    let trend_mass = (posterior[1] + posterior[2]).clamp(0.0, 1.0);
    let agreement = if dir_probs[1] >= dir_probs[2] {
        posterior[1] // voted long -> HMM mass on buy
    } else {
        posterior[2] // voted short -> HMM mass on sell
    };
    (trend_mass * agreement).clamp(0.0, 1.0)
}

/// Pure anomaly-scale math: map a raw anomaly score `a` ∈ [0,1] to a confidence
/// multiplier ∈ [0,1]. Below `lo` → 1.0 (no penalty); ramps down linearly to
/// `hi`; at/above `hi` → 0.0 (hard veto).
pub fn anomaly_scale_from_score(a: f32, lo: f32, hi: f32) -> f32 {
    if !(a.is_finite()) {
        return 1.0;
    }
    if hi <= lo {
        return if a >= hi { 0.0 } else { 1.0 };
    }
    let frac = ((a - lo) / (hi - lo)).clamp(0.0, 1.0);
    (1.0 - frac).clamp(0.0, 1.0)
}

/// One prediction (one expert, one input row).
///
/// The `values` field carries the raw native output of the expert;
/// its length must match `kind.expected_length()`. The validator
/// [`ExpertPrediction::validate`] enforces this so a buggy expert
/// can't corrupt the aggregator.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertPrediction {
    /// Native output type — drives the aggregator's combine logic.
    pub kind: ExpertOutputKind,
    /// Raw expert output, length determined by [`Self::kind`].
    /// For `Classification3` the values are probabilities in
    /// `[0, 1]` summing to ~1.0 (the validator tolerates a small
    /// rounding slack). For `ActionValues3` arbitrary reals. For
    /// `Forecast1` arbitrary reals. For `AnomalyScore` `[0.0, 1.0]`.
    pub values: Vec<f32>,
}

impl ExpertPrediction {
    /// Sanity-check that the values length and ranges match
    /// `kind`. Aggregators MUST call this before combining;
    /// `MockExpert` does call it in tests so a future trait impl
    /// that violates the contract is caught at unit-test time.
    pub fn validate(&self) -> Result<()> {
        let expected = self.kind.expected_length();
        if self.values.len() != expected {
            anyhow::bail!(
                "ExpertPrediction shape mismatch: kind {:?} expects {} values, got {}",
                self.kind,
                expected,
                self.values.len()
            );
        }
        for v in &self.values {
            if !v.is_finite() {
                anyhow::bail!(
                    "ExpertPrediction contains non-finite value (NaN/Inf) for kind {:?}",
                    self.kind
                );
            }
        }
        match self.kind {
            ExpertOutputKind::Classification3 | ExpertOutputKind::ExitDecision3 => {
                // Same shape contract as Classification3: 3 values
                // in [0, 1] summing to ~1.0. ExitDecision3 just
                // carries different semantics ([p_hold, p_neutral,
                // p_close]) but the same shape/range invariants.
                for v in &self.values {
                    if *v < -1e-4 || *v > 1.0 + 1e-4 {
                        anyhow::bail!("{:?} probability out of [0, 1]: {}", self.kind, v);
                    }
                }
                let sum: f32 = self.values.iter().sum();
                if (sum - 1.0).abs() > 1e-2 {
                    anyhow::bail!(
                        "{:?} probabilities do not sum to 1.0: sum = {}",
                        self.kind,
                        sum
                    );
                }
            }
            ExpertOutputKind::AnomalyScore => {
                let v = self.values[0];
                if !(-1e-4..=1.0 + 1e-4).contains(&v) {
                    anyhow::bail!("AnomalyScore out of [0, 1]: {}", v);
                }
            }
            // ActionValues3, Forecast1 — no range constraints.
            _ => {}
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Expert trait
// ---------------------------------------------------------------------------

/// Uniform inference contract for every trained expert.
///
/// All 34 expert families in
/// [`crate::runtime::capabilities::KNOWN_MODEL_NAMES`] implement
/// this trait via a thin adapter (D1.2.x follow-up commits +
/// HMM Phase 2 added the 34th 2026-05-25). The aggregating
/// [`EnsemblePredictor`] holds a `Vec<Box<dyn ExpertModel>>` and
/// treats each one uniformly.
///
/// ## Conventions
///
/// - **`name`** matches an entry in `KNOWN_MODEL_NAMES` exactly
///   (lowercase + underscores). The registry key is the name.
/// - **`predict`** returns one [`ExpertPrediction`] per input row.
///   Implementations must validate via
///   [`ExpertPrediction::validate`] before returning so contract
///   violations surface at unit-test time, not at the broker fill
///   path.
/// - **`feature_columns`** returns the columns the expert was
///   trained on, in the order it expects them in the DataFrame.
///   The registry can use this to detect column-layout drift after
///   a retraining session.
pub trait ExpertModel: Send + Sync {
    /// Canonical expert name — matches `KNOWN_MODEL_NAMES`.
    fn name(&self) -> &str;
    /// Family the expert belongs to (Tree / Deep / Meta / RL / …).
    fn family(&self) -> ModelFamily;
    /// What kind of native output this expert produces.
    fn output_kind(&self) -> ExpertOutputKind;
    /// Column names this expert was trained on, in the order it
    /// expects them in the input DataFrame.
    fn feature_columns(&self) -> &[String];
    /// Run inference. Returns one [`ExpertPrediction`] per row of
    /// `df`. Implementations must validate via
    /// [`ExpertPrediction::validate`] before returning.
    fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>>;
}

// ---------------------------------------------------------------------------
// Registry / loader plumbing
// ---------------------------------------------------------------------------

/// Categorised reason a particular expert failed to load.
///
/// Carried in [`ExpertLoadOutcome::degraded`] so the operator
/// chrome can render specifics ("xgboost: artifact JSON corrupt"
/// rather than just "9 experts failed"). The variants follow the
/// most common failure modes the audit identified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertLoadError {
    /// The expert's artifact directory exists but cannot be read
    /// (permission denied, IO error). Reason carries the underlying
    /// error string.
    Io { name: String, reason: String },
    /// The artifact directory exists but does not contain the
    /// expected files (e.g. `metadata.json` missing).
    InvalidArtifact { name: String, reason: String },
    /// Schema/version skew — the artifact was saved by an older
    /// or newer code revision and the loader refuses to interpret
    /// it. Caller should retrain.
    IncompatibleVersion {
        name: String,
        expected: String,
        found: String,
    },
    /// A required native backend (LightGBM C lib, libtorch CUDA,
    /// etc.) is missing or refused to initialise on this host.
    /// Reason carries the original anyhow chain.
    Backend { name: String, reason: String },
}

impl fmt::Display for ExpertLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { name, reason } => write!(f, "{name}: IO error ({reason})"),
            Self::InvalidArtifact { name, reason } => {
                write!(f, "{name}: invalid artifact ({reason})")
            }
            Self::IncompatibleVersion {
                name,
                expected,
                found,
            } => write!(
                f,
                "{name}: incompatible artifact version (expected {expected}, found {found})"
            ),
            Self::Backend { name, reason } => write!(f, "{name}: backend error ({reason})"),
        }
    }
}

impl std::error::Error for ExpertLoadError {}

impl ExpertLoadError {
    /// The expert name this error refers to. Useful for grouping
    /// errors in the chrome banner.
    pub fn name(&self) -> &str {
        match self {
            Self::Io { name, .. }
            | Self::InvalidArtifact { name, .. }
            | Self::IncompatibleVersion { name, .. }
            | Self::Backend { name, .. } => name,
        }
    }
}

/// One-shot result of [`ExpertRegistry::load_with_partial`].
///
/// Operator directive 2026-05-17 option β: the registry does NOT
/// fail when an expert artifact is missing or degraded; instead it
/// reports every expert's outcome category here. Callers
/// (typically [`EnsemblePredictor`] constructors) decide whether
/// to proceed with the partial set or refuse the start-up.
///
/// Invariants:
/// - The `loaded`, `missing`, and `degraded` lists are disjoint —
///   each requested expert name appears in EXACTLY one of them.
/// - `loaded.iter().map(|e| e.name()).chain(missing.iter()).chain(degraded.iter().map(|e| e.name()))`
///   forms a multiset equal to the original `requested` list.
pub struct ExpertLoadOutcome {
    /// Healthy experts ready for inference, in the order they were
    /// requested.
    pub loaded: Vec<Box<dyn ExpertModel>>,
    /// Experts whose artifact directory was not present on disk.
    /// Typical cause: training never ran for that expert (e.g.
    /// disabled in the operator's config, or the training job was
    /// killed before reaching it).
    pub missing: Vec<String>,
    /// Experts whose artifact directory existed but did not load
    /// cleanly. Each entry names the expert and the categorised
    /// reason.
    pub degraded: Vec<ExpertLoadError>,
}

impl fmt::Debug for ExpertLoadOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExpertLoadOutcome")
            .field(
                "loaded",
                &self.loaded.iter().map(|e| e.name()).collect::<Vec<_>>(),
            )
            .field("missing", &self.missing)
            .field("degraded", &self.degraded)
            .finish()
    }
}

impl ExpertLoadOutcome {
    /// Number of healthy experts.
    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }
    /// Number of experts whose artifact dir was absent.
    pub fn missing_count(&self) -> usize {
        self.missing.len()
    }
    /// Number of experts whose artifact failed to load.
    pub fn degraded_count(&self) -> usize {
        self.degraded.len()
    }
    /// Total number of requested experts (loaded + missing + degraded).
    pub fn requested_count(&self) -> usize {
        self.loaded.len() + self.missing.len() + self.degraded.len()
    }
    /// Names of healthy experts. Useful for the chrome "active
    /// experts" banner.
    pub fn loaded_names(&self) -> Vec<&str> {
        self.loaded.iter().map(|e| e.name()).collect()
    }
    /// `true` when at least one expert loaded successfully — the
    /// ensemble has SOMETHING to predict with. `false` when every
    /// requested expert was missing/degraded; the ensemble cannot
    /// emit signals and the auto-trade producer must refuse to
    /// start.
    pub fn has_any_loaded(&self) -> bool {
        !self.loaded.is_empty()
    }
    /// Build an empty outcome — used by tests and by error paths
    /// that need to surface a "nothing loaded" state without
    /// constructing fake experts.
    pub fn empty() -> Self {
        Self {
            loaded: Vec::new(),
            missing: Vec::new(),
            degraded: Vec::new(),
        }
    }
}

/// Per-family loader trait. The registry holds one of these per
/// expert name and delegates to it during partial-load.
///
/// Each family's D1.2.x follow-up commit implements this trait for
/// its struct(s). E.g. `LightGbmLoader::load("models/EURUSD/H1/lightgbm")
/// -> Result<Box<dyn ExpertModel>>` opens the on-disk artifact and
/// returns a ready-to-predict expert.
pub trait ExpertLoader: Send + Sync {
    /// Canonical expert name this loader produces. Must match
    /// [`ExpertModel::name`] of the loaded result and an entry in
    /// `KNOWN_MODEL_NAMES`.
    fn name(&self) -> &str;
    /// Load the expert's artifact from `artifact_dir` (typically
    /// `<models_root>/<symbol>/<tf>/<name>/`). The implementation
    /// owns the disk-layout convention for its expert family.
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>>;
}

/// Central registry of [`ExpertLoader`]s. The neoethos-app bootstrap
/// builds one of these by registering one loader per family and
/// then calls [`Self::load_with_partial`] to load every requested
/// expert in one shot.
///
/// Lookups are by canonical expert name (matching
/// `KNOWN_MODEL_NAMES`). Duplicate registration of the same name
/// is rejected so a typo can't silently shadow an existing loader.
pub struct ExpertRegistry {
    loaders: HashMap<String, Box<dyn ExpertLoader>>,
}

impl ExpertRegistry {
    /// Build an empty registry. Caller fills it via [`Self::register`].
    pub fn new() -> Self {
        Self {
            loaders: HashMap::new(),
        }
    }

    /// Register a loader. Returns `Err` if a loader with the same
    /// canonical name was already registered (a typo / shadowing
    /// guard).
    pub fn register(&mut self, loader: Box<dyn ExpertLoader>) -> Result<()> {
        let name = loader.name().to_string();
        if self.loaders.contains_key(&name) {
            anyhow::bail!("expert loader '{name}' already registered");
        }
        self.loaders.insert(name, loader);
        Ok(())
    }

    /// `true` if a loader for `name` is registered.
    pub fn has_loader(&self, name: &str) -> bool {
        self.loaders.contains_key(name)
    }

    /// Canonical names of every registered loader, sorted for
    /// determinism.
    pub fn registered_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.loaders.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Load every expert in `requested` from `root`, in order.
    ///
    /// Partial-load semantics per operator directive 2026-05-17
    /// option β:
    /// - If the loader for a requested name is NOT registered →
    ///   counted as `degraded` with [`ExpertLoadError::InvalidArtifact`]
    ///   (no code path for it).
    /// - If the artifact directory doesn't exist on disk →
    ///   `missing`.
    /// - If the loader returns `Err` → `degraded` with the
    ///   categorised error.
    /// - Otherwise → `loaded`.
    pub fn load_with_partial(&self, root: &Path, requested: &[&str]) -> ExpertLoadOutcome {
        let mut outcome = ExpertLoadOutcome::empty();
        for name in requested {
            let loader = match self.loaders.get(*name) {
                Some(l) => l,
                None => {
                    outcome.degraded.push(ExpertLoadError::InvalidArtifact {
                        name: (*name).to_string(),
                        reason: "no loader registered for this expert name".to_string(),
                    });
                    continue;
                }
            };
            let artifact_dir: PathBuf = root.join(name);
            if !artifact_dir.exists() {
                outcome.missing.push((*name).to_string());
                continue;
            }
            match loader.load(&artifact_dir) {
                Ok(expert) => {
                    // Defensive: the loader must return an expert
                    // whose name matches the registry key — a typo
                    // here would silently confuse the aggregator.
                    if expert.name() != *name {
                        outcome.degraded.push(ExpertLoadError::InvalidArtifact {
                            name: (*name).to_string(),
                            reason: format!(
                                "loader returned expert with name '{}' but registry key is '{}'",
                                expert.name(),
                                name
                            ),
                        });
                        continue;
                    }
                    outcome.loaded.push(expert);
                }
                Err(err) => {
                    // Categorise the error string heuristically.
                    // Loaders that want a precise variant should
                    // return ExpertLoadError directly via anyhow
                    // chains; the heuristic here is the fallback.
                    let lower = err.to_string().to_ascii_lowercase();
                    let categorised = if lower.contains("version") {
                        ExpertLoadError::IncompatibleVersion {
                            name: (*name).to_string(),
                            expected: "unknown".to_string(),
                            found: err.to_string(),
                        }
                    } else if lower.contains("backend")
                        || lower.contains("cuda")
                        || lower.contains("native")
                    {
                        ExpertLoadError::Backend {
                            name: (*name).to_string(),
                            reason: err.to_string(),
                        }
                    } else if lower.contains("permission")
                        || lower.contains("io error")
                        || lower.contains("not found")
                    {
                        ExpertLoadError::Io {
                            name: (*name).to_string(),
                            reason: err.to_string(),
                        }
                    } else {
                        ExpertLoadError::InvalidArtifact {
                            name: (*name).to_string(),
                            reason: err.to_string(),
                        }
                    };
                    outcome.degraded.push(categorised);
                }
            }
        }
        outcome
    }

    /// [`Self::load_with_partial`] plus two fixes for "trained but
    /// invisible" artifacts (operator report 2026-07-11):
    ///
    /// 1. **Replica resolution.** Training with
    ///    `models.num_transformers > 1` writes `transformer_01/`,
    ///    `transformer_02/`, … and never a plain `transformer/` dir —
    ///    so the exact-name lookup above counted the transformer as
    ///    `missing` and NO transformer ever voted, silently wasting
    ///    every replica's training time. When a requested name's dir is
    ///    missing, this scans for `{name}_NN` dirs and loads EACH one
    ///    through the canonical loader, renamed to its replica dir name
    ///    so each is an independent voter (`expert_role` maps replica
    ///    names to the canonical role).
    ///
    /// 2. **Orphan detection.** Any artifact directory on disk that no
    ///    loader claimed is reported loudly instead of silently ignored
    ///    — a systematic name mismatch (the transformer bug) or a
    ///    consumer-less model must be visible, not lost in the
    ///    "partial success" tolerance.
    pub fn load_with_partial_replica_aware(
        &self,
        root: &Path,
        requested: &[&str],
    ) -> ExpertLoadOutcome {
        let mut outcome = self.load_with_partial(root, requested);

        // ── 1. Replica resolution for the `missing` names ──────────────
        let mut still_missing: Vec<String> = Vec::new();
        for name in outcome.missing.drain(..) {
            let loader = match self.loaders.get(&name) {
                Some(l) => l,
                None => {
                    still_missing.push(name);
                    continue;
                }
            };
            let mut replica_dirs: Vec<PathBuf> = std::fs::read_dir(root)
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.is_dir()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .and_then(|n| n.strip_prefix(&format!("{name}_")))
                            .is_some_and(|suffix| {
                                !suffix.is_empty()
                                    && suffix.chars().all(|c| c.is_ascii_digit())
                            })
                })
                .collect();
            replica_dirs.sort();
            if replica_dirs.is_empty() {
                still_missing.push(name);
                continue;
            }
            for dir in replica_dirs {
                let replica_name = dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&name)
                    .to_string();
                match loader.load(&dir) {
                    Ok(expert) => {
                        tracing::info!(
                            target: "neoethos_models::ensemble",
                            canonical = %name,
                            replica = %replica_name,
                            "loaded replica artifact as an independent voter"
                        );
                        outcome.loaded.push(Box::new(RenamedExpert {
                            name: replica_name,
                            inner: expert,
                        }));
                    }
                    Err(err) => {
                        outcome.degraded.push(ExpertLoadError::InvalidArtifact {
                            name: replica_name,
                            reason: err.to_string(),
                        });
                    }
                }
            }
        }
        outcome.missing = still_missing;

        // ── 2. Orphan artifacts: on disk, claimed by nothing ────────────
        let claimed: std::collections::HashSet<String> = outcome
            .loaded
            .iter()
            .map(|e| e.name().to_string())
            .chain(outcome.missing.iter().cloned())
            .chain(outcome.degraded.iter().map(|e| e.name().to_string()))
            .collect();
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if claimed.contains(dir_name) {
                    continue;
                }
                // Known non-voters, documented: `genetic` discovers
                // strategies (search-only exemption), `exit_agent` is
                // exit-pipeline-only (F-318). (`swarm_forecaster` votes
                // since D1.2.8 — no longer exempt.)
                let by_design = matches!(dir_name, "genetic" | "exit_agent");
                if dir_name == "exit_agent" {
                    // Audit #173/#175 — MAKE THE WASTE VISIBLE, do not decide it.
                    // `exit_agent` is pushed into the training plan on EVERY run
                    // (`training_orchestrator.rs:547`), is absent from
                    // DEFAULT_BOOTSTRAP_EXPERT_NAMES (`bootstrap.rs:102-149`),
                    // and is whitelisted here as a non-voter. So it trains every
                    // run and NOTHING consumes its `ExitDecision3` output. That
                    // is either an unshipped exit-side loop or wasted training
                    // time, and only the operator can say which — but until he
                    // does it must not read as a benign `info!` line.
                    tracing::warn!(
                        target: "neoethos_models::ensemble",
                        artifact = %dir_name,
                        "exit_agent was TRAINED and has NO CONSUMER — it emits ExitDecision3 and \
                         no production path reads it (audit #173/#175, decision pending: ship the \
                         exit-side loop or stop training it). Its training time is spent every run."
                    );
                } else if by_design {
                    tracing::info!(
                        target: "neoethos_models::ensemble",
                        artifact = %dir_name,
                        "trained artifact present but not a voter (by design)"
                    );
                } else {
                    tracing::warn!(
                        target: "neoethos_models::ensemble",
                        artifact = %dir_name,
                        "ORPHAN trained artifact — present on disk but no loader claims it; \
                         its training time is being wasted (check loader registration/naming)"
                    );
                }
            }
        }

        // ── 3. Numerical sanity: on disk ≠ fit to vote ──────────────────
        numeric_sanity_screen(root, &mut outcome);

        outcome
    }
}

/// Peer-relative ceiling for a recorded training loss. An artifact whose
/// `best_loss` is more than this many times the peer median is refused.
///
/// **Three orders of magnitude**, deliberately loose: the point is to catch a
/// model that diverged, not to rank models. The measured case (2026-08-09) is
/// `tide` at **1,308,811.5** and `tide_nf` at **51,699,690** against healthy
/// peers in the units of a cross-entropy — six to eight orders out, not three.
const PEER_LOSS_DIVERGENCE_FACTOR: f64 = 1_000.0;

/// Fewest peers that make a median meaningful. Below this the screen does not
/// run — and says so, rather than silently passing everything.
const MIN_PEERS_FOR_LOSS_SCREEN: usize = 3;

/// Absolute ceiling on a SAC entropy temperature (`final_alpha`).
///
/// SAC's temperature is an O(0.1–10) quantity by construction — the healthy
/// peers in the operator's store sit at ~0.48. There is exactly ONE sac
/// artifact per symbol, so no peer median exists for it and the bound has to be
/// absolute. `1e3` is two orders above any healthy value and six orders below
/// the measured divergence (**5.69e9**), so it cannot refuse a working model.
const MAX_SAC_FINAL_ALPHA: f64 = 1_000.0;

/// Refuse to let a numerically divergent artifact vote (audit #299 + #310).
///
/// **The gap this closes.** Nothing anywhere connected "an artifact is on disk"
/// to "an artifact may vote". Three artifacts in the operator's own store are
/// numerically broken — `tide` (best_loss 1,308,811.5), `tide_nf`
/// (51,699,690) and `sac` (final_alpha 5.69e9) — and all three are in
/// [`DEFAULT_BOOTSTRAP_EXPERT_NAMES`], i.e. all three load and all three would
/// vote. That is harmless *today* only because `live_ml_gate` is false and
/// `models.ensemble_voting.expert_weights` ships empty — which since 2026-08-10
/// (audit #168) is a SETTING the operator can change, not an unreachable
/// literal. The moment the gate is flipped, `tide`'s vote scales real position
/// size.
///
/// **This does not flip any gate.** It only removes divergent artifacts from
/// the voter set, loudly. Refusals become `degraded` entries, exactly like a
/// corrupt file — so the existing "no Classification3 voter loaded ⇒ refuse to
/// start auto-trade" contract keeps working and the trader falls back to
/// gene-only rather than voting on garbage. Fail-closed.
///
/// **No silent drops.** Every artifact is accounted for: screened, refused, or
/// `unscreened` because it records no number this can read (tree models, meta
/// learners). The counts are logged once per load.
///
/// **Call site.** Only [`ExpertRegistry::load_with_partial_replica_aware`],
/// which is the one production ensembles use (`bootstrap.rs:223`). The plain
/// `load_with_partial` is a lower-level primitive used by tests and is left
/// unscreened deliberately, so a test can assert on an unfiltered outcome.
fn numeric_sanity_screen(root: &Path, outcome: &mut ExpertLoadOutcome) {
    let names: Vec<String> = outcome.loaded.iter().map(|e| e.name().to_string()).collect();

    // Collect what each artifact recorded about its own training.
    let mut losses: Vec<(String, f64)> = Vec::new();
    let mut alphas: Vec<(String, f64)> = Vec::new();
    let mut unscreened: Vec<String> = Vec::new();
    for name in &names {
        let report = read_recorded_training_numbers(&root.join(name));
        match report {
            RecordedNumbers {
                best_loss: Some(l), ..
            } => losses.push((name.clone(), l)),
            RecordedNumbers {
                final_alpha: Some(a),
                ..
            } => alphas.push((name.clone(), a)),
            _ => unscreened.push(name.clone()),
        }
    }

    let mut refusals: Vec<(String, String)> = Vec::new();

    // ── Screen A: peer-relative loss ────────────────────────────────────
    let finite: Vec<f64> = losses
        .iter()
        .map(|(_, l)| *l)
        .filter(|l| l.is_finite() && *l > 0.0)
        .collect();
    if finite.len() >= MIN_PEERS_FOR_LOSS_SCREEN {
        let median = median_of(&finite);
        let ceiling = median * PEER_LOSS_DIVERGENCE_FACTOR;
        for (name, loss) in &losses {
            if !loss.is_finite() {
                refusals.push((
                    name.clone(),
                    format!("recorded training loss is not finite ({loss})"),
                ));
            } else if *loss > ceiling {
                refusals.push((
                    name.clone(),
                    format!(
                        "recorded training loss {loss:.3} exceeds {PEER_LOSS_DIVERGENCE_FACTOR:.0}× \
                         the peer median {median:.6} (ceiling {ceiling:.3}) — this artifact \
                         diverged during training and must not vote"
                    ),
                ));
            }
        }
    } else if !losses.is_empty() {
        tracing::warn!(
            target: "neoethos_models::ensemble",
            peers = finite.len(),
            required = MIN_PEERS_FOR_LOSS_SCREEN,
            "numeric sanity: too few peers to compute a loss median — the divergence screen \
             did NOT run for this symbol/timeframe"
        );
    }

    // ── Screen B: absolute SAC temperature ──────────────────────────────
    for (name, alpha) in &alphas {
        if !alpha.is_finite() || alpha.abs() > MAX_SAC_FINAL_ALPHA {
            refusals.push((
                name.clone(),
                format!(
                    "recorded final_alpha {alpha:e} is outside the sane entropy-temperature \
                     range (|alpha| ≤ {MAX_SAC_FINAL_ALPHA}) — this artifact diverged during \
                     training and must not vote"
                ),
            ));
        }
    }

    // ── Apply ───────────────────────────────────────────────────────────
    if !refusals.is_empty() {
        let refused: std::collections::HashSet<&str> =
            refusals.iter().map(|(n, _)| n.as_str()).collect();
        outcome.loaded.retain(|e| !refused.contains(e.name()));
        for (name, reason) in refusals.iter() {
            tracing::error!(
                target: "neoethos_models::ensemble",
                expert = %name,
                reason = %reason,
                "REFUSED numerically divergent artifact — it will NOT vote"
            );
            outcome.degraded.push(ExpertLoadError::InvalidArtifact {
                name: name.clone(),
                reason: reason.clone(),
            });
        }
    }

    tracing::info!(
        target: "neoethos_models::ensemble",
        screened_loss = losses.len(),
        screened_alpha = alphas.len(),
        unscreened = unscreened.len(),
        refused = refusals.len(),
        "numeric sanity screen complete (unscreened artifacts record no comparable number)"
    );
}

/// What an artifact wrote down about its own training, as far as it can be read
/// back from `<artifact_dir>/config.json`.
#[derive(Default)]
struct RecordedNumbers {
    /// `burn_training_report.best_loss` — every burn-trained deep model.
    best_loss: Option<f64>,
    /// `final_alpha` — the SAC entropy temperature.
    final_alpha: Option<f64>,
}

/// Read the recorded training numbers without deserialising the whole config
/// (each model kind writes a different shape). A missing/unreadable file is not
/// an error: it means "no number to screen", counted as `unscreened`.
fn read_recorded_training_numbers(dir: &Path) -> RecordedNumbers {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return RecordedNumbers::default();
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return RecordedNumbers::default();
    };
    RecordedNumbers {
        best_loss: json
            .get("burn_training_report")
            .and_then(|r| r.get("best_loss"))
            .and_then(|v| v.as_f64()),
        final_alpha: json.get("final_alpha").and_then(|v| v.as_f64()),
    }
}

/// Median of a non-empty slice. Callers guarantee non-empty.
fn median_of(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Thin rename shim for replica ensemble members: delegates everything
/// to the wrapped expert but reports the replica's own name
/// (`transformer_01`) so soft-voting treats each replica as an
/// independent voter and per-expert weights/exclusions stay addressable.
struct RenamedExpert {
    name: String,
    inner: Box<dyn ExpertModel>,
}

impl ExpertModel for RenamedExpert {
    fn name(&self) -> &str {
        &self.name
    }
    fn family(&self) -> ModelFamily {
        self.inner.family()
    }
    fn output_kind(&self) -> ExpertOutputKind {
        self.inner.output_kind()
    }
    fn feature_columns(&self) -> &[String] {
        self.inner.feature_columns()
    }
    fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
        self.inner.predict(df)
    }
}

impl Default for ExpertRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// EnsemblePredictor trait
// ---------------------------------------------------------------------------

/// Load-reporting contract over a set of loaded experts.
///
/// **2026-08-09 (batch D4): this trait no longer produces predictions.** It
/// used to require `fn predict(&self, df) -> Array2<f32>`, a flat
/// `[p_neutral, p_buy, p_sell]` average over every Classification3 expert.
/// That method had exactly one implementation and zero production callers —
/// the live loop and the replay loop both call
/// [`super::soft_voting::SoftVotingEnsemble::predict_with_roles`], which
/// returns [`EnsembleDecision`] and keeps `hmm_regime` / `isolation_forest`
/// OUT of the direction vote instead of averaging them into it. Inference is
/// therefore an inherent method on the concrete aggregator, not a trait
/// method; what the trait carries is the load snapshot the app reads at
/// `live_trading.rs:541`.
///
/// If a second aggregator ever arrives (the MoE gating network, D1.6), give it
/// `predict_with_roles` — do not resurrect a flat average alongside it.
pub trait EnsemblePredictor: Send + Sync {
    /// Snapshot of which experts loaded / missed / degraded at
    /// construction time. Used by the chrome to render the
    /// "running ensemble: X/Y experts active" banner.
    fn load_outcome(&self) -> &ExpertLoadOutcome;
    /// Read-only handle to the loaded experts. Useful for
    /// diagnostics + tests; production code should go through the
    /// concrete aggregator's `predict_with_roles`.
    fn experts(&self) -> &[Box<dyn ExpertModel>] {
        &self.load_outcome().loaded
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    // -- ExpertPrediction validate ---------------------------------------

    #[test]
    fn classification3_validates_normal_probabilities() {
        let pred = ExpertPrediction {
            kind: ExpertOutputKind::Classification3,
            values: vec![0.2, 0.5, 0.3],
        };
        assert!(pred.validate().is_ok());
    }

    #[test]
    fn classification3_rejects_wrong_length() {
        let pred = ExpertPrediction {
            kind: ExpertOutputKind::Classification3,
            values: vec![0.5, 0.5],
        };
        let err = pred.validate().expect_err("must reject");
        assert!(err.to_string().contains("expects 3 values"));
    }

    #[test]
    fn classification3_rejects_non_finite() {
        let pred = ExpertPrediction {
            kind: ExpertOutputKind::Classification3,
            values: vec![0.5, f32::NAN, 0.5],
        };
        assert!(pred.validate().is_err());
    }

    #[test]
    fn classification3_rejects_out_of_range_probability() {
        let pred = ExpertPrediction {
            kind: ExpertOutputKind::Classification3,
            values: vec![-0.1, 0.5, 0.6],
        };
        assert!(pred.validate().is_err());
    }

    #[test]
    fn classification3_rejects_sum_not_one() {
        let pred = ExpertPrediction {
            kind: ExpertOutputKind::Classification3,
            values: vec![0.5, 0.5, 0.5], // sums to 1.5
        };
        assert!(pred.validate().is_err());
    }

    #[test]
    fn anomaly_score_validates_zero_to_one() {
        let pred = ExpertPrediction {
            kind: ExpertOutputKind::AnomalyScore,
            values: vec![0.42],
        };
        assert!(pred.validate().is_ok());
    }

    #[test]
    fn anomaly_score_rejects_out_of_range() {
        let pred = ExpertPrediction {
            kind: ExpertOutputKind::AnomalyScore,
            values: vec![1.5],
        };
        assert!(pred.validate().is_err());
    }

    #[test]
    fn forecast1_accepts_arbitrary_real() {
        let pred = ExpertPrediction {
            kind: ExpertOutputKind::Forecast1,
            values: vec![-12.5],
        };
        assert!(pred.validate().is_ok());
    }

    #[test]
    fn action_values3_accepts_arbitrary_reals() {
        let pred = ExpertPrediction {
            kind: ExpertOutputKind::ActionValues3,
            values: vec![-1.5, 3.7, 0.0],
        };
        assert!(pred.validate().is_ok());
    }

    #[test]
    fn output_kind_expected_length_matches_variant() {
        assert_eq!(ExpertOutputKind::Classification3.expected_length(), 3);
        assert_eq!(ExpertOutputKind::ActionValues3.expected_length(), 3);
        assert_eq!(ExpertOutputKind::Forecast1.expected_length(), 1);
        assert_eq!(ExpertOutputKind::AnomalyScore.expected_length(), 1);
    }

    // -- Mock expert + loader ---------------------------------------------

    /// Deterministic mock for foundation testing. Returns a
    /// constant Classification3 prediction per row.
    struct MockExpert {
        name: String,
        feature_columns: Vec<String>,
        constant_probs: [f32; 3],
    }

    impl ExpertModel for MockExpert {
        fn name(&self) -> &str {
            &self.name
        }
        fn family(&self) -> ModelFamily {
            ModelFamily::Tree
        }
        fn output_kind(&self) -> ExpertOutputKind {
            ExpertOutputKind::Classification3
        }
        fn feature_columns(&self) -> &[String] {
            &self.feature_columns
        }
        fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
            let n = df.height();
            let out: Vec<ExpertPrediction> = (0..n)
                .map(|_| ExpertPrediction {
                    kind: ExpertOutputKind::Classification3,
                    values: self.constant_probs.to_vec(),
                })
                .collect();
            for p in &out {
                p.validate()?;
            }
            Ok(out)
        }
    }

    struct MockLoader {
        name: String,
        /// If true, `load` returns an Err for the categorisation
        /// test.
        fail_with: Option<String>,
    }

    impl ExpertLoader for MockLoader {
        fn name(&self) -> &str {
            &self.name
        }
        fn load(&self, _artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
            if let Some(reason) = &self.fail_with {
                anyhow::bail!("{reason}");
            }
            Ok(Box::new(MockExpert {
                name: self.name.clone(),
                feature_columns: vec!["f1".to_string(), "f2".to_string()],
                constant_probs: [0.2, 0.6, 0.2],
            }))
        }
    }

    // -- ExpertRegistry tests --------------------------------------------

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tempdir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join("neoethos-ensemble-foundation")
            .join(format!("{label}-{nanos}-{n}-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    // -- numeric sanity screen (#299 + #310) ------------------------------

    fn write_config(root: &std::path::Path, name: &str, json: &str) {
        let dir = root.join(name);
        fs::create_dir_all(&dir).expect("artifact dir");
        fs::write(dir.join("config.json"), json).expect("config.json");
    }

    fn burn_loss(loss: f64) -> String {
        format!("{{\"burn_training_report\":{{\"best_loss\":{loss}}}}}")
    }

    fn mock_expert(name: &str) -> Box<dyn ExpertModel> {
        Box::new(MockExpert {
            name: name.to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            constant_probs: [0.2, 0.6, 0.2],
        })
    }

    #[test]
    fn median_of_handles_even_and_odd_lengths() {
        assert_eq!(median_of(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median_of(&[1.0, 2.0, 3.0, 5.0]), 2.5);
        assert_eq!(median_of(&[7.0]), 7.0);
    }

    #[test]
    fn recorded_numbers_read_both_shapes_and_tolerate_absence() {
        let root = tempdir("recorded-numbers");
        write_config(&root, "tide", &burn_loss(1308811.5));
        write_config(&root, "sac", "{\"final_alpha\":5693284000.0}");
        write_config(&root, "lightgbm", "{\"num_leaves\":31}");
        assert_eq!(
            read_recorded_training_numbers(&root.join("tide")).best_loss,
            Some(1308811.5)
        );
        assert_eq!(
            read_recorded_training_numbers(&root.join("sac")).final_alpha,
            Some(5693284000.0)
        );
        let none = read_recorded_training_numbers(&root.join("lightgbm"));
        assert!(none.best_loss.is_none() && none.final_alpha.is_none());
        // A directory with no config.json at all must not panic — it is
        // "nothing to screen", not an error.
        let absent = read_recorded_training_numbers(&root.join("does_not_exist"));
        assert!(absent.best_loss.is_none() && absent.final_alpha.is_none());
    }

    /// The exact numbers measured in the operator's store on 2026-08-09:
    /// eleven burn artifacts at ~0.48 plus `tide` at 1,308,811.5 and `tide_nf`
    /// at 51,699,690, and one `sac` at final_alpha 5.69e9. The screen must
    /// refuse exactly those three and keep every healthy peer.
    #[test]
    fn divergent_artifacts_are_refused_and_healthy_peers_kept() {
        let root = tempdir("numeric-sanity");
        let healthy = [
            ("nbeats", 0.48608416),
            ("timesnet", 0.48188955),
            ("mlp", 0.48186147),
            ("nbeatsx_nf", 0.480542),
            ("kan", 0.48044226),
            ("patchtst", 0.47902533),
            ("tabnet", 0.47879004),
        ];
        for (n, l) in healthy {
            write_config(&root, n, &burn_loss(l));
        }
        write_config(&root, "tide", &burn_loss(1308811.5));
        write_config(&root, "tide_nf", &burn_loss(51699690.0));
        write_config(&root, "sac", "{\"final_alpha\":5693284000.0}");
        // `lightgbm` records no comparable number — it must survive untouched
        // rather than being refused for lack of evidence.
        write_config(&root, "lightgbm", "{\"num_leaves\":31}");

        let mut outcome = ExpertLoadOutcome {
            loaded: healthy
                .iter()
                .map(|(n, _)| mock_expert(n))
                .chain(
                    ["tide", "tide_nf", "sac", "lightgbm"]
                        .into_iter()
                        .map(mock_expert),
                )
                .collect(),
            missing: Vec::new(),
            degraded: Vec::new(),
        };

        numeric_sanity_screen(&root, &mut outcome);

        let kept: Vec<&str> = outcome.loaded.iter().map(|e| e.name()).collect();
        for (n, _) in healthy {
            assert!(kept.contains(&n), "healthy peer {n} must keep voting");
        }
        assert!(kept.contains(&"lightgbm"), "unscreened artifact must survive");
        for bad in ["tide", "tide_nf", "sac"] {
            assert!(!kept.contains(&bad), "{bad} must be refused");
            assert!(
                outcome.degraded.iter().any(|d| d.name() == bad),
                "{bad} must be recorded as degraded, not silently dropped"
            );
        }
    }

    /// Fail-SAFE, not fail-loud-and-wrong: with fewer than three peers a median
    /// means nothing, so the screen must refuse NOTHING rather than guess.
    #[test]
    fn too_few_peers_refuses_nothing() {
        let root = tempdir("numeric-sanity-few");
        write_config(&root, "mlp", &burn_loss(0.48));
        write_config(&root, "tide", &burn_loss(1308811.5));
        let mut outcome = ExpertLoadOutcome {
            loaded: ["mlp", "tide"].into_iter().map(mock_expert).collect(),
            missing: Vec::new(),
            degraded: Vec::new(),
        };
        numeric_sanity_screen(&root, &mut outcome);
        assert_eq!(outcome.loaded.len(), 2);
        assert!(outcome.degraded.is_empty());
    }

    #[test]
    fn registry_register_rejects_duplicate_name() {
        let mut reg = ExpertRegistry::new();
        reg.register(Box::new(MockLoader {
            name: "lightgbm".into(),
            fail_with: None,
        }))
        .expect("first registration");
        let err = reg
            .register(Box::new(MockLoader {
                name: "lightgbm".into(),
                fail_with: None,
            }))
            .expect_err("duplicate must error");
        assert!(err.to_string().contains("already registered"));
    }

    #[test]
    fn registry_has_loader_query() {
        let mut reg = ExpertRegistry::new();
        assert!(!reg.has_loader("xgboost"));
        reg.register(Box::new(MockLoader {
            name: "xgboost".into(),
            fail_with: None,
        }))
        .expect("register");
        assert!(reg.has_loader("xgboost"));
    }

    #[test]
    fn registered_names_returns_sorted_names() {
        let mut reg = ExpertRegistry::new();
        for n in ["xgboost", "lightgbm", "mlp"] {
            reg.register(Box::new(MockLoader {
                name: n.into(),
                fail_with: None,
            }))
            .expect("register");
        }
        let names = reg.registered_names();
        assert_eq!(names, vec!["lightgbm", "mlp", "xgboost"]);
    }

    #[test]
    fn load_with_partial_returns_loaded_when_artifacts_present() {
        let root = tempdir("loaded");
        fs::create_dir_all(root.join("lightgbm")).expect("lightgbm dir");
        fs::create_dir_all(root.join("xgboost")).expect("xgboost dir");
        let mut reg = ExpertRegistry::new();
        reg.register(Box::new(MockLoader {
            name: "lightgbm".into(),
            fail_with: None,
        }))
        .expect("register");
        reg.register(Box::new(MockLoader {
            name: "xgboost".into(),
            fail_with: None,
        }))
        .expect("register");
        let outcome = reg.load_with_partial(&root, &["lightgbm", "xgboost"]);
        assert_eq!(outcome.loaded_count(), 2);
        assert_eq!(outcome.missing_count(), 0);
        assert_eq!(outcome.degraded_count(), 0);
        assert!(outcome.has_any_loaded());
        assert_eq!(outcome.loaded_names(), vec!["lightgbm", "xgboost"]);
    }

    #[test]
    fn load_with_partial_reports_missing_when_artifact_dir_absent() {
        let root = tempdir("missing");
        // Only create lightgbm dir; xgboost is missing.
        fs::create_dir_all(root.join("lightgbm")).expect("lightgbm dir");
        let mut reg = ExpertRegistry::new();
        reg.register(Box::new(MockLoader {
            name: "lightgbm".into(),
            fail_with: None,
        }))
        .expect("register");
        reg.register(Box::new(MockLoader {
            name: "xgboost".into(),
            fail_with: None,
        }))
        .expect("register");
        let outcome = reg.load_with_partial(&root, &["lightgbm", "xgboost"]);
        assert_eq!(outcome.loaded_count(), 1);
        assert_eq!(outcome.missing_count(), 1);
        assert_eq!(outcome.degraded_count(), 0);
        assert_eq!(outcome.missing, vec!["xgboost"]);
    }

    #[test]
    fn load_with_partial_categorises_load_errors() {
        let root = tempdir("degraded");
        // Create dirs but the loaders will return categorised errors.
        for n in ["a_io", "a_backend", "a_version", "a_invalid"] {
            fs::create_dir_all(root.join(n)).expect("dir");
        }
        let mut reg = ExpertRegistry::new();
        reg.register(Box::new(MockLoader {
            name: "a_io".into(),
            fail_with: Some("file not found while reading".to_string()),
        }))
        .expect("register");
        reg.register(Box::new(MockLoader {
            name: "a_backend".into(),
            fail_with: Some("CUDA backend failed to initialise".to_string()),
        }))
        .expect("register");
        reg.register(Box::new(MockLoader {
            name: "a_version".into(),
            fail_with: Some("artifact version mismatch".to_string()),
        }))
        .expect("register");
        reg.register(Box::new(MockLoader {
            name: "a_invalid".into(),
            fail_with: Some("metadata.json malformed".to_string()),
        }))
        .expect("register");

        let outcome =
            reg.load_with_partial(&root, &["a_io", "a_backend", "a_version", "a_invalid"]);
        assert_eq!(outcome.loaded_count(), 0);
        assert_eq!(outcome.missing_count(), 0);
        assert_eq!(outcome.degraded_count(), 4);

        // Spot-check each categorisation.
        let mut by_name: HashMap<&str, &ExpertLoadError> = HashMap::new();
        for d in &outcome.degraded {
            by_name.insert(d.name(), d);
        }
        assert!(matches!(
            by_name.get("a_io"),
            Some(ExpertLoadError::Io { .. })
        ));
        assert!(matches!(
            by_name.get("a_backend"),
            Some(ExpertLoadError::Backend { .. })
        ));
        assert!(matches!(
            by_name.get("a_version"),
            Some(ExpertLoadError::IncompatibleVersion { .. })
        ));
        assert!(matches!(
            by_name.get("a_invalid"),
            Some(ExpertLoadError::InvalidArtifact { .. })
        ));
    }

    #[test]
    fn load_with_partial_reports_invalid_when_no_loader_registered() {
        let root = tempdir("no_loader");
        // No loader registered for "ghost"; the dir's presence
        // doesn't matter — the registry rejects before touching disk.
        let reg = ExpertRegistry::new();
        let outcome = reg.load_with_partial(&root, &["ghost"]);
        assert_eq!(outcome.loaded_count(), 0);
        assert_eq!(outcome.degraded_count(), 1);
        assert!(matches!(
            outcome.degraded[0],
            ExpertLoadError::InvalidArtifact { .. }
        ));
        assert_eq!(outcome.degraded[0].name(), "ghost");
    }

    #[test]
    fn load_with_partial_detects_loader_name_typo() {
        // The loader's `load` returns an expert whose name does
        // NOT match the registry key — this is a programmer error
        // that the registry must catch to prevent silent
        // mis-aggregation.
        let root = tempdir("typo");
        fs::create_dir_all(root.join("lightgbm")).expect("dir");
        struct TypoLoader;
        impl ExpertLoader for TypoLoader {
            fn name(&self) -> &str {
                "lightgbm"
            }
            fn load(&self, _: &Path) -> Result<Box<dyn ExpertModel>> {
                Ok(Box::new(MockExpert {
                    name: "actually_xgboost".to_string(),
                    feature_columns: Vec::new(),
                    constant_probs: [0.3, 0.4, 0.3],
                }))
            }
        }
        let mut reg = ExpertRegistry::new();
        reg.register(Box::new(TypoLoader)).expect("register");
        let outcome = reg.load_with_partial(&root, &["lightgbm"]);
        assert_eq!(outcome.loaded_count(), 0);
        assert_eq!(outcome.degraded_count(), 1);
        match &outcome.degraded[0] {
            ExpertLoadError::InvalidArtifact { reason, .. } => {
                assert!(reason.contains("actually_xgboost"));
                assert!(reason.contains("lightgbm"));
            }
            other => panic!("expected InvalidArtifact, got {other:?}"),
        }
    }

    #[test]
    fn empty_outcome_round_trips_counts() {
        let o = ExpertLoadOutcome::empty();
        assert_eq!(o.loaded_count(), 0);
        assert_eq!(o.missing_count(), 0);
        assert_eq!(o.degraded_count(), 0);
        assert_eq!(o.requested_count(), 0);
        assert!(!o.has_any_loaded());
        assert!(o.loaded_names().is_empty());
    }

    // -- EnsemblePredictor trait round-trip -----------------------------

    /// Minimal in-test EnsemblePredictor. Used only to pin the
    /// trait's shape now that it carries the load snapshot alone.
    struct StubEnsemble {
        outcome: ExpertLoadOutcome,
    }

    impl EnsemblePredictor for StubEnsemble {
        fn load_outcome(&self) -> &ExpertLoadOutcome {
            &self.outcome
        }
    }

    #[test]
    fn ensemble_predictor_trait_round_trips_through_box_dyn() {
        let outcome = ExpertLoadOutcome {
            loaded: vec![Box::new(MockExpert {
                name: "lightgbm".to_string(),
                feature_columns: vec!["f1".to_string()],
                constant_probs: [0.2, 0.6, 0.2],
            })],
            missing: vec!["xgboost".to_string()],
            degraded: vec![],
        };
        let ens: Box<dyn EnsemblePredictor> = Box::new(StubEnsemble { outcome });
        assert_eq!(ens.load_outcome().loaded_count(), 1);
        assert_eq!(ens.load_outcome().missing_count(), 1);
        assert_eq!(ens.experts().len(), 1);
        assert_eq!(ens.experts()[0].name(), "lightgbm");
    }

    #[test]
    fn mock_expert_predict_returns_one_per_row() {
        let exp = MockExpert {
            name: "mock".to_string(),
            feature_columns: vec!["f1".to_string()],
            constant_probs: [0.2, 0.5, 0.3],
        };
        let df = df!("f1" => &[1.0_f32, 2.0, 3.0]).expect("df");
        let preds = exp.predict(&df).expect("predict");
        assert_eq!(preds.len(), 3);
        for p in &preds {
            assert_eq!(p.kind, ExpertOutputKind::Classification3);
            assert_eq!(p.values, vec![0.2, 0.5, 0.3]);
        }
    }
}
