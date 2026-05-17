//! Inference-half foundation for the 33-model ensemble.
//!
//! ## Why this module exists
//!
//! The training side of forex-models already trains all 33 expert
//! families ([`crate::runtime::capabilities::KNOWN_MODEL_NAMES`]) in
//! parallel via [`crate::training_orchestrator::TrainingOrchestrator`]
//! and saves each one's artifacts to disk. But until this module
//! landed, the **inference side did not exist**: no runtime code
//! loaded the trained experts back from disk, no orchestrator ran
//! `predict_proba` on each, and no aggregator combined their outputs.
//! The 33-model "ensemble" was 33 INDEPENDENT models with no
//! consumer.
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
//! - **D1.3** — `SoftVotingEnsemble` — the first concrete
//!   [`EnsemblePredictor`] (weighted average of loaded experts'
//!   3-class probabilities). Useable from day one against existing
//!   trained artifacts.
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
//! render "Running ensemble: 24/33 experts active — 9 degraded
//! (see system log)". This is the **tracked degradation** the
//! operator explicitly asked for.
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
pub mod deep_classification_adapters;
pub mod deep_timeseries_adapters;
pub mod meta_adapters;
pub mod mixed_adapters;
pub mod tree_adapters;

pub use deep_classification_adapters::{
    register_deep_classification_loaders, KanAdapter, KanLoader, MlpAdapter, MlpLoader,
    TabNetAdapter, TabNetLoader,
};
pub use deep_timeseries_adapters::{
    register_deep_timeseries_loaders, NBeatsAdapter, NBeatsLoader, NBeatsxNfAdapter,
    NBeatsxNfLoader, PatchTstAdapter, PatchTstLoader, TiDEAdapter, TiDELoader, TiDENfAdapter,
    TiDENfLoader, TimesNetAdapter, TimesNetLoader, TransformerAdapter, TransformerLoader,
};
pub use mixed_adapters::{
    register_mixed_loaders, IsolationForestAdapter, IsolationForestLoader,
    OnlineHoeffdingAdapter, OnlineHoeffdingLoader, OnlinePaAdapter, OnlinePaLoader,
};
pub use meta_adapters::{
    register_meta_loaders, BayesLogitAdapter, BayesLogitLoader, ConformalGateAdapter,
    ConformalGateLoader, ElasticNetAdapter, ElasticNetLoader, LogisticAdapter, LogisticLoader,
    MetaBlenderAdapter, MetaBlenderLoader, MetaStackAdapter, MetaStackLoader,
    ProbabilityCalibratorAdapter, ProbabilityCalibratorLoader,
};
pub use tree_adapters::{
    register_tree_loaders, CatboostAdapter, CatboostLoader, LightGbmAdapter, LightGbmLoader,
    SklearsTreeAdapter, SklearsTreeLoader, XgboostAdapter, XgboostLoader,
};

// ---------------------------------------------------------------------------
// Expert output taxonomy
// ---------------------------------------------------------------------------

/// What kind of native output an [`ExpertModel`] produces per input
/// row. The [`EnsemblePredictor`] consults this when deciding how
/// to combine an expert's predictions with the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertOutputKind {
    /// 3-class probability vector `[p_sell, p_neutral, p_buy]`,
    /// rows sum to 1.0. Produced by classification-head experts
    /// (LightGBM, XGBoost, MLP-classifier, transformer-classifier,
    /// etc.). This is the most directly aggregable output for a
    /// trading decision.
    Classification3,
    /// 3-action Q-values for `[sell, hold, buy]` from an RL agent
    /// (dqn). Not a probability distribution — values are arbitrary
    /// reals; the action with the highest Q is the recommended
    /// action. Soft-voting treats argmax as a Classification3 vote.
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
}

impl fmt::Display for ExpertOutputKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Classification3 => "classification_3",
            Self::ActionValues3 => "action_values_3",
            Self::Forecast1 => "forecast_1",
            Self::AnomalyScore => "anomaly_score",
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
        }
    }
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
            ExpertOutputKind::Classification3 => {
                for v in &self.values {
                    if *v < -1e-4 || *v > 1.0 + 1e-4 {
                        anyhow::bail!(
                            "Classification3 probability out of [0, 1]: {}",
                            v
                        );
                    }
                }
                let sum: f32 = self.values.iter().sum();
                if (sum - 1.0).abs() > 1e-2 {
                    anyhow::bail!(
                        "Classification3 probabilities do not sum to 1.0: sum = {}",
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
/// All 33 expert families in
/// [`crate::runtime::capabilities::KNOWN_MODEL_NAMES`] will
/// implement this trait via a thin adapter (D1.2.x follow-up
/// commits). The aggregating [`EnsemblePredictor`] holds a
/// `Vec<Box<dyn ExpertModel>>` and treats each one uniformly.
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

/// Central registry of [`ExpertLoader`]s. The forex-app bootstrap
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
}

impl Default for ExpertRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// EnsemblePredictor trait
// ---------------------------------------------------------------------------

/// Aggregator contract over a set of loaded experts.
///
/// The concrete implementations are:
/// - `SoftVotingEnsemble` (D1.3) — weighted average of every
///   loaded expert's `Classification3` / `ActionValues3` outputs.
/// - `MoeEnsemble` (D1.6) — gating network that learns which
///   experts to trust under which conditions, trained jointly
///   (operator directive 2026-05-17 — diversity from
///   specialization, not artificial feature restrictions).
///
/// Output is a `(n_rows, 3)` `ndarray::Array2<f32>` of
/// `[p_sell, p_neutral, p_buy]` probabilities. Downstream callers
/// (the auto-trade producer's `ModelPredictor` adapter — D1.3 also)
/// pick the argmax + confidence and map to `AutoTradeSide`.
pub trait EnsemblePredictor: Send + Sync {
    /// Run inference on every row of `df`. Returns a `(n_rows, 3)`
    /// matrix of `[p_sell, p_neutral, p_buy]`.
    fn predict(&self, df: &DataFrame) -> Result<ndarray::Array2<f32>>;
    /// Snapshot of which experts loaded / missed / degraded at
    /// construction time. Used by the chrome to render the
    /// "running ensemble: X/Y experts active" banner.
    fn load_outcome(&self) -> &ExpertLoadOutcome;
    /// Read-only handle to the loaded experts. Useful for
    /// diagnostics + tests; production code should go through
    /// [`Self::predict`].
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
    use ndarray::Array2;
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
            .join("forex-ai-ensemble-foundation")
            .join(format!("{label}-{nanos}-{n}-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        dir
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

        let outcome = reg.load_with_partial(
            &root,
            &["a_io", "a_backend", "a_version", "a_invalid"],
        );
        assert_eq!(outcome.loaded_count(), 0);
        assert_eq!(outcome.missing_count(), 0);
        assert_eq!(outcome.degraded_count(), 4);

        // Spot-check each categorisation.
        let mut by_name: HashMap<&str, &ExpertLoadError> = HashMap::new();
        for d in &outcome.degraded {
            by_name.insert(d.name(), d);
        }
        assert!(matches!(by_name.get("a_io"), Some(ExpertLoadError::Io { .. })));
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

    /// Minimal in-test EnsemblePredictor that returns a fixed
    /// probability vector regardless of input. Used only to pin
    /// the trait's shape.
    struct StubEnsemble {
        outcome: ExpertLoadOutcome,
        constant: [f32; 3],
    }

    impl EnsemblePredictor for StubEnsemble {
        fn predict(&self, df: &DataFrame) -> Result<Array2<f32>> {
            let n = df.height();
            let flat: Vec<f32> = (0..n).flat_map(|_| self.constant.iter().copied()).collect();
            Ok(Array2::from_shape_vec((n, 3), flat)?)
        }
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
        let ens: Box<dyn EnsemblePredictor> = Box::new(StubEnsemble {
            outcome,
            constant: [0.1, 0.7, 0.2],
        });
        // Build a 4-row DataFrame.
        let df = df!("f1" => &[1.0_f32, 2.0, 3.0, 4.0]).expect("df");
        let probs = ens.predict(&df).expect("predict");
        assert_eq!(probs.shape(), &[4, 3]);
        for row in probs.outer_iter() {
            assert!((row[0] - 0.1).abs() < 1e-6);
            assert!((row[1] - 0.7).abs() < 1e-6);
            assert!((row[2] - 0.2).abs() < 1e-6);
        }
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
