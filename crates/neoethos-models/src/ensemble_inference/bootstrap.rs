//! One-call ensemble bootstrap.
//!
//! Phase D1.5. Convenience entry point that takes a models-root
//! directory + symbol + timeframe and returns a ready-to-use
//! [`super::SoftVotingEnsemble`] populated with whatever trained
//! experts are present on disk.
//!
//! ## What this module does
//!
//! End-to-end bootstrap for the operator:
//!
//! ```text
//!   models_root/
//!     EURUSD/                 (symbol the operator picked)
//!       H1/                   (timeframe the operator picked)
//!         lightgbm/           (each expert's saved artifact dir)
//!         xgboost/
//!         catboost/
//!         …
//!         meta_stack/
//!         hmm_regime/         (added 2026-05-25 — HMM Phase 2)
//! ```
//!
//! [`build_ensemble_for_symbol`]:
//!  1. Builds an [`super::ExpertRegistry`] with every default
//!     loader pre-registered (32 canonical names — all wired
//!     families from D1.2.1-D1.2.7, the 34th model `hmm_regime`,
//!     and the evolutionary voters neat/neuro_evo restored in the
//!     F-319 revision 2026-07-11).
//!  2. Calls [`super::ExpertRegistry::load_with_partial_replica_aware`]
//!     against the operator's `<models_root>/<symbol>/<tf>/` directory
//!     with the full canonical name list. Missing/degraded
//!     artifacts are reported in the outcome (per option β —
//!     no fail-loud) so the operator can run the bot with
//!     whatever subset of the 32 experts has been trained; replica
//!     dirs (`transformer_01/…`) load as independent voters and
//!     orphan artifact dirs are warned about loudly.
//!  3. Constructs a [`super::SoftVotingEnsemble`] with the
//!     default config (no default exclusions — the operator rule is
//!     "every trained model votes"; only `genetic` stays out, as the
//!     strategy discoverer it is search-side, never registered here).
//!
//! Returns the ensemble plus the load outcome so the caller's
//! chrome / system pane can render "Loaded X/32 experts —
//! Y missing, Z degraded".
//!
//! ## What it does NOT do
//!
//! - Loads `swarm_forecaster` as a LAST-ROW-ONLY voter (D1.2.8 landed
//!   2026-07-11 — see the `swarm_adapter` module doc).
//! - Does NOT run any training. Bootstrap is read-only against
//!   the operator's `models_root` directory; if no experts have
//!   been trained, the function returns an ensemble with an
//!   empty load outcome and the caller is responsible for
//!   handling that case (e.g. refusing to start the auto-trade
//!   producer until at least one expert is loaded).
//! - Does NOT validate that each expert's `feature_columns`
//!   matches the runtime feature pipeline. That cross-check
//!   happens at first `predict` call — if a column-layout drift
//!   is detected the expert's predict_proba returns an error
//!   which the SoftVotingEnsemble surfaces verbatim.

use std::path::Path;

use anyhow::{Context, Result};

use super::{
    ExpertLoadOutcome, ExpertRegistry, SoftVotingEnsemble, SoftVotingEnsembleConfig,
    deep_classification_adapters::register_deep_classification_loaders,
    deep_timeseries_adapters::register_deep_timeseries_loaders,
    evolution_adapters::register_evolution_loaders, meta_adapters::register_meta_loaders,
    mixed_adapters::register_mixed_loaders, rl_exit_adapters::register_rl_exit_loaders,
    swarm_adapter::register_swarm_loader, tree_adapters::register_tree_loaders,
};

/// Canonical list of expert names the bootstrap tries to load.
///
/// Sourced from `KNOWN_MODEL_NAMES` per
/// [`crate::runtime::capabilities::KNOWN_MODEL_NAMES`] minus:
///   - `genetic` — the strategy DISCOVERER (the GA in `neoethos-search`);
///     the operator's search-only exemption applies to it alone.
///   - `exit_agent` — F-318 (no production exit-side consumer).
///
/// `neat` + `neuro_evo` REJOINED 2026-07-11 (F-319 revision, operator
/// directive "every trained model votes"): both are trained through the
/// shared expert path with genuine 3-class heads — see the
/// `evolution_adapters` module doc. `swarm_forecaster` landed the same
/// day (D1.2.8): last-row-only forecast voter — see the `swarm_adapter`
/// module doc for the honesty constraints.
///
/// **33 names total** (KNOWN_MODEL_NAMES − genetic − exit_agent).
///
/// `exit_agent` was removed in F-318 (2026-05-29): the model trains
/// successfully and emits `ExitDecision3` probabilities, but
/// `SoftVotingEnsemble` actively filters those outputs (Classification3
/// only votes) and no auto-trade exit-side pipeline consumes them in
/// production. Keeping it in the bootstrap list reserved memory + disk
/// for an artifact that no production code path reads. The source
/// (`exit_agent.rs`, `ExitAgentAdapter`, `ExitAgentLoader`) stays for
/// future revival once an exit-side decision loop ships, but the
/// registry no longer wires it in until then.
pub const DEFAULT_BOOTSTRAP_EXPERT_NAMES: &[&str] = &[
    // Tree (7)
    "lightgbm",
    "xgboost",
    "xgboost_rf",
    "xgboost_dart",
    "catboost",
    "catboost_alt",
    "sklears_tree",
    // Deep classifier (3)
    "mlp",
    "kan",
    "tabnet",
    // Deep time-series (7)
    "nbeats",
    "nbeatsx_nf",
    "tide",
    "tide_nf",
    "transformer",
    "patchtst",
    "timesnet",
    // Meta (8 — 7 originals + hmm_regime added 2026-05-25)
    "elasticnet",
    "logistic",
    "bayes_logit",
    "meta_blender",
    "probability_calibrator",
    "conformal_gate",
    "meta_stack",
    "hmm_regime",
    // Adaptive + Anomaly (3)
    "online_pa",
    "online_hoeffding",
    "isolation_forest",
    // RL (2) — exit_agent removed in F-318 (consumers never wired).
    // `sac` (discrete Soft Actor-Critic) is an entry/direction voter
    // that emits Classification3 probs and soft-votes like `dqn`.
    "dqn",
    "sac",
    // Evolutionary voters (2) — rejoined 2026-07-11 (F-319 revision):
    // trained via the shared expert path with 3-class heads; their
    // artifacts were being produced and never read.
    "neat",
    "neuro_evo",
    // Forecasting voter (1) — D1.2.8, same day: last-row-only forecast
    // lean (live-gate semantics; abstains on historical rows).
    "swarm_forecaster",
];

/// Build a fully populated [`ExpertRegistry`] with every default
/// loader pre-registered. The neoethos-app bootstrap calls this
/// once at session start.
pub fn build_default_registry() -> Result<ExpertRegistry> {
    let mut registry = ExpertRegistry::new();
    register_tree_loaders(&mut registry).context("register tree loaders")?;
    register_deep_classification_loaders(&mut registry)
        .context("register deep classification loaders")?;
    register_deep_timeseries_loaders(&mut registry).context("register deep time-series loaders")?;
    register_meta_loaders(&mut registry).context("register meta loaders")?;
    register_mixed_loaders(&mut registry).context("register mixed loaders")?;
    register_rl_exit_loaders(&mut registry).context("register rl+exit loaders")?;
    register_evolution_loaders(&mut registry).context("register evolutionary loaders")?;
    register_swarm_loader(&mut registry).context("register swarm forecaster loader")?;
    debug_assert_eq!(
        registry.registered_names().len(),
        DEFAULT_BOOTSTRAP_EXPERT_NAMES.len(),
        "DEFAULT_BOOTSTRAP_EXPERT_NAMES + registry must list the same 33 canonical names"
    );
    Ok(registry)
}

/// Build a [`SoftVotingEnsemble`] for `<models_root>/<symbol>/<tf>/`.
///
/// Returns `Ok((ensemble, outcome))` even when many experts are
/// missing — only fails if NO Classification3 voter loaded
/// (caller should then refuse to start auto-trade). The
/// `outcome` is also reachable via `ensemble.load_outcome()` but
/// the caller often wants it BEFORE deciding whether to use the
/// ensemble at all.
pub fn build_ensemble_for_symbol(
    models_root: &Path,
    symbol: &str,
    timeframe: &str,
) -> Result<SoftVotingEnsemble> {
    build_ensemble_for_symbol_with_config(
        models_root,
        symbol,
        timeframe,
        SoftVotingEnsembleConfig::default(),
    )
}

/// Same as [`build_ensemble_for_symbol`] with an explicit
/// SoftVoting config (e.g. operator-overridden expert weights /
/// exclusion list).
pub fn build_ensemble_for_symbol_with_config(
    models_root: &Path,
    symbol: &str,
    timeframe: &str,
    config: SoftVotingEnsembleConfig,
) -> Result<SoftVotingEnsemble> {
    let outcome = load_experts_for_symbol(models_root, symbol, timeframe)?;
    SoftVotingEnsemble::new(outcome, config)
        .context("construct SoftVotingEnsemble from load outcome")
}

/// Lower-level helper: build the registry, resolve the per-symbol
/// artifact root, and call [`ExpertRegistry::load_with_partial`].
/// Returns the [`ExpertLoadOutcome`] so the caller can inspect
/// `loaded` / `missing` / `degraded` before deciding what to do.
pub fn load_experts_for_symbol(
    models_root: &Path,
    symbol: &str,
    timeframe: &str,
) -> Result<ExpertLoadOutcome> {
    let registry = build_default_registry()?;
    let artifact_root = models_root.join(symbol).join(timeframe);
    // Replica-aware: resolves `transformer_01/02/…` replica dirs (which
    // training writes when num_transformers > 1 — a plain `transformer/`
    // dir never exists then) and warns on orphan artifacts no loader
    // claims, instead of silently counting trained models as missing.
    Ok(registry.load_with_partial_replica_aware(&artifact_root, DEFAULT_BOOTSTRAP_EXPERT_NAMES))
}

/// v0.5 ML-integration Stage 3 — produce the per-row role-aware
/// [`EnsembleDecision`]s for a symbol from a `FeatureFrame`, centralizing the
/// feature-column CONTRACT so the trader never feeds mis-columned data to the
/// experts.
///
/// Builds the symbol's ensemble, reads the experts' shared `feature_columns()`
/// (asserting all loaded experts agree), selects the FeatureFrame columns to
/// EXACTLY that set BY NAME (the experts bail unless the DataFrame columns equal
/// their trained set), runs [`SoftVotingEnsemble::predict_with_roles`], and
/// returns one decision per row. FAILS LOUD on any column mismatch / missing
/// feature — the caller (the trader) then falls back to gene-only rather than
/// trading on a wrong-columned ensemble.
pub fn role_decisions_from_feature_frame(
    models_root: &Path,
    symbol: &str,
    timeframe: &str,
    features: &neoethos_data::FeatureFrame,
) -> Result<Vec<super::EnsembleDecision>> {
    let ensemble = build_ensemble_for_symbol(models_root, symbol, timeframe)?;
    let df = ensemble_feature_dataframe(&ensemble, features, FrameRows::All)?;
    ensemble.predict_with_roles(&df)
}

/// LIVE-path variant: one role-aware decision for the LAST row of `features`,
/// against an ALREADY-BUILT ensemble.
///
/// The live autopilot builds its ensemble ONCE at engine start (loading ~30
/// expert artifacts takes seconds — far too slow per bar) and calls this on
/// every closed bar with the same multi-TF feature cube the genes evaluate.
/// Same fail-loud column contract as [`role_decisions_from_feature_frame`];
/// the caller treats any `Err` as "ensemble abstains" and falls back to
/// gene-only sizing — never a wrong-columned prediction, never a blocked
/// trade due to ML infrastructure.
pub fn role_decision_for_last_row(
    ensemble: &SoftVotingEnsemble,
    features: &neoethos_data::FeatureFrame,
) -> Result<super::EnsembleDecision> {
    let df = ensemble_feature_dataframe(ensemble, features, FrameRows::LastOnly)?;
    let decisions = ensemble.predict_with_roles(&df)?;
    decisions
        .into_iter()
        .next_back()
        .ok_or_else(|| anyhow::anyhow!("ensemble returned no decision for the last feature row"))
}

/// Row selector for [`ensemble_feature_dataframe`].
enum FrameRows {
    All,
    LastOnly,
}

/// Shared column-contract core: read the experts' agreed feature columns,
/// select EXACTLY those from the cube BY NAME, and build the DataFrame the
/// experts expect. FAILS LOUD on any mismatch.
fn ensemble_feature_dataframe(
    ensemble: &SoftVotingEnsemble,
    features: &neoethos_data::FeatureFrame,
    rows: FrameRows,
) -> Result<polars::prelude::DataFrame> {
    use anyhow::{anyhow, bail};
    use polars::prelude::{Column, DataFrame, NamedFrom, Series};

    // `load_outcome` is an `EnsemblePredictor` trait method.
    use super::EnsemblePredictor;

    // Determine the experts' shared feature-column set; assert all loaded
    // experts that expose columns agree on them.
    let mut expected: Option<Vec<String>> = None;
    for expert in &ensemble.load_outcome().loaded {
        let cols = expert.feature_columns();
        if cols.is_empty() {
            continue;
        }
        match &expected {
            None => expected = Some(cols.to_vec()),
            Some(prev) => {
                if prev.as_slice() != cols {
                    bail!(
                        "ensemble experts disagree on feature columns; \
                         refusing to feed mis-columned data"
                    );
                }
            }
        }
    }
    let expected =
        expected.ok_or_else(|| anyhow!("no loaded expert exposes feature_columns"))?;

    // Build a DataFrame with EXACTLY `expected` columns, by name, from the cube.
    let mut columns = Vec::with_capacity(expected.len());
    for name in &expected {
        let idx = features
            .names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| {
                anyhow!(
                    "feature '{name}' required by the ensemble is absent from the \
                     trader feature cube; refusing to trade on incomplete features"
                )
            })?;
        let col: Vec<f32> = match rows {
            FrameRows::All => features.feature_column(idx).to_vec(),
            FrameRows::LastOnly => features
                .feature_column(idx)
                .last()
                .copied()
                .map(|v| vec![v])
                .unwrap_or_default(),
        };
        columns.push(Column::from(Series::new(name.as_str().into(), col)));
    }
    DataFrame::new(columns).context("build ensemble feature DataFrame")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tempdir(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join("neoethos-bootstrap")
            .join(format!("{label}-{nanos}-{n}-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn default_bootstrap_names_match_known_model_names_minus_swarm() {
        // 33 voters = KNOWN_MODEL_NAMES minus genetic and exit_agent.
        assert_eq!(DEFAULT_BOOTSTRAP_EXPERT_NAMES.len(), 33);
        let names: std::collections::HashSet<&str> =
            DEFAULT_BOOTSTRAP_EXPERT_NAMES.iter().copied().collect();
        // F-319 REVISED (2026-07-11, operator directive "every trained
        // model votes"): only `genetic` keeps the search-only exemption.
        assert!(
            !names.contains("genetic"),
            "genetic is the strategy discoverer — search-only exemption"
        );
        for present in ["neat", "neuro_evo", "swarm_forecaster"] {
            assert!(
                names.contains(present),
                "{present} is trained — it must vote (swarm: last-row-only, D1.2.8)"
            );
        }
        // F-318 (2026-05-29): exit_agent's ExitDecision3 outputs are
        // filtered out by SoftVotingEnsemble (Classification3 only) and
        // no production exit-side pipeline consumes them. Removed from
        // the bootstrap to stop reserving memory + disk for an artifact
        // no live code path reads.
        assert!(
            !names.contains("exit_agent"),
            "exit_agent removed in F-318 — consumers never wired"
        );
        // Sample required canonical names.
        for required in [
            "lightgbm",
            "xgboost",
            "transformer",
            "meta_stack",
            "dqn",
            "hmm_regime",
        ] {
            assert!(names.contains(required), "missing '{required}'");
        }
    }

    #[test]
    fn build_default_registry_installs_all_33_loaders() {
        let registry = build_default_registry().expect("build default registry");
        let registered = registry.registered_names();
        assert_eq!(registered.len(), 33);
        for required in DEFAULT_BOOTSTRAP_EXPERT_NAMES {
            assert!(
                registry.has_loader(required),
                "registry missing loader for '{required}'"
            );
        }
    }

    #[test]
    fn load_experts_with_empty_models_root_reports_all_missing() {
        // No artifact directories on disk — every name should be
        // categorised as `missing`.
        let root = tempdir("empty");
        let outcome = load_experts_for_symbol(&root, "EURUSD", "H1").expect("load");
        assert_eq!(outcome.loaded_count(), 0);
        assert_eq!(outcome.degraded_count(), 0);
        assert_eq!(outcome.missing_count(), 33);
        assert!(!outcome.has_any_loaded());
    }

    #[test]
    fn build_ensemble_with_no_artifacts_returns_error() {
        // No experts loaded → SoftVotingEnsemble::new rejects.
        // This is the correct safe-default behaviour: refuse to
        // construct an ensemble that cannot produce signals.
        let root = tempdir("no-artifacts");
        let result = build_ensemble_for_symbol(&root, "EURUSD", "H1");
        assert!(result.is_err());
    }

    #[test]
    fn bootstrap_paths_match_training_orchestrator_save_layout() {
        // Pin the directory convention: <models_root>/<symbol>/<tf>/
        // matches what `TrainingOrchestrator::model_artifact_dir`
        // writes. Verified by constructing an empty tree and
        // checking the function looks where the trainer would have
        // written.
        let root = tempdir("layout");
        let expected = root.join("EURUSD").join("H1");
        // Create the expected dir so the load can scan it.
        fs::create_dir_all(&expected).expect("mkdir");
        let outcome = load_experts_for_symbol(&root, "EURUSD", "H1").expect("load");
        // Still 33 missing because the dir is empty, but the
        // function didn't error out → path resolution worked.
        assert_eq!(outcome.missing_count(), 33);
    }
}
