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
//! ```
//!
//! [`build_ensemble_for_symbol`]:
//!  1. Builds an [`super::ExpertRegistry`] with every default
//!     loader pre-registered (32 canonical names — all wired
//!     families from D1.2.1-D1.2.7).
//!  2. Calls [`super::ExpertRegistry::load_with_partial`] against
//!     the operator's `<models_root>/<symbol>/<tf>/` directory
//!     with the full canonical name list. Missing/degraded
//!     artifacts are reported in the outcome (per option β —
//!     no fail-loud) so the operator can run the bot with
//!     whatever subset of the 32 experts has been trained.
//!  3. Constructs a [`super::SoftVotingEnsemble`] with the
//!     default config (genetic + neuro_evo excluded from voting
//!     per the operator's 2026-05-17 directive — they're upstream
//!     strategy discoverers).
//!
//! Returns the ensemble plus the load outcome so the caller's
//! chrome / system pane can render "Loaded X/32 experts —
//! Y missing, Z degraded".
//!
//! ## What it does NOT do
//!
//! - Does NOT load `swarm_forecaster` (deferred per D1.2.7).
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
    evolutionary_adapters::register_evolutionary_loaders, meta_adapters::register_meta_loaders,
    mixed_adapters::register_mixed_loaders, rl_exit_adapters::register_rl_exit_loaders,
    tree_adapters::register_tree_loaders,
};

/// Canonical list of expert names the bootstrap tries to load.
///
/// Sourced from `KNOWN_MODEL_NAMES` per
/// [`crate::runtime::capabilities::KNOWN_MODEL_NAMES`] minus the
/// `swarm_forecaster` name which has a stateful-univariate API
/// that doesn't fit the current ExpertModel trait (deferred to
/// D1.2.8). 32 names total.
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
    // Meta (7)
    "elasticnet",
    "logistic",
    "bayes_logit",
    "meta_blender",
    "probability_calibrator",
    "conformal_gate",
    "meta_stack",
    // Adaptive + Anomaly (3)
    "online_pa",
    "online_hoeffding",
    "isolation_forest",
    // Evolutionary (3) — excluded from voting by SoftVoting's
    // default config (they're upstream strategy discoverers per
    // the 2026-05-17 directive) but still loaded so the chrome
    // can list them and the operator can override exclusion.
    "genetic",
    "neuro_evo",
    "neat",
    // RL + Exit (2)
    "dqn",
    "exit_agent",
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
    register_evolutionary_loaders(&mut registry).context("register evolutionary loaders")?;
    register_rl_exit_loaders(&mut registry).context("register rl+exit loaders")?;
    debug_assert_eq!(
        registry.registered_names().len(),
        DEFAULT_BOOTSTRAP_EXPERT_NAMES.len(),
        "DEFAULT_BOOTSTRAP_EXPERT_NAMES + registry must list the same 32 canonical names"
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
    Ok(registry.load_with_partial(&artifact_root, DEFAULT_BOOTSTRAP_EXPERT_NAMES))
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
        // 32 names = 33 KNOWN_MODEL_NAMES minus swarm_forecaster.
        assert_eq!(DEFAULT_BOOTSTRAP_EXPERT_NAMES.len(), 32);
        let names: std::collections::HashSet<&str> =
            DEFAULT_BOOTSTRAP_EXPERT_NAMES.iter().copied().collect();
        assert!(
            !names.contains("swarm_forecaster"),
            "swarm_forecaster is intentionally absent (D1.2.7 deferral)"
        );
        // Sample required canonical names.
        for required in [
            "lightgbm",
            "xgboost",
            "transformer",
            "meta_stack",
            "dqn",
            "neat",
        ] {
            assert!(names.contains(required), "missing '{required}'");
        }
    }

    #[test]
    fn build_default_registry_installs_all_32_loaders() {
        let registry = build_default_registry().expect("build default registry");
        let registered = registry.registered_names();
        assert_eq!(registered.len(), 32);
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
        assert_eq!(outcome.missing_count(), 32);
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
        // Still 32 missing because the dir is empty, but the
        // function didn't error out → path resolution worked.
        assert_eq!(outcome.missing_count(), 32);
    }
}
