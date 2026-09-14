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
use neoethos_data::FeatureFrame;
use neoethos_execution_budget::CpuLease;

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
/// Succeeds even when many experts are missing — it fails only if NO
/// Classification3 voter loaded (the caller should then refuse to start
/// auto-trade). The load outcome is reachable via `ensemble.load_outcome()`.
///
/// The voting config comes from `models.ensemble_voting`
/// (audit #168). There is no second builder that takes a
/// [`SoftVotingEnsembleConfig`] argument: `build_ensemble_for_symbol_with_config`
/// existed, was called by nothing, and was the reason the
/// live ensemble ran on `SoftVotingEnsembleConfig::default()` —
/// all ~33 experts at weight 1.0 — on every install. It is
/// deleted; this is the only way to build the live ensemble,
/// and it reads the operator's file.
pub fn build_ensemble_for_symbol(
    models_root: &Path,
    symbol: &str,
    timeframe: &str,
) -> Result<SoftVotingEnsemble> {
    let input = symbol_model_feature_input(models_root, symbol, timeframe)?;
    let outcome = load_experts_for_symbol(models_root, symbol, timeframe)?;
    if let Some(input) = &input {
        validate_loaded_model_inputs(models_root, symbol, timeframe, input, &outcome)?;
    }
    let ensemble = SoftVotingEnsemble::new(outcome, voting_config_from_settings()?)
        .context("construct SoftVotingEnsemble from load outcome")?;
    match input {
        Some(input) => ensemble.bind_model_feature_input(input),
        None => {
            tracing::warn!(%symbol, %timeframe, models_root = %models_root.display(),
                "legacy model directory has no persisted numerical input contract; callers requiring verified model preprocessing must refuse it");
            Ok(ensemble)
        }
    }
}

fn validate_loaded_model_inputs(
    models_root: &Path,
    symbol: &str,
    timeframe: &str,
    input: &crate::runtime::feature_input::ModelFeatureInputV1,
    outcome: &ExpertLoadOutcome,
) -> Result<()> {
    for expert in &outcome.loaded {
        let name = expert.name();
        anyhow::ensure!(
            Path::new(name).components().count() == 1
                && !name.contains(['/', '\\'])
                && name != "."
                && name != "..",
            "model loader returned an invalid artifact directory name"
        );
        input
            .validate_expert_runtime_artifact(
                &models_root.join(symbol).join(timeframe).join(name),
                name,
                expert.feature_columns(),
            )
            .with_context(|| format!("verify actual model-owned input for expert '{name}'"))?;
    }
    Ok(())
}

fn symbol_model_feature_input(
    models_root: &Path,
    symbol: &str,
    timeframe: &str,
) -> Result<Option<crate::runtime::feature_input::ModelFeatureInputV1>> {
    let path = models_root
        .join(symbol)
        .join(timeframe)
        .join(crate::runtime::feature_input::MODEL_FEATURE_INPUT_FILE_V1);
    // Only an absent file is legacy. Permission errors, malformed contracts,
    // directories and dangling links must never become a guessed raw input.
    match std::fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        other => {
            other.with_context(|| format!("inspect model input {}", path.display()))?;
        }
    }
    let input = crate::runtime::feature_input::ModelFeatureInputV1::read_from_path(&path)?;
    let anchor = input.producer_receipt().validate()?;
    anyhow::ensure!(
        anchor.symbol_name() == symbol && anchor.timeframe().as_str() == timeframe,
        "persisted model input belongs to a different symbol/timeframe than the selected directory"
    );
    Ok(Some(input))
}

/// Reopen an explicitly selected candidate tree through the existing expert
/// adapters, without searching the legacy or latest model directories.
/// Every model installed for this candidate must load; models outside its
/// configured inventory are not required. This proves loading only, not
/// feature/prediction parity, combined OOS validity, or deployment authority.
///
/// Voting uses the explicit job settings. The training identity does not seal
/// this combining policy: a combined-validation caller must bind that policy
/// with its evaluation result before any later promotion can rely on it.
pub fn build_ensemble_for_candidate(
    candidate_root: &Path,
    manifest: &crate::promotion_candidate_training_v1::PromotionCandidateTrainingManifestV1,
    settings: &neoethos_core::Settings,
) -> Result<SoftVotingEnsemble> {
    let handoff = manifest
        .reopen_handoff(candidate_root)
        .context("reopen the selected candidate's exact model tree and training handoff")?;
    handoff
        .validate_against_settings_v1(settings)
        .context("selected candidate does not match the training job settings")?;
    build_ensemble_from_validated_candidate(candidate_root, manifest, settings, &handoff)
}

/// Reopen an explicitly selected installed candidate for inference without
/// requiring the current data/cache paths or worker capacity to equal its
/// original training runtime. Model parameters and inventory still match the
/// sealed training plan, and the same strict tree/input/loader checks are used.
///
/// No generic directory fallback or accelerator substitution is added. Voting
/// uses the explicit settings and must be bound by the caller's saved combined
/// policy. Successful loading is not final-test or deployment permission.
pub fn build_ensemble_for_candidate_inference(
    candidate_root: &Path,
    manifest: &crate::promotion_candidate_training_v1::PromotionCandidateTrainingManifestV1,
    settings: &neoethos_core::Settings,
) -> Result<SoftVotingEnsemble> {
    let handoff = manifest
        .reopen_handoff(candidate_root)
        .context("reopen the selected candidate's exact model tree and training handoff")?;
    handoff
        .validate_inference_settings_v1(settings)
        .context("selected candidate does not match the inference model settings")?;
    build_ensemble_from_validated_candidate(candidate_root, manifest, settings, &handoff)
}

fn build_ensemble_from_validated_candidate(
    candidate_root: &Path,
    manifest: &crate::promotion_candidate_training_v1::PromotionCandidateTrainingManifestV1,
    settings: &neoethos_core::Settings,
    handoff: &crate::promotion_candidate_training_v1::PromotionCandidateTrainingHandoffV1,
) -> Result<SoftVotingEnsemble> {
    let models_root = candidate_root.join(manifest.candidate_relative_dir());
    let model_feature_input =
        crate::runtime::feature_input::load_model_feature_input_for_handoff_v1(
            &models_root,
            handoff,
        )
        .context("load the selected candidate's own fitted model preprocessing")?;
    let symbol = handoff.canonical_series().anchor().identity().symbol_name();
    let mut outcome =
        load_experts_for_symbol(&models_root, symbol, handoff.base_timeframe().as_str())?;
    let expected = manifest
        .model_artifacts()
        .iter()
        .map(|artifact| artifact.model_name())
        .collect::<Vec<_>>();
    bind_candidate_inventory(&mut outcome, &expected)?;
    validate_loaded_model_inputs(
        &models_root,
        symbol,
        handoff.base_timeframe().as_str(),
        &model_feature_input,
        &outcome,
    )?;
    let ensemble =
        SoftVotingEnsemble::new(outcome, voting_config(&settings.models.ensemble_voting)?)
            .context("construct the selected candidate's ensemble")?
            .bind_model_feature_input(model_feature_input)?;
    // Loading must not rewrite the installed tree or accept a persistent
    // change between the pre-load check and the returned in-memory models.
    manifest
        .verify_installed(candidate_root)
        .context("selected candidate tree changed while its experts were loading")?;
    Ok(ensemble)
}

fn bind_candidate_inventory(outcome: &mut ExpertLoadOutcome, expected: &[&str]) -> Result<()> {
    let mut expected_names = expected.to_vec();
    expected_names.sort_unstable();
    let mut loaded_names = outcome.loaded_names();
    loaded_names.sort_unstable();
    anyhow::ensure!(
        !expected_names.is_empty()
            && expected_names.windows(2).all(|names| names[0] != names[1])
            && loaded_names == expected_names
            && outcome
                .missing
                .iter()
                .all(|name| !expected_names.contains(&name.as_str()))
            && outcome.degraded.is_empty(),
        "selected candidate inference inventory is incomplete or different: expected={expected_names:?}, loaded={loaded_names:?}, missing={:?}, degraded={:?}",
        outcome.missing,
        outcome.degraded,
    );
    // The legacy loader probes all registered families. Absent families
    // outside the exact plan are not failed candidate models; retaining them
    // here would falsely report a healthy deliberately small ensemble as partial.
    outcome.missing.clear();
    Ok(())
}

/// Resolve [`SoftVotingEnsembleConfig`] from `models.ensemble_voting`.
///
/// Fail-loud on BOTH arms. A config that cannot be read, or one whose anomaly
/// knees are inverted, must not be silently replaced with the built-in default:
/// that default is a different combining rule, and this one scales live
/// position size. A caller that explicitly requested ML must refuse a failed
/// build, not silently substitute genes-only sizing.
fn voting_config_from_settings() -> Result<SoftVotingEnsembleConfig> {
    let settings: neoethos_core::Settings = neoethos_core::Settings::load()
        .context("read models.ensemble_voting for the live soft-voting ensemble")?;
    voting_config(&settings.models.ensemble_voting)
}

/// The translation itself, on an explicitly typed borrow so the config-recipient
/// scanner can see which fields are read and so this is testable without
/// touching the operator's file.
fn voting_config(
    voting: &neoethos_core::config::EnsembleVotingConfig,
) -> Result<SoftVotingEnsembleConfig> {
    voting
        .validate()
        .map_err(|why| anyhow::anyhow!("{why}"))
        .context("models.ensemble_voting is not a usable configuration")?;
    Ok(SoftVotingEnsembleConfig {
        expert_weights: voting
            .expert_weights
            .iter()
            .map(|(name, weight)| (name.clone(), *weight))
            .collect(),
        excluded_names: voting.excluded_experts.iter().cloned().collect(),
        anomaly_lo: voting.anomaly_lo,
        anomaly_hi: voting.anomaly_hi,
    })
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
/// Each adapter projects its own trained feature set by name from the shared
/// frame, so heterogeneous experts do not need to pretend they were trained on
/// one identical column list. Missing/invalid required features fail closed.
pub fn role_decisions_from_feature_frame(
    models_root: &Path,
    symbol: &str,
    timeframe: &str,
    features: &FeatureFrame,
    lease: &CpuLease,
) -> Result<Vec<super::EnsembleDecision>> {
    let ensemble = build_ensemble_for_symbol(models_root, symbol, timeframe)?;
    ensemble.predict_with_roles(features, lease)
}

/// LIVE-path variant: one role-aware decision for the LAST row of `features`,
/// against an ALREADY-BUILT ensemble.
///
/// The live autopilot builds its ensemble ONCE at engine start (loading ~30
/// expert artifacts takes seconds — far too slow per bar) and calls this on
/// every closed bar with the model producer's own fitted feature frame.
/// Same fail-loud column contract as [`role_decisions_from_feature_frame`];
/// a required ML caller treats any `Err` as an entry abstention, never as
/// authorization for unverified input or full-size genes-only fallback.
///
/// Audit B12: this used to build a ONE-row DataFrame, which starved the
/// swarm forecaster (it refits on the frame's price series and needs
/// history — with 1 row it always abstained live). The experts now get the
/// trailing [`LIVE_DECISION_TAIL_ROWS`] rows and the LAST row's decision is
/// returned; per-row classifiers pay a small batch cost, history-hungry
/// voters actually vote.
pub fn role_decision_for_last_row(
    ensemble: &SoftVotingEnsemble,
    features: &FeatureFrame,
    lease: &CpuLease,
) -> Result<super::EnsembleDecision> {
    let start = features.n_samples().saturating_sub(LIVE_DECISION_TAIL_ROWS);
    let window = features.row_window(start, features.n_samples())?;
    let decisions = ensemble.predict_with_roles(&window, lease)?;
    decisions
        .into_iter()
        .next_back()
        .ok_or_else(|| anyhow::anyhow!("ensemble returned no decision for the last feature row"))
}

/// How much trailing history the live gate feeds the experts per bar.
/// Enough for the swarm forecaster's refit-and-forecast (needs ≥16, wants a
/// few hundred for stable candidate models) while keeping the per-bar batch
/// cost of the row-wise classifiers negligible.
pub const LIVE_DECISION_TAIL_ROWS: usize = 256;

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

    struct InventoryExpert(&'static str);

    impl super::super::ExpertModel for InventoryExpert {
        fn name(&self) -> &str {
            self.0
        }
        fn family(&self) -> crate::runtime::capabilities::ModelFamily {
            crate::runtime::capabilities::ModelFamily::Meta
        }
        fn output_kind(&self) -> super::super::ExpertOutputKind {
            super::super::ExpertOutputKind::Classification3
        }
        fn feature_columns(&self) -> &[String] {
            &[]
        }
        fn predict(
            &self,
            _: &FeatureFrame,
            _: &CpuLease,
        ) -> Result<Vec<super::super::ExpertPrediction>> {
            anyhow::bail!("inventory tests must not call model inference")
        }
    }

    fn inventory(names: &[&'static str]) -> ExpertLoadOutcome {
        ExpertLoadOutcome {
            loaded: names
                .iter()
                .map(|name| Box::new(InventoryExpert(name)) as Box<dyn super::super::ExpertModel>)
                .collect(),
            missing: vec!["unselected-family".into()],
            degraded: vec![],
        }
    }

    #[test]
    fn ordinary_symbol_loader_requires_valid_existing_contract_and_preserves_explicit_legacy_absence()
     {
        use crate::runtime::feature_input::{MODEL_FEATURE_INPUT_FILE_V1, ModelFeatureInputV1};
        use neoethos_data::{FeatureBuildControl, FeatureCellValidity, FeatureColumnF64};
        let raw = std::sync::Arc::new(
            neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
                neoethos_data::test_fixtures::canonical_test_timestamps(12),
                vec![
                    FeatureColumnF64::new(
                        "f1",
                        (0..12).map(|r| r as f64).collect(),
                        vec![FeatureCellValidity::Valid; 12],
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
        );
        let fit = raw
            .fit_normalization(0..6, true, &FeatureBuildControl::default())
            .unwrap();
        let model = raw.with_fitted_normalization(&fit).unwrap();
        let anchor = model.provenance().bindings()[0].dataset_identity();
        let symbol = anchor.symbol_name();
        let tf = anchor.timeframe().as_str();
        let input = ModelFeatureInputV1::from_training_frame(anchor, &model, None, 2).unwrap();
        let root = tempdir("ordinary-input-contract");
        assert!(
            symbol_model_feature_input(&root, symbol, tf)
                .unwrap()
                .is_none()
        );
        let path = root.join(symbol).join(tf).join(MODEL_FEATURE_INPUT_FILE_V1);
        neoethos_core::storage::json::write_bytes_atomic(&path, &input.to_json_bytes().unwrap())
            .unwrap();
        let loaded = symbol_model_feature_input(&root, symbol, tf)
            .unwrap()
            .unwrap();
        let ensemble = SoftVotingEnsemble::new(
            inventory(&["bayes_logit"]),
            SoftVotingEnsembleConfig::default(),
        )
        .unwrap()
        .bind_model_feature_input(loaded)
        .unwrap();
        assert_eq!(ensemble.model_feature_input(), Some(&input));
        let foreign = root
            .join("OTHER")
            .join(tf)
            .join(MODEL_FEATURE_INPUT_FILE_V1);
        neoethos_core::storage::json::write_bytes_atomic(&foreign, &input.to_json_bytes().unwrap())
            .unwrap();
        assert!(symbol_model_feature_input(&root, "OTHER", tf).is_err());
        neoethos_core::storage::json::write_bytes_atomic(&path, b"not a model input").unwrap();
        assert!(
            symbol_model_feature_input(&root, symbol, tf).is_err(),
            "an existing malformed contract cannot become legacy raw behavior"
        );
    }

    #[test]
    fn candidate_inventory_accepts_only_the_exact_configured_set_in_any_load_order() {
        let mut outcome = inventory(&["bayes_logit", "transformer_02", "transformer_01"]);
        bind_candidate_inventory(
            &mut outcome,
            &["transformer_01", "bayes_logit", "transformer_02"],
        )
        .unwrap();
        assert_eq!(outcome.loaded_count(), 3);
        assert!(
            outcome.missing.is_empty(),
            "unselected families are not candidate failures"
        );
    }

    #[test]
    fn candidate_inventory_refuses_missing_extra_and_duplicate_models() {
        for names in [
            vec![],
            vec!["bayes_logit"],
            vec!["bayes_logit", "transformer"],
            vec!["bayes_logit", "bayes_logit"],
            vec!["bayes_logit", "transformer_01", "legacy-extra"],
        ] {
            let mut outcome = inventory(&names);
            assert!(
                bind_candidate_inventory(&mut outcome, &["bayes_logit", "transformer_01"]).is_err()
            );
        }
        assert!(bind_candidate_inventory(&mut inventory(&[]), &[]).is_err());
        assert!(
            bind_candidate_inventory(
                &mut inventory(&["bayes_logit", "bayes_logit"]),
                &["bayes_logit", "bayes_logit"]
            )
            .is_err()
        );
    }

    #[test]
    fn candidate_inventory_never_hides_a_degraded_or_contradictory_load() {
        let mut degraded = inventory(&["bayes_logit"]);
        degraded
            .degraded
            .push(super::super::ExpertLoadError::InvalidArtifact {
                name: "bayes_logit".into(),
                reason: "fixture invalid artifact".into(),
            });
        assert!(bind_candidate_inventory(&mut degraded, &["bayes_logit"]).is_err());
        let mut missing = inventory(&["bayes_logit"]);
        missing.missing.push("bayes_logit".into());
        assert!(bind_candidate_inventory(&mut missing, &["bayes_logit"]).is_err());
    }

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

    fn sidecar_frame(prefix: bool) -> std::sync::Arc<FeatureFrame> {
        use neoethos_data::{FeatureBuildOptions, FeatureCellValidity, FeatureColumnF64};
        let columns = ["quant_close", "f1"]
            .into_iter()
            .enumerate()
            .map(|(col, name)| {
                FeatureColumnF64::new(
                    if prefix {
                        format!("M1_{name}")
                    } else {
                        name.to_owned()
                    },
                    (0..12)
                        .map(|row| 1.0 + (row * (col + 1)) as f64 * 0.01)
                        .collect(),
                    vec![FeatureCellValidity::Valid; 12],
                )
                .unwrap()
            })
            .collect();
        std::sync::Arc::new(
            neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns_with_options(
                neoethos_data::test_fixtures::canonical_test_timestamps(12),
                columns,
                FeatureBuildOptions {
                    prefix_base_features: prefix,
                    normalization_training_rows: Some(0..6),
                    ..Default::default()
                },
            )
            .unwrap(),
        )
    }

    fn write_actual_input_sidecar(dir: &Path, name: &str, frame: FeatureFrame, swarm: bool) {
        use crate::parallel_trainer::{ModelConfig, ModelType, TrainingPayload};
        use crate::runtime::capabilities::{CapabilityState, ModelFamily};
        let anchor = frame.provenance().bindings()[0].dataset_identity();
        let mut profile = crate::runtime::profile::tests::sample_profile();
        profile.model_name = name.to_owned();
        profile.symbol = anchor.symbol_name().to_owned();
        profile.base_timeframe = anchor.timeframe().as_str().to_owned();
        profile.capability_family = if swarm {
            ModelFamily::Forecasting
        } else {
            ModelFamily::Meta
        };
        profile.feature_count = frame.n_features();
        profile.dataset_rows = frame.n_samples();
        profile.row_budget_applied = None;
        let options = frame.feature_build_options().unwrap();
        profile.higher_timeframes = options.higher_tfs.clone();
        profile.multi_resolution_enabled = !options.higher_tfs.is_empty();
        profile.base_features_prefixed = options.prefix_base_features;
        profile.l1_feature_selection_enabled = !swarm;
        profile.requested_backend = Some("cpu".into());
        profile.requested_device = Some("cpu".into());
        profile.planned_backend = Some("cpu".into());
        profile.planned_device = Some("cpu".into());
        profile.planned_precision = Some("fp64".into());
        profile.planned_cpu_threads = Some(1);
        let labels = (0..frame.n_samples())
            .map(|row| (row % 3) as i32 - 1)
            .collect();
        let payload = TrainingPayload::from_frame(frame, labels).unwrap();
        let config = ModelConfig {
            name: name.to_owned(),
            model_type: if swarm {
                ModelType::SwarmForecaster
            } else {
                ModelType::Logistic
            },
            capability_family: profile.capability_family,
            capability_state: CapabilityState::Implemented,
            params: Default::default(),
        };
        crate::runtime::training_artifact::write_model_runtime_artifact_contract_sidecar(
            dir,
            &neoethos_core::Settings::default(),
            &config,
            &payload,
            &profile,
        )
        .unwrap();
    }

    #[test]
    fn registered_logistic_loader_checks_real_training_sidecars_and_each_replica() {
        use crate::base::ExpertModel as TrainingModel;
        use crate::runtime::feature_input::ModelFeatureInputV1;
        use neoethos_execution_budget::{CpuPermitBroker, CpuPermitRequest, WorkerLimit};
        let width = WorkerLimit::new(1).unwrap();
        let lease = CpuPermitBroker::new(width)
            .acquire(CpuPermitRequest::local(width))
            .unwrap();
        let raw = sidecar_frame(false);
        let fit = raw
            .fit_normalization(0..6, false, &Default::default())
            .unwrap();
        let normalized = raw.with_fitted_normalization(&fit).unwrap();
        for frame in [raw.shared_view().unwrap(), normalized] {
            let anchor = frame.provenance().bindings()[0].dataset_identity();
            let input = ModelFeatureInputV1::from_training_frame(anchor, &frame, None, 2).unwrap();
            let symbol = anchor.symbol_name();
            let tf = anchor.timeframe().as_str();
            let root = tempdir("real-model-input");
            let dir = root.join(symbol).join(tf).join("logistic");
            // A real, bounded CPU training/save/load cycle on an L1-style
            // projection. No manufactured model payload or permissive loader.
            assert_eq!(
                crate::statistical::common::statistical_device_policy("logistic"),
                "cpu"
            );
            let selected = frame.select_columns(&[1]).unwrap();
            let mut trained = crate::statistical::linear_impl::LogisticExpert::new();
            trained.epochs = 2;
            let labels: Vec<_> = (0..12).map(|row| (row % 3) as i32 - 1).collect();
            trained.fit(&selected, &labels, &lease).unwrap();
            trained.save(&dir).unwrap();
            write_actual_input_sidecar(&dir, "logistic", selected, false);
            let outcome = load_experts_for_symbol(&root, symbol, tf).unwrap();
            assert_eq!(outcome.loaded_count(), 1, "{:?}", outcome.degraded);
            validate_loaded_model_inputs(&root, symbol, tf, &input, &outcome).unwrap();
            let ensemble = SoftVotingEnsemble::new(outcome, SoftVotingEnsembleConfig::default())
                .unwrap()
                .bind_model_feature_input(input.clone())
                .unwrap();
            assert_eq!(
                ensemble.predict_with_roles(&frame, &lease).unwrap().len(),
                12
            );
            if frame.normalization_fitted_state().is_some() {
                let other_fit = raw
                    .fit_normalization(0..8, false, &Default::default())
                    .unwrap();
                let other_frame = raw.with_fitted_normalization(&other_fit).unwrap();
                let newer = ModelFeatureInputV1::from_training_frame(anchor, &other_frame, None, 2)
                    .unwrap();
                let outcome = load_experts_for_symbol(&root, symbol, tf).unwrap();
                assert!(
                    format!(
                        "{:#}",
                        validate_loaded_model_inputs(&root, symbol, tf, &newer, &outcome)
                            .unwrap_err()
                    )
                    .contains("plan/fit/recipe")
                );
            }
            // The same real registered Logistic loader is used by the generic
            // replica-discovery route. This proves per-replica input binding,
            // not a new voting role for Logistic replica names.
            let replica_root = tempdir("real-model-input-replicas");
            for name in ["logistic_01", "logistic_02"] {
                let replica = replica_root.join(symbol).join(tf).join(name);
                trained.save(&replica).unwrap();
                write_actual_input_sidecar(
                    &replica,
                    name,
                    frame.select_columns(&[1]).unwrap(),
                    false,
                );
            }
            let replicas = load_experts_for_symbol(&replica_root, symbol, tf).unwrap();
            assert_eq!(replicas.loaded_count(), 2, "{:?}", replicas.degraded);
            validate_loaded_model_inputs(&replica_root, symbol, tf, &input, &replicas).unwrap();
            let bad = replica_root.join(symbol).join(tf).join("logistic_02");
            write_actual_input_sidecar(
                &bad,
                "logistic_02",
                frame.select_columns(&[0]).unwrap(),
                false,
            );
            let err = validate_loaded_model_inputs(&replica_root, symbol, tf, &input, &replicas)
                .unwrap_err();
            assert!(format!("{err:#}").contains("logistic_02"));
            assert!(format!("{err:#}").contains("feature order"));
        }
    }

    #[test]
    fn swarm_sidecar_requires_exact_raw_base_price_plan_including_prefix_and_replica() {
        use crate::runtime::feature_input::ModelFeatureInputV1;
        for prefix in [false, true] {
            let raw = sidecar_frame(prefix);
            let fit = raw
                .fit_normalization(0..6, false, &Default::default())
                .unwrap();
            let frame = raw.with_fitted_normalization(&fit).unwrap();
            let anchor = frame.provenance().bindings()[0].dataset_identity();
            let input = ModelFeatureInputV1::from_training_frame(anchor, &frame, None, 2).unwrap();
            let price = vec![input.base_feature_name("quant_close").unwrap()];
            for name in ["swarm_forecaster", "swarm_forecaster_01"] {
                let dir = tempdir("swarm-input-sidecar");
                write_actual_input_sidecar(&dir, name, raw.select_columns(&[0]).unwrap(), true);
                input
                    .validate_expert_runtime_artifact(&dir, name, &price)
                    .unwrap();
                write_actual_input_sidecar(&dir, name, frame.select_columns(&[0]).unwrap(), true);
                assert!(
                    input
                        .validate_expert_runtime_artifact(&dir, name, &price)
                        .is_err(),
                    "a normalized price sidecar cannot masquerade as raw Swarm training"
                );
            }
            let dir = tempdir("swarm-input-invalid-name");
            write_actual_input_sidecar(
                &dir,
                "swarm_forecaster_extra",
                raw.select_columns(&[0]).unwrap(),
                true,
            );
            assert!(
                input
                    .validate_expert_runtime_artifact(&dir, "swarm_forecaster_extra", &price)
                    .is_err()
            );
        }
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

    /// Audit #168. The shipped default must combine exactly as the deleted
    /// `SoftVotingEnsembleConfig::default()` did, or wiring the config would
    /// itself change live sizing on every install that never edits the file.
    #[test]
    fn the_shipped_voting_config_reproduces_the_old_hardcoded_default() {
        let resolved = voting_config(&neoethos_core::config::EnsembleVotingConfig::default())
            .expect("the shipped default must be a valid configuration");
        let old = SoftVotingEnsembleConfig::default();
        assert!(resolved.expert_weights.is_empty());
        assert!(resolved.excluded_names.is_empty());
        assert_eq!(resolved.anomaly_lo, old.anomaly_lo);
        assert_eq!(resolved.anomaly_hi, old.anomaly_hi);
    }

    /// A weight the operator can type must actually reach the aggregator —
    /// that is the whole defect this item names.
    #[test]
    fn an_operator_weight_reaches_the_aggregator() {
        let mut voting = neoethos_core::config::EnsembleVotingConfig::default();
        voting.expert_weights.insert("xgboost".into(), 3.0);
        voting.excluded_experts.push("tide".into());
        let resolved = voting_config(&voting).expect("valid");
        assert_eq!(resolved.expert_weights.get("xgboost"), Some(&3.0_f64));
        assert!(resolved.excluded_names.contains("tide"));
    }

    /// Inverted knees veto every trade. That is a refusal, not a default.
    #[test]
    fn an_inverted_anomaly_band_is_refused_by_name() {
        let mut voting = neoethos_core::config::EnsembleVotingConfig::default();
        voting.anomaly_hi = 0.1;
        let error = voting_config(&voting).expect_err("must refuse");
        assert!(format!("{error:#}").contains("anomaly_hi"), "{error:#}");
    }

    #[test]
    fn a_negative_vote_weight_is_refused_by_name() {
        let mut voting = neoethos_core::config::EnsembleVotingConfig::default();
        voting.expert_weights.insert("lightgbm".into(), -1.0);
        let error = voting_config(&voting).expect_err("must refuse");
        assert!(format!("{error:#}").contains("lightgbm"), "{error:#}");
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
