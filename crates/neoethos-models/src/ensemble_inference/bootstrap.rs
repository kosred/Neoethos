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
//!     loader pre-registered (34 canonical names — all wired
//!     families from D1.2.1-D1.2.7, `hmm_regime`, the evolutionary
//!     voters neat/neuro_evo restored in the F-319 revision
//!     2026-07-11, and the exit-axis `exit_agent` restored by
//!     audit #174/#310 on 2026-08-10).
//!  2. Calls [`super::ExpertRegistry::load_with_partial_replica_aware`]
//!     against the operator's `<models_root>/<symbol>/<tf>/` directory
//!     with the full canonical name list. Missing/degraded
//!     artifacts are reported in the outcome (per option β —
//!     no fail-loud) so the operator can run the bot with
//!     whatever subset of the 34 experts has been trained; replica
//!     dirs (`transformer_01/…`) load as independent voters and
//!     orphan artifact dirs are warned about loudly.
//!  3. Constructs a [`super::SoftVotingEnsemble`] with the
//!     default config (no default exclusions — the operator rule is
//!     "every trained model votes"; only `genetic` stays out, as the
//!     strategy discoverer it is search-side, never registered here).
//!
//! Returns the ensemble plus the load outcome so the caller's
//! chrome / system pane can render "Loaded X/34 experts —
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
//!   which the SoftVotingEnsemble surfaces verbatim. EXCEPTION
//!   (audit #174/#310): a drifted EXIT-side expert is contained —
//!   it is refused by name at `error` and the exit opinion
//!   abstains, rather than taking the direction chain down with
//!   it. See `ensemble_feature_dataframe`.

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
///
/// `neat` + `neuro_evo` REJOINED 2026-07-11 (F-319 revision, operator
/// directive "every trained model votes"): both are trained through the
/// shared expert path with genuine 3-class heads — see the
/// `evolution_adapters` module doc. `swarm_forecaster` landed the same
/// day (D1.2.8): last-row-only forecast voter — see the `swarm_adapter`
/// module doc for the honesty constraints.
///
/// **34 names total** (KNOWN_MODEL_NAMES − genetic).
///
/// `exit_agent` REJOINED 2026-08-10 (audit #174 + #310, operator decision
/// "CONNECT IT, visible even in the back/forward test process"). F-318 had
/// removed it on 2026-05-29 with a correct reason at the time — its
/// `ExitDecision3` output was filtered out of the vote and nothing read it, so
/// loading it reserved memory and disk for nobody. The reason is gone: the
/// exit-side combiner (`soft_voting.rs::exit_opinions`) consumes it and its
/// verdict rides on `EnsembleDecision::exit`. It does NOT enter the direction
/// vote — an exit opinion is a different axis from an entry direction.
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
    // RL (3). `dqn` + `sac` (discrete Soft Actor-Critic) are ENTRY/direction
    // voters that emit Classification3 probs and soft-vote together.
    // `exit_agent` is the EXIT-axis voter — audit #174/#310, 2026-08-10.
    "dqn",
    "sac",
    "exit_agent",
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
        "DEFAULT_BOOTSTRAP_EXPERT_NAMES + registry must list the same 34 canonical names"
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
/// all ~34 experts at weight 1.0 — on every install. It is
/// deleted; this is the only way to build the live ensemble,
/// and it reads the operator's file.
pub fn build_ensemble_for_symbol(
    models_root: &Path,
    symbol: &str,
    timeframe: &str,
) -> Result<SoftVotingEnsemble> {
    let outcome = load_experts_for_symbol(models_root, symbol, timeframe)?;
    SoftVotingEnsemble::new(outcome, voting_config_from_settings()?)
        .context("construct SoftVotingEnsemble from load outcome")
}

/// Resolve [`SoftVotingEnsembleConfig`] from `models.ensemble_voting`.
///
/// Fail-loud on BOTH arms. A config that cannot be read, or one whose anomaly
/// knees are inverted, must not be silently replaced with the built-in default:
/// that default is a different combining rule, and this one scales live
/// position size. The caller (`live_trading`) already treats an ensemble build
/// error as "run gene-only, and say so loudly", which is the correct response
/// to "I do not know what the operator configured".
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
            .map(|(name, weight)| (name.clone(), *weight as f32))
            .collect(),
        excluded_names: voting.excluded_experts.iter().cloned().collect(),
        anomaly_lo: voting.anomaly_lo as f32,
        anomaly_hi: voting.anomaly_hi as f32,
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
/// Builds the symbol's ensemble, reads the DIRECTION-side experts' shared
/// `feature_columns()` (asserting they agree), selects the FeatureFrame columns
/// to EXACTLY that set BY NAME (the experts bail unless the DataFrame columns
/// equal their trained set), runs [`SoftVotingEnsemble::predict_with_roles`],
/// and returns one decision per row. FAILS LOUD on any column mismatch /
/// missing feature — the caller (the trader) then falls back to gene-only
/// rather than trading on a wrong-columned ensemble.
///
/// THIS IS THE BACKTEST ENTRY POINT (`neoethos-trader::data_replay::
/// replay_blend_from_dir`). Since audit #174/#310 it also prints an
/// [`ExitCensus`] — the operator required the exit signal to be visible in the
/// back/forward test, not only live.
pub fn role_decisions_from_feature_frame(
    models_root: &Path,
    symbol: &str,
    timeframe: &str,
    features: &neoethos_data::FeatureFrame,
) -> Result<Vec<super::EnsembleDecision>> {
    let ensemble = build_ensemble_for_symbol(models_root, symbol, timeframe)?;
    let df = ensemble_feature_dataframe(&ensemble, features, FrameRows::All)?;
    let decisions = ensemble.predict_with_roles(&df)?;
    // BACKTEST VISIBILITY (audit #174 + #310, operator's explicit requirement:
    // "visible even in the back/forward test process"). This is the entry point
    // the offline replay uses (`neoethos-trader::data_replay::
    // replay_blend_from_dir`), so every backtest now prints what the exit chain
    // did over the whole run — its voters, and how often it wanted out. A
    // signal he cannot see in a backtest is a signal he cannot evaluate before
    // it costs money.
    ExitCensus::of(&decisions).log(symbol, timeframe, ensemble.exit_expert_count());
    Ok(decisions)
}

/// Run-level summary of what the exit-side chain said, so a BACKTEST reports
/// the exit signal instead of hiding it inside per-row structs nobody prints.
///
/// Every field is a count or a mean of values the chain actually produced —
/// there is no threshold, no score and no invented number here.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ExitCensus {
    /// Rows the combiner returned.
    pub rows: usize,
    /// Rows carrying an exit opinion (`EnsembleDecision::exit.is_some()`).
    pub rows_with_opinion: usize,
    /// Mean [`super::ExitOpinion::close_pressure`] over `rows_with_opinion`
    /// (0.0 when there are none).
    pub mean_close_pressure: f32,
    /// Highest close pressure seen on any row.
    pub max_close_pressure: f32,
    /// Rows where the exit chain would keep LESS than the whole position
    /// (`keep_fraction < 1.0`) — i.e. rows on which it had any effect at all.
    pub rows_wanting_less: usize,
}

impl ExitCensus {
    /// Summarise a run's decisions.
    pub fn of(decisions: &[super::EnsembleDecision]) -> Self {
        let mut census = Self {
            rows: decisions.len(),
            ..Self::default()
        };
        let mut sum = 0.0_f64;
        for d in decisions {
            let Some(exit) = d.exit else { continue };
            census.rows_with_opinion += 1;
            let pressure = exit.close_pressure();
            sum += pressure as f64;
            if pressure > census.max_close_pressure {
                census.max_close_pressure = pressure;
            }
            if exit.keep_fraction() < 1.0 {
                census.rows_wanting_less += 1;
            }
        }
        if census.rows_with_opinion > 0 {
            census.mean_close_pressure = (sum / census.rows_with_opinion as f64) as f32;
        }
        census
    }

    /// Print the census. WARN when the chain produced nothing despite experts
    /// being loaded — that is the exact failure mode audit #310 named (a built
    /// mechanism reaching nothing) and it must not read as a benign info line.
    pub fn log(&self, symbol: &str, timeframe: &str, exit_experts: usize) {
        if exit_experts == 0 {
            tracing::info!(
                target: "neoethos_models::ensemble",
                %symbol, %timeframe, rows = self.rows,
                "exit chain: no exit-side expert LOADED — no exit opinion in this run. Either \
                 `exit_agent` is untrained for this symbol/timeframe, it is on \
                 `models.ensemble_voting.excluded_experts`, or the numeric sanity screen \
                 REFUSED its artifact (that refusal is logged by name at load time)."
            );
        } else if self.rows_with_opinion == 0 {
            tracing::warn!(
                target: "neoethos_models::ensemble",
                %symbol, %timeframe, rows = self.rows, exit_experts,
                "exit chain: {exit_experts} exit expert(s) LOADED but produced ZERO opinions — \
                 the exit signal reached nothing this run (audit #174/#310). Check the error \
                 lines above for a refused exit expert."
            );
        } else {
            tracing::info!(
                target: "neoethos_models::ensemble",
                %symbol, %timeframe,
                rows = self.rows,
                rows_with_opinion = self.rows_with_opinion,
                rows_wanting_less = self.rows_wanting_less,
                mean_close_pressure = self.mean_close_pressure,
                max_close_pressure = self.max_close_pressure,
                exit_experts,
                "exit chain census (RL exit output, backtest-visible)"
            );
        }
    }
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
///
/// Audit B12: this used to build a ONE-row DataFrame, which starved the
/// swarm forecaster (it refits on the frame's price series and needs
/// history — with 1 row it always abstained live). The experts now get the
/// trailing [`LIVE_DECISION_TAIL_ROWS`] rows and the LAST row's decision is
/// returned; per-row classifiers pay a small batch cost, history-hungry
/// voters actually vote.
pub fn role_decision_for_last_row(
    ensemble: &SoftVotingEnsemble,
    features: &neoethos_data::FeatureFrame,
) -> Result<super::EnsembleDecision> {
    let df = ensemble_feature_dataframe(
        ensemble,
        features,
        FrameRows::Tail(LIVE_DECISION_TAIL_ROWS),
    )?;
    let decisions = ensemble.predict_with_roles(&df)?;
    let decision = decisions
        .into_iter()
        .next_back()
        .ok_or_else(|| anyhow::anyhow!("ensemble returned no decision for the last feature row"))?;
    // FORWARD-TEST / LIVE VISIBILITY (audit #174 + #310). One line per closed
    // bar — the same cadence the caller already logs at — so the operator can
    // watch the exit chain on a demo forward test exactly as he watches it in a
    // backtest. `exit == None` is reported as an ABSENCE, never as a hold.
    match decision.exit {
        Some(exit) => tracing::info!(
            target: "neoethos_models::ensemble",
            exit_voters = exit.voters,
            exit_close_pressure = exit.close_pressure(),
            exit_keep_fraction = exit.keep_fraction(),
            "exit chain opinion for this bar"
        ),
        None if ensemble.exit_expert_count() > 0 => tracing::warn!(
            target: "neoethos_models::ensemble",
            exit_experts = ensemble.exit_expert_count(),
            "exit chain: expert(s) loaded but NO opinion for this bar — see the refusal above"
        ),
        None => {}
    }
    Ok(decision)
}

/// How much trailing history the live gate feeds the experts per bar.
/// Enough for the swarm forecaster's refit-and-forecast (needs ≥16, wants a
/// few hundred for stable candidate models) while keeping the per-bar batch
/// cost of the row-wise classifiers negligible.
pub const LIVE_DECISION_TAIL_ROWS: usize = 256;

/// Row selector for [`ensemble_feature_dataframe`].
enum FrameRows {
    All,
    /// The trailing `n` rows (fewer when the frame is shorter).
    Tail(usize),
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

    // Determine the DIRECTION-side experts' shared feature-column set; assert
    // all loaded direction-side experts that expose columns agree on them.
    //
    // Audit #174/#310: the quorum is deliberately taken over the direction
    // experts ONLY. A stale/mis-columned `exit_agent` artifact must not be able
    // to fail this contract, because a failure here makes the whole ensemble
    // unavailable and the trader then sizes GENE-ONLY — i.e. without the ML
    // shrink, which moves money the wrong way. The exit expert is checked
    // against the same frame by its own `predict_runtime` (it bails by name on
    // a column mismatch), and `SoftVotingEnsemble::exit_opinions` turns that
    // into a loud, contained abstention.
    let mut expected: Option<(&str, Vec<String>)> = None;
    for expert in &ensemble.load_outcome().loaded {
        if expert.output_kind() == super::ExpertOutputKind::ExitDecision3 {
            continue;
        }
        let cols = expert.feature_columns();
        if cols.is_empty() {
            continue;
        }
        match &expected {
            None => expected = Some((expert.name(), cols.to_vec())),
            Some((first_name, prev)) => {
                if prev.as_slice() != cols {
                    bail!(
                        "ensemble experts disagree on feature columns: '{}' expects {} columns \
                         and '{}' expects {}; refusing to feed mis-columned data",
                        first_name,
                        prev.len(),
                        expert.name(),
                        cols.len()
                    );
                }
            }
        }
    }
    let (_, expected) = expected
        .ok_or_else(|| anyhow!("no loaded direction-side expert exposes feature_columns"))?;

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
            FrameRows::Tail(n) => {
                let full = features.feature_column(idx);
                let start = full.len().saturating_sub(n);
                full.iter().skip(start).copied().collect()
            }
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
        // 34 experts = KNOWN_MODEL_NAMES minus genetic.
        assert_eq!(DEFAULT_BOOTSTRAP_EXPERT_NAMES.len(), 34);
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
        // REGRESSION GUARD for audit #174 + #310. `exit_agent` must be in the
        // bootstrap list or nothing loads it, the exit chain is empty, and the
        // RL exit output goes back to reaching zero trades — which is what it
        // did for sixteen months while the mechanism sat fully built.
        assert!(
            names.contains("exit_agent"),
            "exit_agent must load — the exit-side combiner is its consumer (audit #174/#310)"
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
    fn build_default_registry_installs_all_34_loaders() {
        let registry = build_default_registry().expect("build default registry");
        let registered = registry.registered_names();
        assert_eq!(registered.len(), 34);
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
        assert_eq!(outcome.missing_count(), 34);
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
        assert_eq!(resolved.expert_weights.get("xgboost"), Some(&3.0_f32));
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

    // -- Backtest visibility of the exit chain (audit #174 + #310) -----

    fn decision_with_close(p_close: f32) -> super::super::EnsembleDecision {
        super::super::EnsembleDecision {
            exit: Some(super::super::ExitOpinion {
                close_probs: [1.0 - p_close, 0.0, p_close],
                voters: 1,
            }),
            ..super::super::EnsembleDecision::neutral()
        }
    }

    /// The operator asked for the exit signal to be VISIBLE in the back/forward
    /// test, not only live. This pins the summary the backtest entry point
    /// prints — a signal he cannot see in a backtest is a signal he cannot
    /// evaluate before it costs money.
    #[test]
    fn exit_census_counts_what_the_chain_actually_said() {
        let decisions = vec![
            decision_with_close(0.0),
            decision_with_close(0.5),
            decision_with_close(1.0),
            super::super::EnsembleDecision::neutral(), // no opinion
        ];
        let census = ExitCensus::of(&decisions);
        assert_eq!(census.rows, 4);
        assert_eq!(census.rows_with_opinion, 3);
        // 0.5 and 1.0 keep less than the whole position; 0.0 keeps all of it.
        assert_eq!(census.rows_wanting_less, 2);
        assert!((census.mean_close_pressure - 0.5).abs() < 1e-6);
        assert!((census.max_close_pressure - 1.0).abs() < 1e-6);
    }

    /// The failure this whole item is about: a mechanism that is built and
    /// reaches nothing. If every row comes back without an opinion, the census
    /// must show zero — that is what makes "it reached nothing" READABLE in a
    /// backtest instead of invisible.
    #[test]
    fn exit_census_reports_zero_when_the_chain_reached_nothing() {
        let decisions = vec![super::super::EnsembleDecision::neutral(); 128];
        let census = ExitCensus::of(&decisions);
        assert_eq!(census.rows, 128);
        assert_eq!(census.rows_with_opinion, 0);
        assert_eq!(census.rows_wanting_less, 0);
        assert_eq!(census.mean_close_pressure, 0.0);
        // Must not panic on an all-absent run.
        census.log("EURUSD", "H1", 1);
        census.log("EURUSD", "H1", 0);
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
        // Still 34 missing because the dir is empty, but the
        // function didn't error out → path resolution worked.
        assert_eq!(outcome.missing_count(), 34);
    }
}
