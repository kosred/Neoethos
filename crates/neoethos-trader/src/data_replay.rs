//! Load real on-disk `.vortex` history and dry-run it through the Phase-1 engine.
//!
//! This is the single helper that makes the offline replay REACHABLE from both
//! front-ends: `neoethos-cli trader-replay` and the app `POST /autonomous/replay`
//! call the SAME two entry points — [`replay_symbol_from_dir`] for the momentum
//! STUB and [`replay_portfolio_from_dir`] for the operator's real discovered
//! genes. ZERO broker calls (mock execution), real bars in.
//!
//! **Parity is a property of the ARGUMENTS, not of this module — corrected
//! 2026-08-10 (#229).** The header used to assert flatly that the two
//! front-ends "produce byte-identical `EngineStats`". They produce identical
//! stats when they pass the same [`EngineConfig`], and until today they did
//! not: the CLI passed the operator's balance and broker costs while the app
//! route passed `EngineConfig::default()` — a synthetic balance filling at the
//! mark with zero spread, slippage and commission — and had no portfolio option
//! at all, so the Replay button could only ever run the stub. Both now build
//! the config through [`EngineConfig::try_for_replay_from_settings`], the single
//! adapter, and both can pass a portfolio path.

use std::path::Path;

pub use crate::quote_signal_replay::{
    CanonicalSignalQuoteLaneV1, CanonicalSignalQuoteOutcomeV1,
    replay_canonical_signal_quote_lane_v1, replay_locked_canonical_signal_portfolio_v3,
};

use anyhow::Context;

use crate::contracts::{LiveBar, PortfolioEntry, StrategySource, TradeMode};
use crate::decision::{DecisionConfig, DecisionEngine};
use crate::engine::{AutonomousEngine, DEFAULT_REPLAY_STARTING_BALANCE, EngineConfig, EngineStats};
use crate::execution::MockExecutionAdapter;
use crate::portfolio::PortfolioRegistry;
use crate::risk::PermissiveRiskGate;

fn require_broker_real_historical_replay()
-> anyhow::Result<neoethos_core::BrokerFinancialTruthPermitV1> {
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::HistoricalReplay)
        .map_err(anyhow::Error::new)
}
use crate::signal::MomentumStubSignal;

/// Enumerate every way THIS replay is not the operator's strategy, attach the
/// list to the stats, and shout it into the log (audit #220–#231).
///
/// A diagnostic that gives wrong diagnostics is worse than no diagnostic. Until
/// this list is empty, the numbers below it may not be compared with live
/// results, and the operator should be able to see that without reading the
/// source.
fn disclose(
    mut stats: EngineStats,
    path: &str,
    symbol: &str,
    warnings: Vec<String>,
) -> EngineStats {
    if warnings.is_empty() {
        tracing::info!(
            target: "neoethos_trader::replay",
            replay_path = path,
            symbol = %symbol,
            "replay fidelity: no known stubs on this path"
        );
    } else {
        tracing::warn!(
            target: "neoethos_trader::replay",
            replay_path = path,
            symbol = %symbol,
            stub_count = warnings.len(),
            stubs = ?warnings,
            "REPLAY IS NOT YOUR STRATEGY — the numbers this run reports were produced with \
             the listed stubs / synthetic inputs in the path. Do NOT compare them with live \
             results until this list is empty (audit #220-#231)."
        );
    }
    stats.fidelity_warnings = warnings;
    stats
}

/// Warnings common to every replay path: nothing here depends on which signal
/// engine was used.
fn common_warnings(cfg: &EngineConfig) -> Vec<String> {
    let mut w = Vec::new();
    if cfg.costs.is_zero() {
        w.push(
            "COSTS: zero spread, zero commission, zero slippage — every fill is at the mark. \
             Pass EngineConfig::costs (ReplayCostModel::from_pips) with the operator's real \
             broker costs to remove this."
                .to_string(),
        );
    }
    if (cfg.starting_balance - DEFAULT_REPLAY_STARTING_BALANCE).abs() < f64::EPSILON {
        w.push(format!(
            "BALANCE: synthetic {DEFAULT_REPLAY_STARTING_BALANCE:.0} starting balance, not the \
             operator's account. Every percentage figure below is against that number."
        ));
    }
    w.push(
        "RISK GATE: PermissiveRiskGate — no daily-loss, drawdown, exposure or kill-switch \
         rule is applied. No trade in this run was ever refused for risk."
            .to_string(),
    );
    w.push(
        "EXECUTION: MockExecutionAdapter — simulated fills, no broker, no partial fills, \
         no rejections, no requotes."
            .to_string(),
    );
    // TRAILING (audit #227). Real portfolio/blend replay pins this value from
    // the v4 live-portfolio artifact; standalone stub replay may still receive
    // it from the caller's config. An ARMED run says which exact geometry it
    // used, because "trailing on" is not one behaviour, it is four numbers.
    match &cfg.trailing {
        None => w.push(
            "TRAILING: no break-even move and no trailing stop in this run. If \
             the strategy artifact enables trailing, this run does NOT model the exit the live \
             loop and the GA evaluator apply."
                .to_string(),
        ),
        Some(t) => w.push(format!(
            "TRAILING: armed from the replay's pinned policy — break-even at +{:.2}R, stop trailed at \
             {:.2}x the position's own stop distance, minimum lock {:.1} pips (pip {:.5}). The \
             stop is tested BEFORE the take-profit on every later bar, so this run's payoff is \
             capped the same way the search's is.",
            t.be_trigger_r, t.stop_multiplier, t.min_lock_pips, t.pip_size,
        )),
    }
    w
}

/// Load `(symbol, base_tf)` OHLCV from the data directory and map each bar to a
/// [`LiveBar`]. Bars come back in ascending-timestamp order (the loader
/// normalises that). Errors if the timeframe isn't present on disk.
/// Map a loaded `Ohlcv` (column form) to chronological `LiveBar`s.
pub fn ohlcv_to_livebars(ohlcv: &neoethos_data::Ohlcv, symbol: &str, tf: &str) -> Vec<LiveBar> {
    let n = ohlcv.len();
    let mut bars = Vec::with_capacity(n);
    for i in 0..n {
        bars.push(LiveBar {
            symbol: symbol.to_string(),
            tf: tf.to_string(),
            o: ohlcv.open[i],
            h: ohlcv.high[i],
            l: ohlcv.low[i],
            c: ohlcv.close[i],
            volume: ohlcv.volume.as_ref().map(|v| v[i]).unwrap_or(0.0),
            ts: ohlcv.timestamp.as_ref().map(|v| v[i]).unwrap_or(0),
        });
    }
    bars
}

pub fn load_bars_from_dir(
    data_dir: impl AsRef<Path>,
    symbol: &str,
    base_tf: &str,
) -> anyhow::Result<Vec<LiveBar>> {
    let ohlcv = neoethos_data::load_symbol_timeframe(data_dir, symbol, base_tf)?;
    Ok(ohlcv_to_livebars(&ohlcv, symbol, base_tf))
}

/// Offline dry-run of `(symbol, base_tf)` real history through the Phase-1 engine
/// (momentum stub signal + permissive risk gate + mock execution). Returns the
/// resulting [`EngineStats`].
///
/// **THIS PATH DOES NOT TRADE THE OPERATOR'S STRATEGIES.** It runs a
/// three-bar momentum rule with a synthetic 0.5 %-of-price bracket. Every run
/// returns its own disclaimer in [`EngineStats::fidelity_warnings`] and logs it
/// at `warn` (audit #224/#225/#226/#229). Use
/// [`replay_portfolio_from_dir`] for real genes.
///
/// Phase 1.5 wires only the base timeframe; the higher-TF cube + the real Gene /
/// ensemble signal arrive in Phases 3–4 (the registry entry already carries the
/// `higher_tfs` slot for when they do).
pub fn replay_symbol_from_dir(
    data_dir: impl AsRef<Path>,
    symbol: &str,
    base_tf: &str,
    cfg: EngineConfig,
) -> anyhow::Result<EngineStats> {
    let _broker_truth = require_broker_real_historical_replay()?;
    let bars = load_bars_from_dir(&data_dir, symbol, base_tf)?;
    if bars.is_empty() {
        anyhow::bail!(
            "no bars loaded for {symbol} {base_tf} — is the data folder populated for this pair/timeframe?"
        );
    }

    let registry = PortfolioRegistry::from_entries(vec![PortfolioEntry {
        symbol: symbol.to_string(),
        base_tf: base_tf.to_string(),
        higher_tfs: Vec::new(),
        source: StrategySource::Gene {
            id: format!("{symbol}-{base_tf}-stub"),
        },
        mode: TradeMode::PropFirm,
    }]);

    let mut warnings = common_warnings(&cfg);
    warnings.insert(
        0,
        "SIGNAL: MomentumStubSignal — a 3-bar close-vs-close momentum rule. This is NOT any \
         strategy discovery produced. Nothing about the entries below reflects a gene."
            .to_string(),
    );
    warnings.push(
        "BRACKET: synthetic stop of 0.5 % of price (DecisionConfig::stop_frac). On EURUSD at \
         1.08 that is a ~54-pip stop against a GA population whose stops are 6-20 pips."
            .to_string(),
    );
    warnings.push(
        "EXITS: closes on a Flat signal and on a reversal; the GA evaluator does neither."
            .to_string(),
    );

    let mut engine = AutonomousEngine::new(
        registry,
        MomentumStubSignal::default(),
        PermissiveRiskGate,
        MockExecutionAdapter::with_costs(cfg.costs),
        DecisionEngine::default(),
        cfg,
    );

    let stats = crate::replay::replay(&mut engine, &bars);
    Ok(disclose(stats, "replay_symbol_from_dir", symbol, warnings))
}

/// Phase 4: offline dry-run of a DISCOVERED PORTFOLIO (real genes) over real
/// history. Loads the live portfolio artifact, rebuilds the EXACT multi-TF
/// feature cube discovery used, projects it onto the genes' effective feature
/// set, NETs the genes' per-bar signals (parity with the GA via
/// `signals_for_gene_full`), and replays them through the engine. ZERO broker
/// calls. Fails loud on any feature mismatch rather than trading wrong columns.
pub fn replay_portfolio_from_dir(
    data_dir: impl AsRef<Path>,
    portfolio_path: impl AsRef<Path>,
    cfg: EngineConfig,
) -> anyhow::Result<EngineStats> {
    let broker_truth = require_broker_real_historical_replay()?;
    let artifact = neoethos_search::load_live_portfolio_json(&portfolio_path)?;
    if artifact.genes.is_empty() {
        anyhow::bail!(
            "live portfolio {} has no genes to trade",
            portfolio_path.as_ref().display()
        );
    }
    let data_dir = data_dir.as_ref();
    // Reopen the immutable generations from the v2 artifact itself. A current
    // symbol/timeframe publication is not an acceptable substitute. The loader
    // replays the saved Search fit, when present, rather than fitting this data.
    let exact_input = artifact.load_exact_search_input(data_dir)?;
    let pip_size = broker_truth
        .exact_pip_size_v1(&artifact.symbol)
        .map_err(anyhow::Error::new)?;
    replay_portfolio_features(
        &artifact,
        exact_input.features(),
        exact_input.base_frame().ohlcv(),
        cfg,
        pip_size,
    )
}

// The loaded-frame consumer is shared with bounded offline regressions. The
// public entry point above still requires broker truth before any artifact or
// dataset load; this private seam neither grants nor bypasses that capability.
fn replay_portfolio_features(
    artifact: &neoethos_search::LivePortfolioArtifact,
    features: &neoethos_data::FeatureFrame,
    base_ohlcv: &neoethos_data::Ohlcv,
    cfg: EngineConfig,
    pip_size: f64,
) -> anyhow::Result<EngineStats> {
    let symbol = artifact.symbol.clone();
    let base_tf = artifact.base_tf.clone();
    anyhow::ensure!(
        !artifact.genes.is_empty(),
        "live portfolio has no genes to trade"
    );
    if base_ohlcv.is_empty() {
        anyhow::bail!("no base bars for {symbol} {base_tf}");
    }

    // Matching names alone do not prove matching numeric inputs: check the
    // complete artifact and frozen fit/feature plan before projecting the genes.
    let aligned = artifact.project_live_features(features)?;

    if aligned.n_samples() != base_ohlcv.len() {
        anyhow::bail!(
            "feature/bar length mismatch for {symbol} {base_tf}: {} feature rows vs {} bars — \
             the trader's feature pipeline diverged from discovery's",
            aligned.n_samples(),
            base_ohlcv.len()
        );
    }

    // Net the portfolio's genes into one per-bar direction AND carry each
    // bar's own bracket (audit #226). Until 2026-08-09 this path called
    // `combine_gene_signals`, threw the genes' stops away, and replayed them
    // behind a 0.5 %-of-price synthetic stop.
    let mut cfg = cfg;
    cfg.pin_artifact_exit_policy(artifact.live_trading_policy.exit_policy(), pip_size)
        .with_context(|| {
            format!("live portfolio {symbol} {base_tf} carries an unusable sealed exit policy")
        })?;
    let (directions, sl_pips, tp_pips) = crate::gene_signal::combine_gene_signals_with_brackets(
        &artifact.genes,
        &aligned,
        base_ohlcv,
        pip_size,
    )
    .with_context(|| format!("failed to synthesize portfolio signals for {symbol} {base_tf}"))?;
    let bracketless_bars = directions
        .iter()
        .zip(sl_pips.iter())
        .filter(|(d, sl)| **d != crate::contracts::Direction::Flat && **sl <= 0.0)
        .count();
    let bars = ohlcv_to_livebars(base_ohlcv, &symbol, &base_tf);

    let registry = PortfolioRegistry::from_entries(vec![PortfolioEntry {
        symbol: symbol.clone(),
        base_tf: base_tf.clone(),
        higher_tfs: artifact.higher_tfs.clone(),
        source: StrategySource::Gene {
            id: format!("portfolio:{}-genes", artifact.genes.len()),
        },
        mode: TradeMode::PropFirm,
    }]);

    // Exit parity with the GA evaluator (audit #228): stop, target, time stop —
    // and NOT "the signal went flat" or "the signal reversed", neither of which
    // the evaluator has.
    let eval_defaults = neoethos_search::EvaluationConfig::default();
    if cfg.max_hold_bars.is_none() && eval_defaults.max_hold_bars > 0 {
        cfg.max_hold_bars = Some(eval_defaults.max_hold_bars as u64);
    }
    let mut warnings = common_warnings(&cfg);
    if bracketless_bars > 0 {
        warnings.push(format!(
            "BRACKET: {bracketless_bars} directional bars had NO gene stop, so those entries \
             fell back to the synthetic 0.5 %-of-price bracket. Counted, not dropped."
        ));
    }
    if cfg.max_hold_bars.is_none() {
        warnings.push(
            "EXITS: no max_hold_bars time stop is armed (EvaluationConfig::default is 0), so a \
             position exits only on its stop or target."
                .to_string(),
        );
    }

    let mut engine = AutonomousEngine::new(
        registry,
        crate::gene_signal::PrecomputedSignalEngine::with_brackets(
            &symbol, directions, sl_pips, tp_pips,
        ),
        PermissiveRiskGate,
        MockExecutionAdapter::with_costs(cfg.costs),
        DecisionEngine::new(DecisionConfig::gene_parity(pip_size)),
        cfg,
    );
    let stats = crate::replay::replay(&mut engine, &bars);
    Ok(disclose(
        stats,
        "replay_portfolio_from_dir",
        &symbol,
        warnings,
    ))
}

/// v0.5 ML-integration Stage 3 — offline dry-run of a discovered portfolio with
/// the gene-dominant ML meta-gate blend. Identical gene direction path as
/// [`replay_portfolio_from_dir`]; additionally loads the per-(symbol,base_tf)
/// `SoftVotingEnsemble` from `models_root`, runs the role-aware combiner over
/// the SAME feature cube, and gates the gene size via [`crate::blend_signal`].
///
/// Reachable from BOTH front-ends (CLI `trader-replay --blend …`, app
/// `/autonomous/replay`) so they produce identical [`EngineStats`] — the parity
/// mandate. Ensemble load/feature-contract errors and row-count mismatches
/// return an error, never a different genes-only strategy labelled as ML.
/// `blend.mode == GenesOnly` explicitly skips the ensemble entirely.
#[cfg(feature = "ml-blend")]
pub fn replay_blend_from_dir(
    data_dir: impl AsRef<Path>,
    portfolio_path: impl AsRef<Path>,
    models_root: impl AsRef<Path>,
    cfg: EngineConfig,
    blend: crate::blend_signal::BlendConfig,
) -> anyhow::Result<EngineStats> {
    let broker_truth = require_broker_real_historical_replay()?;
    use crate::blend_signal::BlendMode;

    let artifact = neoethos_search::load_live_portfolio_json(&portfolio_path)?;
    if artifact.genes.is_empty() {
        anyhow::bail!(
            "live portfolio {} has no genes to trade",
            portfolio_path.as_ref().display()
        );
    }
    require_replay_model_input(&artifact.symbol, artifact.normalize_features, blend.mode)?;

    let data_dir = data_dir.as_ref();
    let symbol = artifact.symbol.clone();
    let base_tf = artifact.base_tf.clone();

    let exact_input = artifact.load_exact_search_input(data_dir)?;
    let base_ohlcv = exact_input.base_frame().ohlcv();
    if base_ohlcv.is_empty() {
        anyhow::bail!("no base bars for {symbol} {base_tf}");
    }

    let aligned = artifact.project_live_features(exact_input.features())?;
    if aligned.n_samples() != base_ohlcv.len() {
        anyhow::bail!(
            "feature/bar length mismatch for {symbol} {base_tf}: {} feature rows vs {} bars",
            aligned.n_samples(),
            base_ohlcv.len()
        );
    }

    // Same bracket correction as the gene-only path (audit #226): the ML gate
    // may shrink or veto SIZE, it never touches the stop.
    let pip_size = broker_truth
        .exact_pip_size_v1(&symbol)
        .map_err(anyhow::Error::new)?;
    let mut cfg = cfg;
    cfg.pin_artifact_exit_policy(artifact.live_trading_policy.exit_policy(), pip_size)
        .with_context(|| {
            format!("live portfolio {symbol} {base_tf} carries an unusable sealed exit policy")
        })?;
    let (directions, sl_pips, tp_pips) = crate::gene_signal::combine_gene_signals_with_brackets(
        &artifact.genes,
        &aligned,
        base_ohlcv,
        pip_size,
    )
    .with_context(|| format!("failed to synthesize blend signals for {symbol} {base_tf}"))?;
    let bracketless_bars = directions
        .iter()
        .zip(sl_pips.iter())
        .filter(|(d, sl)| **d != crate::contracts::Direction::Flat && **sl <= 0.0)
        .count();
    let bars = ohlcv_to_livebars(base_ohlcv, &symbol, &base_tf);

    // Explicit GenesOnly does not load models or acquire inference capacity.
    // A requested ML strategy cannot silently become a genes-only backtest.
    let signal_engine = replay_signal_engine(
        &symbol,
        directions,
        artifact.normalize_features,
        blend,
        || {
            let installed = neoethos_core::execution_budget::installed_process_budget()
                .context("blend replay unavailable before the process CPU budget is installed")?;
            let inference_lease = installed.broker().acquire(
                neoethos_core::execution_budget::CpuPermitRequest::local(
                    installed.resolved().effective_worker_limit,
                ),
            )?;
            neoethos_models::ensemble_inference::bootstrap::role_decisions_from_feature_frame(
                models_root.as_ref(),
                &symbol,
                &base_tf,
                exact_input.features(),
                &inference_lease,
            )
        },
    )?;

    let signal_engine = signal_engine.with_brackets(&symbol, sl_pips, tp_pips);

    let registry = PortfolioRegistry::from_entries(vec![PortfolioEntry {
        symbol: symbol.clone(),
        base_tf: base_tf.clone(),
        higher_tfs: artifact.higher_tfs.clone(),
        source: StrategySource::Blend {
            gene_id: format!("portfolio:{}-genes", artifact.genes.len()),
            ensemble_dir: models_root.as_ref().display().to_string(),
        },
        mode: TradeMode::PropFirm,
    }]);

    let eval_defaults = neoethos_search::EvaluationConfig::default();
    if cfg.max_hold_bars.is_none() && eval_defaults.max_hold_bars > 0 {
        cfg.max_hold_bars = Some(eval_defaults.max_hold_bars as u64);
    }
    let mut warnings = common_warnings(&cfg);
    if bracketless_bars > 0 {
        warnings.push(format!(
            "BRACKET: {bracketless_bars} directional bars had NO gene stop and fell back to the \
             synthetic 0.5 %-of-price bracket. Counted, not dropped."
        ));
    }
    if !matches!(blend.mode, BlendMode::GenesOnly) {
        warnings.push(format!(
            "ML BLEND ACTIVE (mode {:?}, gate_floor {:.2}, veto_below {:.2}) — the loaded \
             ensemble gates position size; invalid model rows veto entries. This replay \
             does not establish that the deployed live model set or sizing policy matches.",
            blend.mode, blend.gate_floor, blend.veto_below
        ));
    }

    let mut engine = AutonomousEngine::new(
        registry,
        signal_engine,
        PermissiveRiskGate,
        MockExecutionAdapter::with_costs(cfg.costs),
        DecisionEngine::new(DecisionConfig::gene_parity(pip_size)),
        cfg,
    );
    let stats = crate::replay::replay(&mut engine, &bars);
    Ok(disclose(stats, "replay_blend_from_dir", &symbol, warnings))
}

#[cfg(feature = "ml-blend")]
fn require_replay_model_input(
    symbol: &str,
    search_normalized: bool,
    mode: crate::blend_signal::BlendMode,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !search_normalized || matches!(mode, crate::blend_signal::BlendMode::GenesOnly),
        "ML replay for {symbol} refused: Search's persisted normalization fit cannot substitute \
         the model-specific training fit; candidate-bound model preprocessing is not yet proven. \
         No genes-only substitution"
    );
    Ok(())
}

#[cfg(feature = "ml-blend")]
fn replay_signal_engine(
    symbol: &str,
    directions: Vec<crate::contracts::Direction>,
    search_normalized: bool,
    blend: crate::blend_signal::BlendConfig,
    infer: impl FnOnce() -> anyhow::Result<Vec<neoethos_models::ensemble_inference::EnsembleDecision>>,
) -> anyhow::Result<crate::blend_signal::BlendedSignalEngine> {
    use crate::blend_signal::{BlendMode, BlendedSignalEngine, MlDecision};

    require_replay_model_input(symbol, search_normalized, blend.mode)?;
    if matches!(blend.mode, BlendMode::GenesOnly) {
        return Ok(BlendedSignalEngine::genes_only(symbol, directions));
    }
    let decisions = infer().with_context(|| {
        format!(
            "ML replay for {symbol} refused: ensemble inference failed; no genes-only substitution"
        )
    })?;
    anyhow::ensure!(
        decisions.len() == directions.len(),
        "ML replay for {symbol} refused: {} model rows vs {} gene rows; no genes-only substitution",
        decisions.len(),
        directions.len()
    );
    let ml = decisions
        .into_iter()
        .map(|decision| {
            // Preserve explicit ineligibility even if a malformed adapter put
            // finite payloads in a row marked invalid. The blend rejects NaN.
            if !decision.validity.is_valid() {
                MlDecision {
                    dir_probs: [f64::NAN; 3],
                    regime_gate: f64::NAN,
                    anomaly_scale: f64::NAN,
                }
            } else {
                MlDecision {
                    dir_probs: decision.dir_probs,
                    regime_gate: decision.regime_gate,
                    anomaly_scale: decision.anomaly_scale,
                }
            }
        })
        .collect();
    Ok(BlendedSignalEngine::new(symbol, directions, ml, blend))
}

#[cfg(test)]
mod portfolio_replay_tests {
    use super::*;
    use neoethos_data::test_fixtures::{
        ctrader_sample_feature_frame, ctrader_sample_ohlcv,
        ctrader_test_feature_frame_with_normalization,
    };
    use neoethos_search::data_selection::{
        CanonicalSearchArtifactScopeV2, CanonicalSearchEvaluatedWindowV1,
        CanonicalSearchInputReceiptV2, CanonicalSearchWindowRoleV1,
    };
    use neoethos_search::live_portfolio::{
        LIVE_PORTFOLIO_SCHEMA_VERSION, LivePortfolioArtifact, LiveSizingEvidenceV1,
        LiveTradingPolicyV1,
    };

    fn fixture_artifact(features: &neoethos_data::FeatureFrame) -> LivePortfolioArtifact {
        const CONFIG_HASH: &str = "fnv64:0123456789abcdef";
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(
            features.provenance().bindings()[0].dataset_identity(),
            features,
        )
        .unwrap();
        let ohlcv = ctrader_sample_ohlcv();
        let timestamps = ohlcv.timestamp.as_ref().unwrap();
        let scope = |role, start: usize, end: usize| {
            CanonicalSearchArtifactScopeV2::new(
                receipt.clone(),
                CanonicalSearchEvaluatedWindowV1::new(
                    role,
                    start as u64,
                    end as u64,
                    timestamps[start],
                    timestamps[end - 1],
                )
                .unwrap(),
            )
            .unwrap()
        };
        // Fixed TEST policy body, in the production identity's field order.
        // It authenticates fixture consistency, not a real search or broker run.
        let policy_body = concat!(
            r#"{"kind":"neoethos.live-trading-policy-identity.v1","schema_version":1,"#,
            r#""source_search_config_hash":"fnv64:0123456789abcdef","#,
            r#""source_resolved_config_hash":"fnv64:fedcba9876543210","#,
            r#""trailing_enabled":false,"trailing_be_trigger_r":1.25,"#,
            r#""trailing_stop_multiplier":0.75,"trailing_min_lock_pips":3.0,"#,
            r#""kill_zones_enabled":false,"baseline_spread_pips":1.5,"session_spread_pips":null}"#,
        );
        let mut policy_json: serde_json::Value = serde_json::from_str(policy_body).unwrap();
        policy_json.as_object_mut().unwrap().remove("kind");
        policy_json["identity_hash"] = serde_json::json!(format!(
            "fnv64:{:016x}",
            neoethos_core::utils::fnv1a64(policy_body.as_bytes())
        ));
        let live_trading_policy: LiveTradingPolicyV1 = serde_json::from_value(policy_json).unwrap();
        live_trading_policy.validate().unwrap();

        let gene = neoethos_search::Gene {
            strategy_id: "frozen-replay-gene".to_owned(),
            indices: vec![0],
            weights: vec![1.0],
            long_threshold: if features.normalization_fitted_state().is_some() {
                0.1
            } else {
                0.00001
            },
            short_threshold: if features.normalization_fitted_state().is_some() {
                -0.1
            } else {
                -0.00001
            },
            sl_pips: 6.0,
            tp_pips: 12.0,
            ..Default::default()
        };
        let forward_test = neoethos_search::validation::ForwardTestValidationArtifactFile::new(
            scope(CanonicalSearchWindowRoleV1::SelectionValidation, 80, 90),
            CONFIG_HASH,
            &gene,
            neoethos_search::validation::ForwardTestSummary {
                bars: 10,
                metrics: neoethos_search::eval::BacktestMetrics::from_metric_array([
                    1.0, 1.0, 100_001.0, 0.01, 0.55, 1.5, 1.0, 0.5, 1.0, 0.8, 0.005,
                ]),
                span_days: 1.0,
            },
        )
        .unwrap();
        let artifact = LivePortfolioArtifact {
            schema_version: LIVE_PORTFOLIO_SCHEMA_VERSION,
            search_scope: scope(CanonicalSearchWindowRoleV1::InSample, 0, 80),
            final_holdout_scope: scope(CanonicalSearchWindowRoleV1::Holdout, 90, 100),
            search_config_hash: CONFIG_HASH.to_owned(),
            live_trading_policy,
            symbol: "EURUSD".to_owned(),
            base_tf: "M1".to_owned(),
            higher_tfs: Vec::new(),
            effective_feature_names: vec!["close_minus_open".to_owned()],
            normalize_features: features.normalization_fitted_state().is_some(),
            cost_band: vec![(
                gene.strategy_id.clone(),
                neoethos_search::discovery::CostBandVerdict::Unmeasured,
            )],
            genes: vec![gene],
            sizing_evidence: vec![LiveSizingEvidenceV1 { forward_test }],
        };
        artifact.validate().unwrap();
        artifact
    }

    #[test]
    fn raw_and_frozen_normalized_portfolios_replay_persisted_inputs_through_the_engine() {
        let raw = ctrader_sample_feature_frame();
        let ohlcv = ctrader_sample_ohlcv();
        let normalized = ctrader_test_feature_frame_with_normalization(&raw, 0..80, None).unwrap();
        let cfg = EngineConfig {
            max_hold_bars: Some(3),
            ..Default::default()
        };
        for original in [&raw, &normalized] {
            let artifact = fixture_artifact(original);
            let encoded = serde_json::to_vec(&artifact).unwrap();
            let decoded: LivePortfolioArtifact = serde_json::from_slice(&encoded).unwrap();
            decoded.validate().unwrap();
            let replayed = match decoded.search_scope.receipt().normalization_fitted_state() {
                Some(saved) => {
                    ctrader_test_feature_frame_with_normalization(&raw, 0..80, Some(saved)).unwrap()
                }
                None => raw.clone(),
            };
            let expected =
                replay_portfolio_features(&artifact, original, &ohlcv, cfg.clone(), 0.0001)
                    .expect("original search numeric inputs reach the real replay loop");
            let actual =
                replay_portfolio_features(&decoded, &replayed, &ohlcv, cfg.clone(), 0.0001)
                    .expect("persisted raw/frozen input reaches the same replay loop");
            assert_eq!(
                actual, expected,
                "serialization/replay must not alter any engine statistic"
            );
            assert_eq!(actual.bars_processed, ohlcv.len());
            assert_eq!(actual.signals_evaluated, ohlcv.len());
            assert!(actual.positions_opened > 0);
            assert!(actual.positions_closed > 0);
            assert!(actual.equity.is_finite());
            assert!(
                actual
                    .fidelity_warnings
                    .iter()
                    .any(|warning| warning.contains("MockExecutionAdapter"))
            );
            if decoded.normalize_features {
                let error = replay_portfolio_features(&decoded, &raw, &ohlcv, cfg.clone(), 0.0001)
                    .expect_err("raw data cannot silently replace the saved normalized features");
                assert!(error.to_string().contains("normalization presence differs"));
                let refitted =
                    ctrader_test_feature_frame_with_normalization(&raw, 0..20, None).unwrap();
                assert!(
                    replay_portfolio_features(&decoded, &refitted, &ohlcv, cfg.clone(), 0.0001)
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn normalized_legacy_portfolio_without_saved_fit_is_rejected_by_the_consumer() {
        let raw = ctrader_sample_feature_frame();
        let mut json = serde_json::to_value(fixture_artifact(&raw)).unwrap();
        json["normalize_features"] = serde_json::json!(true);
        let legacy: LivePortfolioArtifact = serde_json::from_value(json).unwrap();
        let error = replay_portfolio_features(
            &legacy,
            &raw,
            &ctrader_sample_ohlcv(),
            EngineConfig::default(),
            0.0001,
        )
        .expect_err("normalization metadata may not be omitted");
        assert!(error.to_string().contains("without fitted parameters"));
    }
}

#[cfg(all(test, feature = "ml-blend"))]
mod blend_replay_tests {
    use super::*;
    use crate::blend_signal::{BlendConfig, BlendMode};
    use crate::contracts::{Direction, SignalEngine, SignalSource};
    use neoethos_data::FeatureCellValidity;
    use neoethos_models::ensemble_inference::EnsembleDecision;

    fn entry() -> PortfolioEntry {
        PortfolioEntry {
            symbol: "EURUSD".into(),
            base_tf: "M5".into(),
            higher_tfs: Vec::new(),
            source: StrategySource::Gene { id: "test".into() },
            mode: TradeMode::PropFirm,
        }
    }

    fn buy() -> EnsembleDecision {
        EnsembleDecision {
            dir_probs: [0.05, 0.9, 0.05],
            regime_gate: 1.0,
            anomaly_scale: 1.0,
            validity: FeatureCellValidity::Valid,
        }
    }

    #[test]
    fn explicit_genes_only_never_calls_the_inference_loader() {
        let mut engine = replay_signal_engine(
            "EURUSD",
            vec![Direction::Long],
            false,
            BlendConfig::default(),
            || panic!("GenesOnly must not load models or acquire inference capacity"),
        )
        .expect("explicit genes-only");
        let signal = engine.evaluate(&entry(), &[]);
        assert_eq!(signal.dir, Direction::Long);
        assert_eq!(signal.confidence, 1.0);
        assert_eq!(signal.source, SignalSource::Strategy);
    }

    #[test]
    fn normalized_genes_only_works_but_model_modes_require_their_own_input_contract() {
        for normalized in [false, true] {
            for mode in [
                BlendMode::GenesOnly,
                BlendMode::MlConfirm,
                BlendMode::MlScale,
            ] {
                let calls = std::cell::Cell::new(0);
                let result = replay_signal_engine(
                    "EURUSD",
                    vec![Direction::Long],
                    normalized,
                    BlendConfig {
                        mode,
                        ..Default::default()
                    },
                    || {
                        calls.set(calls.get() + 1);
                        Ok(vec![buy()])
                    },
                );
                if normalized && !matches!(mode, BlendMode::GenesOnly) {
                    let error = result.err().expect("Search fit is not the model's own fit");
                    assert!(error.to_string().contains("model-specific training fit"));
                    assert_eq!(calls.get(), 0, "refuse before loading or running a model");
                } else {
                    let mut engine = result.expect("explicit supported replay mode");
                    assert_eq!(engine.evaluate(&entry(), &[]).dir, Direction::Long);
                    assert_eq!(
                        calls.get(),
                        usize::from(!matches!(mode, BlendMode::GenesOnly))
                    );
                }
            }
        }
    }

    #[test]
    fn requested_ml_cannot_become_genes_only_when_inference_fails() {
        for mode in [BlendMode::MlConfirm, BlendMode::MlScale] {
            let error = replay_signal_engine(
                "EURUSD",
                vec![Direction::Long],
                false,
                BlendConfig {
                    mode,
                    ..Default::default()
                },
                || anyhow::bail!("required model artifact missing"),
            )
            .err()
            .expect("inference failure must be returned");
            let detail = format!("{error:#}");
            assert!(detail.contains("required model artifact missing"));
            assert!(detail.contains("no genes-only substitution"));
        }
    }

    #[test]
    fn both_short_and_extra_prediction_tapes_are_refused() {
        for row_count in [0, 2] {
            let error = replay_signal_engine(
                "EURUSD",
                vec![Direction::Long],
                false,
                BlendConfig {
                    mode: BlendMode::MlScale,
                    ..Default::default()
                },
                || Ok(vec![buy(); row_count]),
            )
            .err()
            .expect("one prediction per gene row is required");
            assert!(
                error
                    .to_string()
                    .contains(&format!("{row_count} model rows vs 1 gene rows"))
            );
        }
    }

    #[test]
    fn explicit_invalidity_vetoes_even_if_its_numeric_payload_looks_valid() {
        let mut invalid = buy();
        invalid.validity = FeatureCellValidity::Warmup;
        let mut engine = replay_signal_engine(
            "EURUSD",
            vec![Direction::Long; 3],
            false,
            BlendConfig {
                mode: BlendMode::MlScale,
                ..Default::default()
            },
            || {
                Ok(vec![
                    buy(),
                    invalid,
                    EnsembleDecision::invalid(FeatureCellValidity::ComputeFailure),
                ])
            },
        )
        .expect("invalid rows stay aligned")
        .with_brackets("EURUSD", vec![12.0; 3], vec![24.0; 3]);
        let valid = engine.evaluate(&entry(), &[]);
        assert_eq!(valid.dir, Direction::Long);
        assert_eq!(valid.confidence, 0.9);
        for _ in 0..2 {
            let signal = engine.evaluate(&entry(), &[]);
            assert_eq!(signal.dir, Direction::Flat);
            assert_eq!(signal.confidence, 0.0);
            assert_eq!(signal.source, SignalSource::Blend);
            assert_eq!(signal.sl_pips, 12.0);
            assert_eq!(signal.tp_pips, 24.0);
        }
    }
}
