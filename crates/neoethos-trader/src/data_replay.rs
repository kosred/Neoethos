//! Load real on-disk `.vortex` history and dry-run it through the Phase-1 engine.
//!
//! This is the single helper that makes the offline replay REACHABLE from both
//! front-ends: `neoethos-cli trader-replay` and the app `POST /autonomous/replay`
//! both call [`replay_symbol_from_dir`], so they produce byte-identical
//! [`EngineStats`] from the same data — the UI↔CLI parity mandate, applied to the
//! trader from day one. ZERO broker calls (mock execution), real bars in.

use std::path::Path;

use crate::contracts::{LiveBar, PortfolioEntry, StrategySource, TradeMode};
use crate::decision::{DecisionConfig, DecisionEngine};
use crate::engine::{AutonomousEngine, DEFAULT_REPLAY_STARTING_BALANCE, EngineConfig, EngineStats};
use crate::execution::MockExecutionAdapter;
use crate::portfolio::PortfolioRegistry;
use crate::risk::PermissiveRiskGate;
use crate::signal::MomentumStubSignal;

/// Enumerate every way THIS replay is not the operator's strategy, attach the
/// list to the stats, and shout it into the log (audit #220–#231).
///
/// A diagnostic that gives wrong diagnostics is worse than no diagnostic. Until
/// this list is empty, the numbers below it may not be compared with live
/// results, and the operator should be able to see that without reading the
/// source.
fn disclose(mut stats: EngineStats, path: &str, symbol: &str, warnings: Vec<String>) -> EngineStats {
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
    w.push(
        "TRAILING: the replay has no trailing stop or break-even move at all. The live loop \
         and the GA evaluator both have exit geometry this path does not model."
            .to_string(),
    );
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
    let artifact = neoethos_search::load_live_portfolio_json(&portfolio_path)?;
    if artifact.genes.is_empty() {
        anyhow::bail!(
            "live portfolio {} has no genes to trade",
            portfolio_path.as_ref().display()
        );
    }
    // Fail loud: we can only reproduce normalization-OFF discovery for now (the
    // per-column normalization stats aren't persisted yet — design §6.1). Trading
    // on mismatched features would be silently wrong.
    if artifact.normalize_features {
        anyhow::bail!(
            "live portfolio '{}' was produced with feature normalization ON, but the per-column \
             normalization stats are not persisted yet, so the trader cannot reproduce the exact \
             feature values. Re-run discovery with feature normalization OFF (the default), or \
             wait for the manifest-stats follow-up.",
            artifact.symbol
        );
    }

    let data_dir = data_dir.as_ref();
    let symbol = artifact.symbol.clone();
    let base_tf = artifact.base_tf.clone();

    // Base-TF OHLCV: drives the engine loop AND the SMC gates.
    let base_ohlcv = neoethos_data::load_symbol_timeframe(data_dir, &symbol, &base_tf)?;
    if base_ohlcv.is_empty() {
        anyhow::bail!("no base bars for {symbol} {base_tf}");
    }

    // Rebuild the SAME multi-TF feature cube discovery used, then project onto the
    // genes' effective feature set (parity by reusing discovery's exact code).
    let dataset = neoethos_data::load_symbol_dataset(data_dir, &symbol)?;
    let higher_refs: Vec<&str> = artifact.higher_tfs.iter().map(|s| s.as_str()).collect();
    let raw_features =
        neoethos_data::prepare_multitimeframe_features(&dataset, &base_tf, &higher_refs)?;
    let aligned = neoethos_search::project_features_to_effective(
        &raw_features,
        &artifact.effective_feature_names,
    )?;

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
    let pip_size = neoethos_search::default_pip_size(&symbol);
    let (directions, sl_pips, tp_pips) = crate::gene_signal::combine_gene_signals_with_brackets(
        &artifact.genes,
        &aligned,
        &base_ohlcv,
        pip_size,
    );
    let bracketless_bars = directions
        .iter()
        .zip(sl_pips.iter())
        .filter(|(d, sl)| **d != crate::contracts::Direction::Flat && **sl <= 0.0)
        .count();
    let bars = ohlcv_to_livebars(&base_ohlcv, &symbol, &base_tf);

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
    let mut cfg = cfg;
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
    Ok(disclose(stats, "replay_portfolio_from_dir", &symbol, warnings))
}

/// v0.5 ML-integration Stage 3 — offline dry-run of a discovered portfolio with
/// the gene-dominant ML meta-gate blend. Identical gene direction path as
/// [`replay_portfolio_from_dir`]; additionally loads the per-(symbol,base_tf)
/// `SoftVotingEnsemble` from `models_root`, runs the role-aware combiner over
/// the SAME feature cube, and gates the gene size via [`crate::blend_signal`].
///
/// Reachable from BOTH front-ends (CLI `trader-replay --blend …`, app
/// `/autonomous/replay`) so they produce identical [`EngineStats`] — the parity
/// mandate. SAFETY: on ANY ensemble load/feature-contract error, or a
/// row-count mismatch, it falls back to the gene-only path (logged) rather than
/// trading on mis-columned ML. `blend.mode == GenesOnly` skips the ensemble
/// entirely — byte-identical to `replay_portfolio_from_dir`.
#[cfg(feature = "ml-blend")]
pub fn replay_blend_from_dir(
    data_dir: impl AsRef<Path>,
    portfolio_path: impl AsRef<Path>,
    models_root: impl AsRef<Path>,
    cfg: EngineConfig,
    blend: crate::blend_signal::BlendConfig,
) -> anyhow::Result<EngineStats> {
    use crate::blend_signal::{BlendMode, BlendedSignalEngine, MlDecision};

    let artifact = neoethos_search::load_live_portfolio_json(&portfolio_path)?;
    if artifact.genes.is_empty() {
        anyhow::bail!(
            "live portfolio {} has no genes to trade",
            portfolio_path.as_ref().display()
        );
    }
    if artifact.normalize_features {
        anyhow::bail!(
            "live portfolio '{}' was produced with feature normalization ON; the trader cannot \
             reproduce the exact feature values. Re-run discovery with normalization OFF.",
            artifact.symbol
        );
    }

    let data_dir = data_dir.as_ref();
    let symbol = artifact.symbol.clone();
    let base_tf = artifact.base_tf.clone();

    let base_ohlcv = neoethos_data::load_symbol_timeframe(data_dir, &symbol, &base_tf)?;
    if base_ohlcv.is_empty() {
        anyhow::bail!("no base bars for {symbol} {base_tf}");
    }

    let dataset = neoethos_data::load_symbol_dataset(data_dir, &symbol)?;
    let higher_refs: Vec<&str> = artifact.higher_tfs.iter().map(|s| s.as_str()).collect();
    let raw_features =
        neoethos_data::prepare_multitimeframe_features(&dataset, &base_tf, &higher_refs)?;
    let aligned = neoethos_search::project_features_to_effective(
        &raw_features,
        &artifact.effective_feature_names,
    )?;
    if aligned.n_samples() != base_ohlcv.len() {
        anyhow::bail!(
            "feature/bar length mismatch for {symbol} {base_tf}: {} feature rows vs {} bars",
            aligned.n_samples(),
            base_ohlcv.len()
        );
    }

    // Same bracket correction as the gene-only path (audit #226): the ML gate
    // may shrink or veto SIZE, it never touches the stop.
    let pip_size = neoethos_search::default_pip_size(&symbol);
    let (directions, sl_pips, tp_pips) = crate::gene_signal::combine_gene_signals_with_brackets(
        &artifact.genes,
        &aligned,
        &base_ohlcv,
        pip_size,
    );
    let bracketless_bars = directions
        .iter()
        .zip(sl_pips.iter())
        .filter(|(d, sl)| **d != crate::contracts::Direction::Flat && **sl <= 0.0)
        .count();
    let bars = ohlcv_to_livebars(&base_ohlcv, &symbol, &base_tf);

    // Build the ML decisions (skipped entirely in GenesOnly). On ANY error or
    // row mismatch, fall back to the gene-only engine — never trade on
    // mis-columned / partial ML.
    let signal_engine = if matches!(blend.mode, BlendMode::GenesOnly) {
        BlendedSignalEngine::genes_only(&symbol, directions)
    } else {
        match neoethos_models::ensemble_inference::bootstrap::role_decisions_from_feature_frame(
            models_root.as_ref(),
            &symbol,
            &base_tf,
            &raw_features,
        ) {
            Ok(decs) if decs.len() == base_ohlcv.len() => {
                let ml: Vec<MlDecision> = decs
                    .into_iter()
                    .map(|d| MlDecision {
                        dir_probs: d.dir_probs,
                        regime_gate: d.regime_gate,
                        anomaly_scale: d.anomaly_scale,
                    })
                    .collect();
                BlendedSignalEngine::new(&symbol, directions, ml, blend)
            }
            Ok(decs) => {
                tracing::warn!(
                    target: "neoethos_trader::blend",
                    symbol = %symbol,
                    base_tf = %base_tf,
                    ml_rows = decs.len(),
                    bar_rows = base_ohlcv.len(),
                    "ensemble decision row count != bars; falling back to gene-only"
                );
                BlendedSignalEngine::genes_only(&symbol, directions)
            }
            Err(error) => {
                tracing::warn!(
                    target: "neoethos_trader::blend",
                    symbol = %symbol,
                    base_tf = %base_tf,
                    %error,
                    "ensemble load/feature-contract failed; falling back to gene-only"
                );
                BlendedSignalEngine::genes_only(&symbol, directions)
            }
        }
    };

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
    let mut cfg = cfg;
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
            "ML BLEND ACTIVE (mode {:?}, gate_floor {:.2}, veto_below {:.2}) — position size is \
             scaled by the ensemble. The live default is GenesOnly, so this run does not \
             describe the live sizing path.",
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
