//! Load real on-disk `.vortex` history and dry-run it through the Phase-1 engine.
//!
//! This is the single helper that makes the offline replay REACHABLE from both
//! front-ends: `neoethos-cli trader-replay` and the app `POST /autonomous/replay`
//! both call [`replay_symbol_from_dir`], so they produce byte-identical
//! [`EngineStats`] from the same data — the UI↔CLI parity mandate, applied to the
//! trader from day one. ZERO broker calls (mock execution), real bars in.

use std::path::Path;

use crate::contracts::{LiveBar, PortfolioEntry, StrategySource, TradeMode};
use crate::decision::DecisionEngine;
use crate::engine::{AutonomousEngine, EngineConfig, EngineStats};
use crate::execution::MockExecutionAdapter;
use crate::portfolio::PortfolioRegistry;
use crate::risk::PermissiveRiskGate;
use crate::signal::MomentumStubSignal;

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

    let mut engine = AutonomousEngine::new(
        registry,
        MomentumStubSignal::default(),
        PermissiveRiskGate,
        MockExecutionAdapter::new(),
        DecisionEngine::default(),
        cfg,
    );

    Ok(crate::replay::replay(&mut engine, &bars))
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
        neoethos_data::prepare_multitimeframe_features(&dataset, &base_tf, &higher_refs, None)?;
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

    // Net the portfolio's genes into one per-bar direction (precomputed once,
    // parity with the GA's batch evaluation).
    let directions =
        crate::gene_signal::combine_gene_signals(&artifact.genes, &aligned, &base_ohlcv);
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
    let mut engine = AutonomousEngine::new(
        registry,
        crate::gene_signal::PrecomputedSignalEngine::new(&symbol, directions),
        PermissiveRiskGate,
        MockExecutionAdapter::new(),
        DecisionEngine::default(),
        cfg,
    );
    Ok(crate::replay::replay(&mut engine, &bars))
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
        neoethos_data::prepare_multitimeframe_features(&dataset, &base_tf, &higher_refs, None)?;
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

    let directions =
        crate::gene_signal::combine_gene_signals(&artifact.genes, &aligned, &base_ohlcv);
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
    let mut engine = AutonomousEngine::new(
        registry,
        signal_engine,
        PermissiveRiskGate,
        MockExecutionAdapter::new(),
        DecisionEngine::default(),
        cfg,
    );
    Ok(crate::replay::replay(&mut engine, &bars))
}
