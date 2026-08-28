//! Phase 4 — evaluate a discovered portfolio (REAL `Gene`s) with backtest parity.
//!
//! Reuses the GA's exact signal functions (`signals_for_gene_full` /
//! `signals_for_gene`) on the discovery feature matrix (rebuilt + projected to
//! `effective_feature_names`), nets the genes per bar into one directional call
//! (design §9 decision 3 — net signed exposure), and serves the precomputed
//! vector as a [`SignalEngine`] (one cursor per symbol). Never re-implements the
//! weighted-sum / threshold / SMC-gate logic ⇒ live signals == backtest signals.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use neoethos_data::{FeatureFrame, Ohlcv};
use neoethos_search::genetic::signals_and_confidence_for_gene_full;
use neoethos_search::{EvaluationConfig, Gene, signals_for_gene, signals_for_gene_full};

use crate::contracts::{Direction, LiveBar, PortfolioEntry, Signal, SignalEngine, SignalSource};

fn gene_uses_smc(gene: &Gene) -> bool {
    gene.use_ob
        || gene.use_fvg
        || gene.use_liq_sweep
        || gene.mtf_confirmation
        || gene.use_premium_discount
        || gene.use_inducement
        || gene.use_bos
        || gene.use_choch
        || gene.use_eqh
        || gene.use_eql
        || gene.use_displacement
}

fn dir_from_net(v: i32) -> Direction {
    if v > 0 {
        Direction::Long
    } else if v < 0 {
        Direction::Short
    } else {
        Direction::Flat
    }
}

/// Combine a portfolio's genes into ONE net per-bar direction. `aligned_features`
/// MUST already be projected onto the genes' `effective_feature_names` (so the
/// gene `indices` reference the right columns); `base_ohlcv` drives the SMC gates
/// for SMC-tagged genes. Genes with no SMC flags take the fast un-gated path
/// (identical result, skips the SMC recompute).
pub fn combine_gene_signals(
    genes: &[Gene],
    aligned_features: &FeatureFrame,
    base_ohlcv: &Ohlcv,
) -> Result<Vec<Direction>> {
    let n = aligned_features.n_samples();
    let cfg = EvaluationConfig::default();
    let mut net = vec![0i32; n];
    for gene in genes {
        let sigs = if gene_uses_smc(gene) {
            signals_for_gene_full(aligned_features, base_ohlcv, gene, &cfg)
        } else {
            signals_for_gene(aligned_features, gene)
        }
        .with_context(|| {
            format!(
                "failed to synthesize signals for gene `{}`",
                gene.strategy_id
            )
        })?;
        ensure!(
            sigs.len() == n,
            "gene `{}` returned {} signals for {n} feature rows",
            gene.strategy_id,
            sigs.len()
        );
        for (i, s) in sigs.iter().enumerate() {
            net[i] += *s as i32;
        }
    }
    Ok(net.into_iter().map(dir_from_net).collect())
}

/// Like [`combine_gene_signals`] but ALSO returns, per bar, the average
/// stop-loss / take-profit (in pips) of the genes that AGREE with the net
/// direction — so the live engine can place the STRATEGY'S OWN brackets, never
/// an externally-imposed stop. `sl_pips`/`tp_pips` are `0.0` on a bar where no
/// agreeing gene carries a stop (a pure signal-exit strategy ⇒ the live order
/// stays bracket-free, exactly matching the backtest's behaviour).
pub fn combine_gene_signals_with_brackets(
    genes: &[Gene],
    aligned_features: &FeatureFrame,
    base_ohlcv: &Ohlcv,
    pip_size: f64,
) -> Result<(Vec<Direction>, Vec<f64>, Vec<f64>)> {
    let n = aligned_features.n_samples();
    let cfg = EvaluationConfig::default();
    let mut net = vec![0i32; n];
    let mut sl_long = vec![0.0f64; n];
    let mut tp_long = vec![0.0f64; n];
    let mut cnt_long = vec![0u32; n];
    let mut sl_short = vec![0.0f64; n];
    let mut tp_short = vec![0.0f64; n];
    let mut cnt_short = vec![0u32; n];

    // Adaptive stops (backtest↔live parity): when any gene is adaptive, build the
    // SAME open-independent per-bar base vol series the discovery backtest uses,
    // so a promoted adaptive gene places the exact volatility-scaled bracket it
    // was scored on. `stop_vol_mult == 0` genes keep their fixed pips.
    let adaptive_base: Option<Vec<f64>> = if genes.iter().any(|g| g.stop_vol_mult > 0.0) {
        Some(
            neoethos_search::adaptive_base_pips_series(
                &base_ohlcv.high,
                &base_ohlcv.low,
                &base_ohlcv.close,
                pip_size,
            )
            .context("adaptive stop base series unavailable")?,
        )
    } else {
        None
    };
    if let Some(base) = &adaptive_base {
        ensure!(
            base.len() == n,
            "adaptive stop base has {} rows for {n} feature rows",
            base.len()
        );
    }
    let adaptive_rr = neoethos_search::adaptive_stops_rr();
    let gene_sl_tp_at = |gene: &Gene, i: usize| -> Result<(f64, f64)> {
        if gene.stop_vol_mult > 0.0 {
            let d = adaptive_base
                .as_ref()
                .and_then(|base| base.get(i))
                .copied()
                .context("adaptive stop base is missing an aligned bar")?;
            ensure!(
                d.is_finite() && d > 0.0,
                "adaptive stop base is invalid at bar {i}: {d}"
            );
            let sl = gene.stop_vol_mult * d;
            return Ok((sl, adaptive_rr * sl));
        }
        Ok((gene.sl_pips, gene.tp_pips))
    };

    for gene in genes {
        let sigs = if gene_uses_smc(gene) {
            signals_for_gene_full(aligned_features, base_ohlcv, gene, &cfg)
        } else {
            signals_for_gene(aligned_features, gene)
        }
        .with_context(|| {
            format!(
                "failed to synthesize bracket signals for gene `{}`",
                gene.strategy_id
            )
        })?;
        ensure!(
            sigs.len() == n,
            "gene `{}` returned {} bracket signals for {n} feature rows",
            gene.strategy_id,
            sigs.len()
        );
        for (i, s) in sigs.iter().enumerate() {
            net[i] += *s as i32;
            let (g_sl, g_tp) = gene_sl_tp_at(gene, i)?;
            if *s > 0 {
                sl_long[i] += g_sl;
                tp_long[i] += g_tp;
                cnt_long[i] += 1;
            } else if *s < 0 {
                sl_short[i] += g_sl;
                tp_short[i] += g_tp;
                cnt_short[i] += 1;
            }
        }
    }

    let mut dirs = Vec::with_capacity(n);
    let mut sl_out = Vec::with_capacity(n);
    let mut tp_out = Vec::with_capacity(n);
    for i in 0..n {
        let dir = dir_from_net(net[i]);
        let (sl, tp) = match dir {
            Direction::Long if cnt_long[i] > 0 => (
                sl_long[i] / cnt_long[i] as f64,
                tp_long[i] / cnt_long[i] as f64,
            ),
            Direction::Short if cnt_short[i] > 0 => (
                sl_short[i] / cnt_short[i] as f64,
                tp_short[i] / cnt_short[i] as f64,
            ),
            _ => (0.0, 0.0),
        };
        dirs.push(dir);
        sl_out.push(sl);
        tp_out.push(tp);
    }
    Ok((dirs, sl_out, tp_out))
}

/// Like [`combine_gene_signals`] but ALSO returns the netted per-bar gene
/// confidence (Stage 3/4 prerequisite). Uses the GA's
/// `signals_and_confidence_for_gene_full` (the same per-bar confidence the
/// faithful OOS eval consumes) for every gene, then per bar nets the signed
/// signals into a direction and averages the confidence of the genes that AGREE
/// with the net side. A Flat net ⇒ confidence 0.0. This gives the blend (and the
/// netted OOS re-validation) a REAL gene confidence to scale, instead of the
/// 1.0/0.0 placeholder.
pub fn combine_gene_signals_with_confidence(
    genes: &[Gene],
    aligned_features: &FeatureFrame,
    base_ohlcv: &Ohlcv,
) -> Result<(Vec<Direction>, Vec<f64>)> {
    let n = aligned_features.n_samples();
    let cfg = EvaluationConfig::default();
    let mut net = vec![0i32; n];
    // Per-bar accumulators of confidence on each side.
    let mut conf_long = vec![0.0f64; n];
    let mut cnt_long = vec![0u32; n];
    let mut conf_short = vec![0.0f64; n];
    let mut cnt_short = vec![0u32; n];

    for gene in genes {
        let (sigs, confs) =
            signals_and_confidence_for_gene_full(aligned_features, base_ohlcv, gene, &cfg)
                .with_context(|| {
                    format!(
                        "failed to synthesize signals and confidence for gene `{}`",
                        gene.strategy_id
                    )
                })?;
        ensure!(
            sigs.len() == n && confs.len() == n,
            "gene `{}` returned {} signals and {} confidences for {n} feature rows",
            gene.strategy_id,
            sigs.len(),
            confs.len()
        );
        for i in 0..n {
            let s = sigs[i];
            let c = confs[i];
            net[i] += s as i32;
            if s > 0 {
                conf_long[i] += c;
                cnt_long[i] += 1;
            } else if s < 0 {
                conf_short[i] += c;
                cnt_short[i] += 1;
            }
        }
    }

    let mut dirs = Vec::with_capacity(n);
    let mut out_conf = Vec::with_capacity(n);
    for i in 0..n {
        let dir = dir_from_net(net[i]);
        let conf = match dir {
            Direction::Long if cnt_long[i] > 0 => {
                (conf_long[i] / cnt_long[i] as f64).clamp(0.0, 1.0)
            }
            Direction::Short if cnt_short[i] > 0 => {
                (conf_short[i] / cnt_short[i] as f64).clamp(0.0, 1.0)
            }
            _ => 0.0,
        };
        dirs.push(dir);
        out_conf.push(conf);
    }
    Ok((dirs, out_conf))
}

/// A `SignalEngine` that serves a precomputed per-bar direction vector by cursor.
/// The portfolio's signal is computed ONCE over the whole series (parity with the
/// GA's batch evaluation), then handed out one bar at a time. One cursor per
/// symbol — the engine calls `evaluate` once per base-TF bar in chronological
/// order, so `cursor` tracks the bar index.
pub struct PrecomputedSignalEngine {
    per_symbol: HashMap<String, Vec<Direction>>,
    /// Per-bar STRATEGY brackets in pips, aligned 1:1 with `per_symbol`.
    /// Empty ⇒ the engine serves no bracket and the DecisionEngine falls back
    /// to its synthetic stop (audit #226).
    per_symbol_sl: HashMap<String, Vec<f64>>,
    per_symbol_tp: HashMap<String, Vec<f64>>,
    cursors: HashMap<String, usize>,
}

impl PrecomputedSignalEngine {
    pub fn new(symbol: &str, signals: Vec<Direction>) -> Self {
        let mut per_symbol = HashMap::new();
        per_symbol.insert(symbol.to_string(), signals);
        Self {
            per_symbol,
            per_symbol_sl: HashMap::new(),
            per_symbol_tp: HashMap::new(),
            cursors: HashMap::new(),
        }
    }

    /// Serve the genes' OWN per-bar brackets alongside the direction, so the
    /// replay places the stop the gene was SCORED on instead of an arbitrary
    /// fraction of price. `sl_pips`/`tp_pips` come from
    /// [`combine_gene_signals_with_brackets`] and are `0.0` on bars where no
    /// agreeing gene carries a stop — the DecisionEngine treats that as
    /// "no bracket" exactly as the live loop does.
    pub fn with_brackets(
        symbol: &str,
        signals: Vec<Direction>,
        sl_pips: Vec<f64>,
        tp_pips: Vec<f64>,
    ) -> Self {
        let mut engine = Self::new(symbol, signals);
        engine.per_symbol_sl.insert(symbol.to_string(), sl_pips);
        engine.per_symbol_tp.insert(symbol.to_string(), tp_pips);
        engine
    }

    /// Multi-symbol constructor (Phase 6 — a precomputed vector per symbol).
    pub fn from_map(per_symbol: HashMap<String, Vec<Direction>>) -> Self {
        Self {
            per_symbol,
            per_symbol_sl: HashMap::new(),
            per_symbol_tp: HashMap::new(),
            cursors: HashMap::new(),
        }
    }
}

impl SignalEngine for PrecomputedSignalEngine {
    fn evaluate(&mut self, entry: &PortfolioEntry, _window: &[LiveBar]) -> Signal {
        let cursor = self.cursors.entry(entry.symbol.clone()).or_insert(0);
        let cur = *cursor;
        let dir = self
            .per_symbol
            .get(&entry.symbol)
            .and_then(|v| v.get(cur).copied())
            .unwrap_or(Direction::Flat);
        *cursor += 1;
        let sl_pips = self
            .per_symbol_sl
            .get(&entry.symbol)
            .and_then(|v| v.get(cur).copied())
            .unwrap_or(0.0);
        let tp_pips = self
            .per_symbol_tp
            .get(&entry.symbol)
            .and_then(|v| v.get(cur).copied())
            .unwrap_or(0.0);
        // Confidence 1.0 when the net is directional, 0 when flat — the
        // DecisionEngine floors sizing so a flat call simply yields no trade.
        let confidence = if dir == Direction::Flat { 0.0 } else { 1.0 };
        Signal {
            symbol: entry.symbol.clone(),
            dir,
            confidence,
            source: SignalSource::Strategy,
            sl_pips,
            tp_pips,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature_frame(data: ndarray::Array2<f64>, names: &[&str]) -> FeatureFrame {
        let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(data.nrows());
        neoethos_data::test_fixtures::ctrader_test_feature_frame_from_matrix(
            timestamps,
            names.iter().map(|name| (*name).to_string()).collect(),
            data,
        )
        .expect("valid f64 trader test feature frame")
    }

    fn flat_ohlcv(rows: usize) -> Ohlcv {
        Ohlcv {
            timestamp: Some(neoethos_data::test_fixtures::canonical_test_timestamps(
                rows,
            )),
            open: vec![1.0; rows],
            high: vec![1.0; rows],
            low: vec![1.0; rows],
            close: vec![1.0; rows],
            volume: None,
        }
    }

    fn gene_with_invalid_feature_index() -> Gene {
        let mut gene = Gene::default();
        gene.indices = vec![1];
        gene.weights = vec![1.0];
        gene
    }

    fn assert_error_chain_contains(error: &anyhow::Error, expected: &str) {
        assert!(
            error
                .chain()
                .any(|cause| cause.to_string().contains(expected)),
            "expected `{expected}` in error chain: {error:#}"
        );
    }

    #[test]
    fn combine_gene_signals_rejects_invalid_gene_feature_index() {
        let features = feature_frame(ndarray::array![[1.0_f64]], &["f0"]);
        let error = combine_gene_signals(
            &[gene_with_invalid_feature_index()],
            &features,
            &flat_ohlcv(1),
        )
        .expect_err("invalid gene input must fail closed");

        assert_error_chain_contains(&error, "gene feature index 1");
    }

    #[test]
    fn combine_gene_signals_with_brackets_rejects_invalid_gene_feature_index() {
        let features = feature_frame(ndarray::array![[1.0_f64]], &["f0"]);
        let error = combine_gene_signals_with_brackets(
            &[gene_with_invalid_feature_index()],
            &features,
            &flat_ohlcv(1),
            0.0001,
        )
        .expect_err("invalid gene input must fail closed");

        assert_error_chain_contains(&error, "gene feature index 1");
    }

    #[test]
    fn combine_gene_signals_with_confidence_rejects_invalid_gene_feature_index() {
        let features = feature_frame(ndarray::array![[1.0_f64]], &["f0"]);
        let error = combine_gene_signals_with_confidence(
            &[gene_with_invalid_feature_index()],
            &features,
            &flat_ohlcv(1),
        )
        .expect_err("invalid gene input must fail closed");

        assert_error_chain_contains(&error, "gene feature index 1");
    }

    #[test]
    fn adaptive_bracket_base_failure_is_propagated() {
        let features = feature_frame(ndarray::array![[1.0_f64]], &["f0"]);
        let mut gene = Gene::default();
        gene.indices = vec![0];
        gene.weights = vec![1.0];
        gene.stop_vol_mult = 1.0;

        let error = combine_gene_signals_with_brackets(&[gene], &features, &flat_ohlcv(1), 0.0)
            .expect_err("invalid adaptive-stop inputs must fail closed");

        assert!(error.to_string().contains("adaptive stop base"));
    }

    #[test]
    fn combine_single_gene_matches_ga_signals_exactly() {
        // 4 bars, 2 features; gene reads feature 0 with weight 1.0.
        let data = ndarray::array![
            [1.0_f64, 0.0], // combined 1.0 >= 0.5 → Long
            [-1.0, 0.0],    // -1.0 <= -0.5 → Short
            [0.0, 0.0],     // 0.0 → Flat
            [0.8, 0.0],     // 0.8 >= 0.5 → Long
        ];
        let features = feature_frame(data, &["f0", "f1"]);
        let ohlcv = flat_ohlcv(4);
        let mut gene = Gene::default();
        gene.indices = vec![0];
        gene.weights = vec![1.0];
        gene.long_threshold = 0.5;
        gene.short_threshold = -0.5;

        let directions = combine_gene_signals(std::slice::from_ref(&gene), &features, &ohlcv)
            .expect("valid gene signals");
        assert_eq!(
            directions,
            vec![
                Direction::Long,
                Direction::Short,
                Direction::Flat,
                Direction::Long
            ]
        );

        // PARITY: must equal the GA's own signal function mapped to Direction.
        let direct =
            neoethos_search::signals_for_gene(&features, &gene).expect("valid direct GA signals");
        let mapped: Vec<Direction> = direct
            .iter()
            .map(|s| match s {
                1 => Direction::Long,
                -1 => Direction::Short,
                _ => Direction::Flat,
            })
            .collect();
        assert_eq!(
            directions, mapped,
            "combine must match the GA's signals_for_gene"
        );
    }

    #[test]
    fn two_genes_net_to_flat_when_opposed() {
        let data = ndarray::array![[1.0_f64], [1.0]];
        let features = feature_frame(data, &["f0"]);
        let ohlcv = flat_ohlcv(2);
        // Long gene: weight +1, long_thr 0.5 → Long on feat 1.0.
        let mut long_gene = Gene::default();
        long_gene.indices = vec![0];
        long_gene.weights = vec![1.0];
        long_gene.long_threshold = 0.5;
        long_gene.short_threshold = -0.5;
        // Short gene: weight -1 → combined -1.0 <= -0.5 → Short.
        let mut short_gene = Gene::default();
        short_gene.indices = vec![0];
        short_gene.weights = vec![-1.0];
        short_gene.long_threshold = 0.5;
        short_gene.short_threshold = -0.5;

        let net = combine_gene_signals(&[long_gene, short_gene], &features, &ohlcv)
            .expect("valid opposed gene signals");
        assert_eq!(
            net,
            vec![Direction::Flat, Direction::Flat],
            "opposed genes net to flat"
        );
    }

    #[test]
    fn precomputed_engine_serves_by_cursor() {
        let mut engine = PrecomputedSignalEngine::new(
            "EURGBP",
            vec![Direction::Long, Direction::Flat, Direction::Short],
        );
        let entry = PortfolioEntry {
            symbol: "EURGBP".to_string(),
            base_tf: "D1".to_string(),
            higher_tfs: Vec::new(),
            source: crate::contracts::StrategySource::Gene {
                id: "x".to_string(),
            },
            mode: crate::contracts::TradeMode::PropFirm,
        };
        assert_eq!(engine.evaluate(&entry, &[]).dir, Direction::Long);
        assert_eq!(engine.evaluate(&entry, &[]).dir, Direction::Flat);
        assert_eq!(engine.evaluate(&entry, &[]).dir, Direction::Short);
        // Past the end → Flat (defensive).
        assert_eq!(engine.evaluate(&entry, &[]).dir, Direction::Flat);
    }
}
