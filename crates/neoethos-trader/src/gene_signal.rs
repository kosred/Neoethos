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

/// The actual one-position portfolio signal. Every field comes from the same
/// ready, SMC-gated gene votes; confidence and brackets cannot be paired from
/// independently recomputed or differently filtered directions.
#[derive(Debug, Clone, PartialEq)]
pub struct NettedGeneSignals {
    pub directions: Vec<Direction>,
    pub confidences: Vec<f64>,
    pub sl_pips: Vec<f64>,
    pub tp_pips: Vec<f64>,
}

/// Active live/combined-validation signal producer. The caller supplies the
/// artifact's already projected and frozen-fit features. This function reads
/// no settings, ambient SMC override or adaptive-stop default.
pub fn combine_gene_signals_with_archived_policy(
    genes: &[Gene],
    aligned_features: &FeatureFrame,
    base_ohlcv: &Ohlcv,
    policy: &neoethos_search::live_portfolio::LiveTradingPolicyV1,
) -> Result<NettedGeneSignals> {
    combine_gene_signals_with_resolved_policy(
        genes,
        aligned_features,
        base_ohlcv,
        &policy.sealed_evaluation_config()?,
        policy.sealed_smc_gate_disabled()?,
        policy.sealed_adaptive_stops_policy()?,
    )
}

fn combine_gene_signals_with_resolved_policy(
    genes: &[Gene],
    aligned_features: &FeatureFrame,
    base_ohlcv: &Ohlcv,
    evaluation: &EvaluationConfig,
    smc_gate_disabled: bool,
    adaptive_policy: &neoethos_search::stop_target::ResolvedAdaptiveStopsPolicyV1,
) -> Result<NettedGeneSignals> {
    let n = aligned_features.n_samples();
    ensure!(
        base_ohlcv.timestamp.as_deref() == Some(aligned_features.timestamps.as_slice())
            && [
                base_ohlcv.open.len(),
                base_ohlcv.high.len(),
                base_ohlcv.low.len(),
                base_ohlcv.close.len()
            ]
            .iter()
            .all(|length| *length == n),
        "archived netted signals require exact matching feature/OHLC rows and timestamps"
    );
    ensure!(
        evaluation.pip_value.is_finite() && evaluation.pip_value > 0.0,
        "archived signal policy has no finite positive pip size"
    );
    adaptive_policy.validate()?;
    ensure!(
        genes.len() <= i32::MAX as usize,
        "portfolio vote count exceeds its exact accumulator"
    );
    for gene in genes {
        ensure!(
            [gene.sl_pips, gene.tp_pips]
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
                && gene.stop_vol_mult.is_finite()
                && gene.stop_vol_mult >= 0.0
                && gene.long_threshold.is_finite()
                && gene.short_threshold.is_finite()
                && gene.indices.len() == gene.weights.len()
                && gene.weights.iter().all(|weight| weight.is_finite()),
            "gene `{}` has invalid archived signal/bracket inputs",
            gene.strategy_id
        );
    }
    let (signals, confidences) =
        neoethos_search::discovery::locked_holdout_signals_and_confidences_with_policy(
            genes,
            aligned_features,
            base_ohlcv,
            evaluation,
            smc_gate_disabled,
        )
        .context("synthesize archived SMC signal/confidence pairs")?;
    // Search interprets an already-evolved positive multiplier as adaptive.
    // The archived `enabled` flag governed gene generation, not whether replay
    // may silently replace an existing gene's adaptive stops with fixed pips.
    let adaptive_base = if genes.iter().any(|gene| gene.stop_vol_mult > 0.0) {
        match neoethos_search::stop_target::adaptive_base_pips_series_with_settings(
            &base_ohlcv.high,
            &base_ohlcv.low,
            &base_ohlcv.close,
            evaluation.pip_value,
            adaptive_policy.settings(),
        ) {
            Ok(base) => Some(base),
            // Exact Search policy: a whole slice without an estimator window
            // uses fixed pips. Missing cells in an available series do not.
            Err(neoethos_search::stop_target::StopDistanceError::TooShort { .. }) => None,
            Err(error) => return Err(error).context("archived adaptive stop base unavailable"),
        }
    } else {
        None
    };
    let mut out = NettedGeneSignals {
        directions: Vec::with_capacity(n),
        confidences: Vec::with_capacity(n),
        sl_pips: Vec::with_capacity(n),
        tp_pips: Vec::with_capacity(n),
    };
    for row in 0..n {
        let mut net = 0i32;
        let mut sums = [[0.0_f64; 3]; 2];
        let mut counts = [0usize; 2];
        for (index, gene) in genes.iter().enumerate() {
            let signal = signals[index][row];
            let confidence = confidences[index][row];
            ensure!(
                (-1..=1).contains(&signal)
                    && confidence.is_finite()
                    && (0.0..=1.0).contains(&confidence),
                "gene `{}` returned an invalid signal/confidence at row {row}",
                gene.strategy_id
            );
            if signal == 0 {
                continue;
            }
            let Some((stop, target)) = neoethos_search::stop_target::resolve_entry_stop_target_pips(
                gene.sl_pips,
                gene.tp_pips,
                gene.stop_vol_mult,
                adaptive_base
                    .as_ref()
                    .and_then(|base| base.get(row))
                    .copied(),
                adaptive_base.is_some(),
                adaptive_policy.reward_risk_fallback(),
            ) else {
                continue;
            };
            net += i32::from(signal);
            let side = usize::from(signal < 0);
            counts[side] += 1;
            sums[side][0] += confidence;
            sums[side][1] += stop;
            sums[side][2] += target;
            ensure!(
                sums[side].iter().all(|value| value.is_finite()),
                "netted portfolio signal arithmetic overflow at row {row}"
            );
        }
        let direction = dir_from_net(net);
        let values = if direction == Direction::Flat {
            [0.0; 3]
        } else {
            let side = usize::from(direction == Direction::Short);
            let count = counts[side] as f64;
            sums[side].map(|value| value / count)
        };
        out.directions.push(direction);
        out.confidences.push(values[0]);
        out.sl_pips.push(values[1]);
        out.tp_pips.push(values[2]);
    }
    Ok(out)
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
    let adaptive_rr_fallback = neoethos_search::adaptive_stops_rr();
    let gene_sl_tp_at = |gene: &Gene, i: usize| -> Result<Option<(f64, f64)>> {
        if gene.stop_vol_mult > 0.0 {
            let d = adaptive_base
                .as_ref()
                .and_then(|base| base.get(i))
                .copied()
                .context("adaptive stop base is missing an aligned bar")?;
            if !d.is_finite() || d <= 0.0 {
                // Warm-up is an unavailable entry, not a broken whole replay
                // and not permission to emit a bracket-free/fixed-stop order.
                return Ok(None);
            }
            let sl = gene.stop_vol_mult * d;
            let reward_risk = neoethos_search::effective_adaptive_reward_risk(
                gene.sl_pips,
                gene.tp_pips,
                adaptive_rr_fallback,
            );
            let tp = reward_risk * sl;
            return Ok(
                (sl.is_finite() && sl > 0.0 && tp.is_finite() && tp > 0.0).then_some((sl, tp))
            );
        }
        Ok(Some((gene.sl_pips, gene.tp_pips)))
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
            let Some((g_sl, g_tp)) = gene_sl_tp_at(gene, i)? else {
                continue;
            };
            net[i] += *s as i32;
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
        *cursor += 1;
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
    fn adaptive_bracket_warmup_is_flat_then_recovers_without_aborting_replay() {
        let n = 160;
        let features = feature_frame(ndarray::Array2::ones((n, 1)), &["f0"]);
        let mut ohlcv = flat_ohlcv(n);
        ohlcv.high.fill(1.001);
        ohlcv.low.fill(0.999);
        let mut gene = Gene::default();
        gene.indices = vec![0];
        gene.weights = vec![1.0];
        gene.long_threshold = 0.5;
        gene.short_threshold = -0.5;
        gene.sl_pips = 20.0;
        gene.tp_pips = 60.0;
        gene.stop_vol_mult = 0.6694603312742179;
        let (directions, sl, tp) =
            combine_gene_signals_with_brackets(&[gene.clone()], &features, &ohlcv, 0.0001).unwrap();
        assert!(directions[..100].iter().all(|dir| *dir == Direction::Flat));
        assert!(
            sl[..100]
                .iter()
                .chain(&tp[..100])
                .all(|value| *value == 0.0)
        );
        let base = neoethos_search::adaptive_base_pips_series(
            &ohlcv.high,
            &ohlcv.low,
            &ohlcv.close,
            0.0001,
        )
        .unwrap();
        for i in 100..n {
            assert_eq!(directions[i], Direction::Long);
            assert_eq!(sl[i], gene.stop_vol_mult * base[i]);
            assert_eq!(tp[i], 3.0 * sl[i]);
        }

        // A ready fixed-stop gene remains tradable while an adaptive gene warms
        // up; one unavailable component must not disable the whole portfolio.
        let mut fixed = gene.clone();
        fixed.stop_vol_mult = 0.0;
        let (mixed, mixed_sl, _) =
            combine_gene_signals_with_brackets(&[gene, fixed], &features, &ohlcv, 0.0001).unwrap();
        assert!(mixed[..100].iter().all(|dir| *dir == Direction::Long));
        assert!(mixed_sl[..100].iter().all(|value| *value == 20.0));
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

    fn archived_fixture_policy() -> (
        EvaluationConfig,
        neoethos_search::stop_target::ResolvedAdaptiveStopsPolicyV1,
    ) {
        // Explicit numerical signal fixture: no private Search constructor,
        // ambient broker metadata or runtime-policy defaults are involved.
        let evaluation = EvaluationConfig {
            symbol: "EURUSD".into(),
            account_currency: "USD".into(),
            initial_equity: 1_000.0,
            max_hold_bars: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            trailing_min_lock_pips: 0.0,
            pip_value: 0.0001,
            spread_pips: 0.0,
            commission_per_trade: 0.0,
            pip_value_per_lot: 10.0,
            swap_long_pips_per_day: 0.0,
            swap_short_pips_per_day: 0.0,
            pnl_conversion_fee_rate: 0.0,
            kill_zones_enabled: false,
            session_spread_pips: None,
            risk_per_trade_min: 0.01,
            risk_per_trade_max: 0.03,
            high_quality_confidence: 1.0,
            smc_gate_threshold: 1.0,
            smc_weight_ob: 1.0,
            smc_weight_fvg: 1.0,
            smc_weight_liq: 1.0,
            smc_weight_mtf: 1.0,
            smc_weight_premium: 1.0,
            smc_weight_inducement: 1.0,
            smc_weight_bos: 1.0,
            smc_weight_choch: 1.0,
            smc_weight_eqh: 1.0,
            smc_weight_eql: 1.0,
            smc_weight_displacement: 1.0,
            growth_objective: false,
            growth_goal: None,
        };
        let adaptive = neoethos_search::stop_target::ResolvedAdaptiveStopsPolicyV1::checked_new(
            neoethos_search::stop_target::StopTargetSettings {
                vol_estimator: "parkinson".into(),
                vol_window: 2,
                tail_window: 3,
                stop_k_vol: 0.0,
                stop_k_tail: 0.0,
                meta_label_min_dist: 0.002,
                ..Default::default()
            },
            true,
            9.0,
        )
        .unwrap();
        (evaluation, adaptive)
    }

    fn explicit_gene(index: usize, weight: f64, stop: f64, target: f64) -> Gene {
        Gene {
            indices: vec![index],
            weights: vec![weight],
            long_threshold: 1.0,
            short_threshold: -1.0,
            sl_pips: stop,
            tp_pips: target,
            stop_vol_mult: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn archived_net_keeps_only_agreeing_confidence_and_brackets_without_double_weighting() {
        let features = feature_frame(
            ndarray::array![[2.0, -3.0, 4.0], [2.0, 3.0, 0.0], [2.0, -3.0, 0.0]],
            &["a", "b", "c"],
        );
        let genes = vec![
            explicit_gene(0, 1.0, 20.0, 40.0),
            explicit_gene(1, -1.0, 40.0, 120.0),
            explicit_gene(2, -1.0, 80.0, 160.0),
        ];
        let (evaluation, adaptive) = archived_fixture_policy();
        let out = combine_gene_signals_with_resolved_policy(
            &genes,
            &features,
            &flat_ohlcv(3),
            &evaluation,
            false,
            &adaptive,
        )
        .unwrap();
        assert_eq!(
            out.directions,
            [Direction::Long, Direction::Flat, Direction::Long]
        );
        // Long margins 1 and 2 over a threshold gap2 => (.5+1)/2=.75.
        // The opposed third gene affects the vote but not the winning side's risk.
        assert_eq!(out.confidences, [0.75, 0.0, 0.75]);
        assert_eq!(out.sl_pips, [30.0, 0.0, 30.0]);
        assert_eq!(out.tp_pips, [80.0, 0.0, 80.0]);
    }

    #[test]
    fn archived_gate_and_bypass_change_actual_live_signal_without_reloading_defaults() {
        let features = feature_frame(ndarray::array![[2.0, -1.0], [2.0, -1.0]], &["a", "smc_ob"]);
        let mut gene = explicit_gene(0, 1.0, 20.0, 40.0);
        gene.use_ob = true;
        let (evaluation, adaptive) = archived_fixture_policy();
        let gated = combine_gene_signals_with_resolved_policy(
            &[gene.clone()],
            &features,
            &flat_ohlcv(2),
            &evaluation,
            false,
            &adaptive,
        )
        .unwrap();
        assert_eq!(gated.directions, [Direction::Flat; 2]);
        assert_eq!(gated.confidences, [0.0; 2]);
        let bypass = combine_gene_signals_with_resolved_policy(
            &[gene],
            &features,
            &flat_ohlcv(2),
            &evaluation,
            true,
            &adaptive,
        )
        .unwrap();
        assert_eq!(bypass.directions, [Direction::Long; 2]);
        assert_eq!(bypass.confidences, [0.5; 2]);
    }

    #[test]
    fn archived_adaptive_recipe_preserves_gene_rr_and_excludes_unavailable_votes() {
        let features = feature_frame(ndarray::Array2::from_elem((8, 1), 2.0), &["a"]);
        let mut bars = flat_ohlcv(8);
        bars.high.fill(1.001);
        bars.low.fill(0.999);
        let mut adaptive_gene = explicit_gene(0, 1.0, 20.0, 60.0);
        adaptive_gene.stop_vol_mult = 1.5;
        let fixed = explicit_gene(0, 1.0, 10.0, 20.0);
        let genes = [adaptive_gene, fixed];
        let (evaluation, adaptive) = archived_fixture_policy();
        let out = combine_gene_signals_with_resolved_policy(
            &genes,
            &features,
            &bars,
            &evaluation,
            false,
            &adaptive,
        )
        .unwrap();
        // .002 price floor / .0001 pip *1.5 =30pips, gene RR3 =>90;
        // mean with fixed10/20 =>20/55, not fallback RR9.
        assert_eq!(out.sl_pips, [20.0; 8]);
        assert_eq!(out.tp_pips, [55.0; 8]);
        let generation_disabled =
            neoethos_search::stop_target::ResolvedAdaptiveStopsPolicyV1::checked_new(
                adaptive.settings().clone(),
                false,
                adaptive.reward_risk_fallback(),
            )
            .unwrap();
        let preserved = combine_gene_signals_with_resolved_policy(
            &genes,
            &features,
            &bars,
            &evaluation,
            false,
            &generation_disabled,
        )
        .unwrap();
        assert_eq!(
            preserved, out,
            "generation enablement must not reinterpret evolved genes"
        );
        let warming = neoethos_search::stop_target::ResolvedAdaptiveStopsPolicyV1::checked_new(
            neoethos_search::stop_target::StopTargetSettings {
                stop_k_vol: 1.0,
                ..adaptive.settings().clone()
            },
            true,
            9.0,
        )
        .unwrap();
        let out = combine_gene_signals_with_resolved_policy(
            &genes,
            &features,
            &bars,
            &evaluation,
            false,
            &warming,
        )
        .unwrap();
        assert_eq!(out.directions[0], Direction::Long);
        assert_eq!(out.sl_pips[0], 10.0);
        assert_eq!(out.tp_pips[0], 20.0);
        assert_eq!(out.confidences[0], 0.5);
        assert!(out.sl_pips[1] > 10.0);
        // Modifying future closed candles must not change earlier decisions.
        bars.high[6..].fill(2.0);
        let changed = combine_gene_signals_with_resolved_policy(
            &genes,
            &features,
            &bars,
            &evaluation,
            false,
            &warming,
        )
        .unwrap();
        assert_eq!(&out.directions[..6], &changed.directions[..6]);
        assert_eq!(&out.confidences[..6], &changed.confidences[..6]);
        assert_eq!(&out.sl_pips[..6], &changed.sl_pips[..6]);
        assert_eq!(&out.tp_pips[..6], &changed.tp_pips[..6]);
    }

    #[test]
    fn archived_signal_rows_refuse_timestamp_drift_and_overflow_instead_of_tail_guessing() {
        let features = feature_frame(ndarray::Array2::from_elem((2, 1), 2.0), &["a"]);
        let (evaluation, adaptive) = archived_fixture_policy();
        let gene = explicit_gene(0, 1.0, 20.0, 40.0);
        let mut bars = flat_ohlcv(2);
        bars.timestamp.as_mut().unwrap()[1] += 60_000;
        assert!(
            combine_gene_signals_with_resolved_policy(
                &[gene.clone()],
                &features,
                &bars,
                &evaluation,
                false,
                &adaptive
            )
            .is_err()
        );
        let oversized = explicit_gene(0, 1.0, f64::MAX, f64::MAX);
        assert!(
            combine_gene_signals_with_resolved_policy(
                &[oversized.clone(), oversized],
                &features,
                &flat_ohlcv(2),
                &evaluation,
                false,
                &adaptive
            )
            .is_err()
        );
    }
}
