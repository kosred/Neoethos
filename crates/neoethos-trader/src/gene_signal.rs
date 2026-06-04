//! Phase 4 — evaluate a discovered portfolio (REAL `Gene`s) with backtest parity.
//!
//! Reuses the GA's exact signal functions (`signals_for_gene_full` /
//! `signals_for_gene`) on the discovery feature matrix (rebuilt + projected to
//! `effective_feature_names`), nets the genes per bar into one directional call
//! (design §9 decision 3 — net signed exposure), and serves the precomputed
//! vector as a [`SignalEngine`] (one cursor per symbol). Never re-implements the
//! weighted-sum / threshold / SMC-gate logic ⇒ live signals == backtest signals.

use std::collections::HashMap;

use neoethos_data::{FeatureFrame, Ohlcv};
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
) -> Vec<Direction> {
    let n = aligned_features.n_samples();
    let cfg = EvaluationConfig::default();
    let mut net = vec![0i32; n];
    for gene in genes {
        let sigs = if gene_uses_smc(gene) {
            signals_for_gene_full(aligned_features, base_ohlcv, gene, &cfg)
        } else {
            signals_for_gene(aligned_features, gene)
        };
        for (i, s) in sigs.iter().enumerate() {
            if i < n {
                net[i] += *s as i32;
            }
        }
    }
    net.into_iter().map(dir_from_net).collect()
}

/// A `SignalEngine` that serves a precomputed per-bar direction vector by cursor.
/// The portfolio's signal is computed ONCE over the whole series (parity with the
/// GA's batch evaluation), then handed out one bar at a time. One cursor per
/// symbol — the engine calls `evaluate` once per base-TF bar in chronological
/// order, so `cursor` tracks the bar index.
pub struct PrecomputedSignalEngine {
    per_symbol: HashMap<String, Vec<Direction>>,
    cursors: HashMap<String, usize>,
}

impl PrecomputedSignalEngine {
    pub fn new(symbol: &str, signals: Vec<Direction>) -> Self {
        let mut per_symbol = HashMap::new();
        per_symbol.insert(symbol.to_string(), signals);
        Self {
            per_symbol,
            cursors: HashMap::new(),
        }
    }

    /// Multi-symbol constructor (Phase 6 — a precomputed vector per symbol).
    pub fn from_map(per_symbol: HashMap<String, Vec<Direction>>) -> Self {
        Self {
            per_symbol,
            cursors: HashMap::new(),
        }
    }
}

impl SignalEngine for PrecomputedSignalEngine {
    fn evaluate(&mut self, entry: &PortfolioEntry, _window: &[LiveBar]) -> Signal {
        let cursor = self.cursors.entry(entry.symbol.clone()).or_insert(0);
        let dir = self
            .per_symbol
            .get(&entry.symbol)
            .and_then(|v| v.get(*cursor).copied())
            .unwrap_or(Direction::Flat);
        *cursor += 1;
        // Confidence 1.0 when the net is directional, 0 when flat — the
        // DecisionEngine floors sizing so a flat call simply yields no trade.
        let confidence = if dir == Direction::Flat { 0.0 } else { 1.0 };
        Signal {
            symbol: entry.symbol.clone(),
            dir,
            confidence,
            source: SignalSource::Strategy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_single_gene_matches_ga_signals_exactly() {
        // 4 bars, 2 features; gene reads feature 0 with weight 1.0.
        let data = ndarray::array![
            [1.0_f32, 0.0], // combined 1.0 >= 0.5 → Long
            [-1.0, 0.0],    // -1.0 <= -0.5 → Short
            [0.0, 0.0],     // 0.0 → Flat
            [0.8, 0.0],     // 0.8 >= 0.5 → Long
        ];
        let features = FeatureFrame {
            timestamps: vec![0, 1, 2, 3],
            names: vec!["f0".to_string(), "f1".to_string()],
            data: neoethos_data::FeatureData::InMemory(data),
        };
        let ohlcv = Ohlcv {
            timestamp: Some(vec![0, 1, 2, 3]),
            open: vec![1.0; 4],
            high: vec![1.0; 4],
            low: vec![1.0; 4],
            close: vec![1.0; 4],
            volume: None,
        };
        let mut gene = Gene::default();
        gene.indices = vec![0];
        gene.weights = vec![1.0];
        gene.long_threshold = 0.5;
        gene.short_threshold = -0.5;

        let directions = combine_gene_signals(std::slice::from_ref(&gene), &features, &ohlcv);
        assert_eq!(
            directions,
            vec![Direction::Long, Direction::Short, Direction::Flat, Direction::Long]
        );

        // PARITY: must equal the GA's own signal function mapped to Direction.
        let direct = neoethos_search::signals_for_gene(&features, &gene);
        let mapped: Vec<Direction> = direct
            .iter()
            .map(|s| match s {
                1 => Direction::Long,
                -1 => Direction::Short,
                _ => Direction::Flat,
            })
            .collect();
        assert_eq!(directions, mapped, "combine must match the GA's signals_for_gene");
    }

    #[test]
    fn two_genes_net_to_flat_when_opposed() {
        let data = ndarray::array![[1.0_f32], [1.0]];
        let features = FeatureFrame {
            timestamps: vec![0, 1],
            names: vec!["f0".to_string()],
            data: neoethos_data::FeatureData::InMemory(data),
        };
        let ohlcv = Ohlcv {
            timestamp: Some(vec![0, 1]),
            open: vec![1.0; 2],
            high: vec![1.0; 2],
            low: vec![1.0; 2],
            close: vec![1.0; 2],
            volume: None,
        };
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

        let net = combine_gene_signals(&[long_gene, short_gene], &features, &ohlcv);
        assert_eq!(net, vec![Direction::Flat, Direction::Flat], "opposed genes net to flat");
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
            source: crate::contracts::StrategySource::Gene { id: "x".to_string() },
            mode: crate::contracts::TradeMode::PropFirm,
        };
        assert_eq!(engine.evaluate(&entry, &[]).dir, Direction::Long);
        assert_eq!(engine.evaluate(&entry, &[]).dir, Direction::Flat);
        assert_eq!(engine.evaluate(&entry, &[]).dir, Direction::Short);
        // Past the end → Flat (defensive).
        assert_eq!(engine.evaluate(&entry, &[]).dir, Direction::Flat);
    }
}
