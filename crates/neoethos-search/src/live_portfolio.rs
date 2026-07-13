//! Self-describing **live portfolio artifact** — the bridge from discovery to the
//! autonomous trader.
//!
//! THE PARITY PROBLEM (verified 2026-06-04): a discovered `Gene`'s `indices`
//! reference columns in the **prefiltered** (and optionally normalized) feature
//! matrix, not raw `compute_hpc_features`. But no single existing artifact
//! persists BOTH the full genes (with SMC flags — only in the checkpoint /
//! portfolio-selection files) AND the `effective_feature_names` that the indices
//! map to (only in the in-memory `DiscoveryResult`, or per-gene in the
//! `GeneExport`). So a trader that loads one artifact alone cannot reproduce the
//! exact feature columns ⇒ silently wrong signals.
//!
//! [`LivePortfolioArtifact`] fixes that: it pairs the full `Vec<Gene>` with the
//! ordered `effective_feature_names`, the `base_tf` / `higher_tfs` the cube was
//! built from, and the `normalize_features` flag in effect — everything the
//! trader needs to rebuild the EXACT matrix the genes were evolved against.
//!
//! Discovery writes it (`save_live_portfolio_json`, called next to
//! `save_portfolio_json`); the trader reads it (`load_live_portfolio_json`) and
//! projects its freshly-computed features onto `effective_feature_names` with
//! [`project_features_to_effective`] (the same by-name selection discovery's
//! forward-test path uses).

use std::path::Path;

use neoethos_data::{FeatureData, FeatureFrame};
use serde::{Deserialize, Serialize};

use crate::Gene;
use crate::discovery::DiscoveryResult;

/// Bumped when the artifact's shape changes incompatibly.
pub const LIVE_PORTFOLIO_SCHEMA_VERSION: u32 = 1;

/// Everything the autonomous trader needs to evaluate a discovered portfolio on
/// fresh data with backtest parity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivePortfolioArtifact {
    pub schema_version: u32,
    pub symbol: String,
    pub base_tf: String,
    pub higher_tfs: Vec<String>,
    /// Feature names AFTER discovery's prefilter, in the exact column order the
    /// gene `indices` reference.
    pub effective_feature_names: Vec<String>,
    /// Whether discovery's feature pipeline normalized features. If `true`, the
    /// trader must apply the same normalization (and today the per-column stats
    /// are NOT persisted, so the trader must recompute them the same way — see
    /// the design §6.1). Default discovery is `false`.
    pub normalize_features: bool,
    /// The promoted portfolio — FULL genes, including SMC flags + SL/TP.
    pub genes: Vec<Gene>,
}

impl LivePortfolioArtifact {
    pub fn from_discovery(
        symbol: &str,
        base_tf: &str,
        higher_tfs: &[String],
        normalize_features: bool,
        result: &DiscoveryResult,
    ) -> Self {
        Self {
            schema_version: LIVE_PORTFOLIO_SCHEMA_VERSION,
            symbol: symbol.to_string(),
            base_tf: base_tf.to_string(),
            higher_tfs: higher_tfs.to_vec(),
            effective_feature_names: result.effective_feature_names.clone(),
            normalize_features,
            genes: oos_surviving_genes(result),
        }
    }
}

/// OOS gate for LIVE trading (audit B02, 2026-07-13): only strategies that
/// made money on the never-seen held-out tail reach the live portfolio.
///
/// The full `DiscoveryResult` (portfolio JSON, quality report, walkforward
/// artifacts) is untouched — the evidence stays on disk for the operator.
/// This gate applies at the ONE artifact the autonomous trader consumes.
/// Matching is by `stable_json_hash(gene)`, the same hash
/// `compute_discovery_forward_test_artifacts` stamps into each artifact's
/// `scope.strategy_hash`, so no positional assumptions are made.
///
/// When the result carries NO forward-test artifacts (legacy caller, tests,
/// or tail computation failed non-fatally) every member is kept and a
/// warning says the live portfolio ships without OOS evidence.
fn oos_surviving_genes(result: &DiscoveryResult) -> Vec<Gene> {
    if result.portfolio.is_empty() {
        return Vec::new();
    }
    if result.forward_test_validation_artifacts.is_empty() {
        tracing::warn!(
            target: "neoethos_search::live_portfolio",
            members = result.portfolio.len(),
            "live portfolio has NO forward-test artifacts — the OOS gate cannot run, \
             all members are kept WITHOUT held-out-tail evidence"
        );
        return result.portfolio.clone();
    }
    let passing: std::collections::HashSet<&str> = result
        .forward_test_validation_artifacts
        .iter()
        .filter(|a| a.summary.metrics.net_profit > 0.0)
        .map(|a| a.scope.strategy_hash.as_str())
        .collect();
    let mut kept = Vec::with_capacity(result.portfolio.len());
    for gene in &result.portfolio {
        match crate::artifact_io::stable_json_hash(gene) {
            Ok(hash) if passing.contains(hash.as_str()) => kept.push(gene.clone()),
            Ok(hash) => tracing::info!(
                target: "neoethos_search::live_portfolio",
                strategy_hash = %hash,
                strategy_id = %gene.strategy_id,
                "OOS gate: dropped from LIVE portfolio — non-positive net profit on the \
                 held-out tail (it remains in the discovery artifacts for inspection)"
            ),
            Err(err) => {
                // Serialization of a Gene basically cannot fail; if it ever
                // does, keeping the member (loudly) beats silently emptying
                // the live portfolio over an infrastructure hiccup.
                tracing::warn!(
                    target: "neoethos_search::live_portfolio",
                    strategy_id = %gene.strategy_id,
                    error = %err,
                    "OOS gate: could not hash gene — keeping it WITHOUT tail evidence"
                );
                kept.push(gene.clone());
            }
        }
    }
    if kept.is_empty() {
        tracing::warn!(
            target: "neoethos_search::live_portfolio",
            candidates = result.portfolio.len(),
            "OOS gate: NO portfolio member made money on the held-out tail — the live \
             portfolio is EMPTY. An honest empty portfolio beats trading overfits."
        );
    } else if kept.len() < result.portfolio.len() {
        tracing::info!(
            target: "neoethos_search::live_portfolio",
            kept = kept.len(),
            dropped = result.portfolio.len() - kept.len(),
            "OOS gate: live portfolio filtered by held-out-tail profitability"
        );
    }
    kept
}

/// Write the live portfolio artifact as pretty JSON. Additive — does NOT touch
/// any existing discovery artifact. Reads the in-effect normalize flag from the
/// data-runtime overrides so the trader knows whether discovery normalized.
pub fn save_live_portfolio_json(
    path: impl AsRef<Path>,
    symbol: &str,
    base_tf: &str,
    higher_tfs: &[String],
    result: &DiscoveryResult,
) -> anyhow::Result<()> {
    let normalize_features = neoethos_data::current_data_runtime_overrides().normalize_features;
    let artifact =
        LivePortfolioArtifact::from_discovery(symbol, base_tf, higher_tfs, normalize_features, result);
    let json = serde_json::to_string_pretty(&artifact).map_err(|e| {
        anyhow::anyhow!("failed to serialize live portfolio artifact: {e}")
    })?;
    std::fs::write(&path, json).map_err(|e| {
        anyhow::anyhow!(
            "failed to write live portfolio artifact to {}: {e}",
            path.as_ref().display()
        )
    })?;
    Ok(())
}

/// Load a live portfolio artifact written by [`save_live_portfolio_json`].
pub fn load_live_portfolio_json(path: impl AsRef<Path>) -> anyhow::Result<LivePortfolioArtifact> {
    let raw = std::fs::read_to_string(&path).map_err(|e| {
        anyhow::anyhow!(
            "live portfolio artifact {} not readable: {e}",
            path.as_ref().display()
        )
    })?;
    let artifact: LivePortfolioArtifact = serde_json::from_str(&raw).map_err(|e| {
        anyhow::anyhow!(
            "live portfolio artifact {} is not valid: {e}",
            path.as_ref().display()
        )
    })?;
    Ok(artifact)
}

/// Project a freshly-computed raw `FeatureFrame` onto `effective_feature_names`
/// (post-prefilter set), in that exact order, so a gene's `indices` reference
/// the right columns. This is the SAME by-name selection the discovery
/// forward-test path uses (`compute_discovery_forward_test_artifacts`).
///
/// Returns `Err` when any effective name is missing from `raw` — that means the
/// trader's feature pipeline diverged from discovery's, and evaluating a gene on
/// it would be meaningless (fail loud rather than trade on wrong columns).
pub fn project_features_to_effective(
    raw: &FeatureFrame,
    effective_feature_names: &[String],
) -> anyhow::Result<FeatureFrame> {
    if raw.names == effective_feature_names {
        return Ok(raw.clone());
    }
    let mut keep_indices = Vec::with_capacity(effective_feature_names.len());
    for name in effective_feature_names {
        let idx = raw
            .names
            .iter()
            .position(|candidate| candidate == name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "live feature set is missing '{}' from the discovery effective feature set; \
                     the trader must compute features with the SAME pipeline + config as the \
                     discovery run that produced this portfolio",
                    name
                )
            })?;
        keep_indices.push(idx);
    }
    let n_rows = raw.n_samples();
    let mut projected = ndarray::Array2::<f32>::zeros((n_rows, keep_indices.len()));
    for (new_idx, &orig_idx) in keep_indices.iter().enumerate() {
        projected
            .column_mut(new_idx)
            .assign(&raw.feature_column(orig_idx));
    }
    Ok(FeatureFrame {
        timestamps: raw.timestamps.clone(),
        names: effective_feature_names.to_vec(),
        data: FeatureData::InMemory(projected),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_round_trips_through_json() {
        let mut gene = Gene::default();
        gene.indices = vec![0, 2];
        gene.weights = vec![0.5, -0.25];
        gene.long_threshold = 0.1;
        gene.short_threshold = -0.1;
        gene.strategy_id = "test-gene".to_string();

        let artifact = LivePortfolioArtifact {
            schema_version: LIVE_PORTFOLIO_SCHEMA_VERSION,
            symbol: "EURGBP".to_string(),
            base_tf: "D1".to_string(),
            higher_tfs: vec!["W1".to_string()],
            effective_feature_names: vec!["rsi".to_string(), "atr".to_string(), "W1_rsi".to_string()],
            normalize_features: false,
            genes: vec![gene],
        };

        let json = serde_json::to_string(&artifact).unwrap();
        let back: LivePortfolioArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(artifact, back, "artifact must survive a JSON round-trip");
    }

    #[test]
    fn project_selects_and_reorders_by_name() {
        // raw frame: 3 cols [a, b, c]; effective wants [c, a] (subset + reorder).
        let data = ndarray::array![
            [1.0_f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ];
        let raw = FeatureFrame {
            timestamps: vec![0, 1],
            names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            data: FeatureData::InMemory(data),
        };
        let effective = vec!["c".to_string(), "a".to_string()];
        let projected = project_features_to_effective(&raw, &effective).unwrap();
        assert_eq!(projected.names, effective);
        assert_eq!(projected.n_features(), 2);
        // column 0 == raw "c" == [3, 6]; column 1 == raw "a" == [1, 4]
        assert_eq!(projected.feature_at(0, 0), 3.0);
        assert_eq!(projected.feature_at(1, 0), 6.0);
        assert_eq!(projected.feature_at(0, 1), 1.0);
        assert_eq!(projected.feature_at(1, 1), 4.0);
    }

    #[test]
    fn oos_gate_drops_tail_losers_and_keeps_tail_winners() {
        use crate::validation::{
            ForwardTestSummary, ForwardTestValidationArtifactFile, ForwardTestValidationScope,
        };

        fn gene(id: &str, long_threshold: f32) -> Gene {
            Gene {
                strategy_id: id.to_string(),
                indices: vec![0],
                weights: vec![1.0],
                long_threshold,
                short_threshold: -0.5,
                ..Gene::default()
            }
        }
        fn artifact(gene: &Gene, net_profit: f64) -> ForwardTestValidationArtifactFile {
            let scope = ForwardTestValidationScope {
                dataset_hash: "tail-hash".to_string(),
                evaluation_config_hash: "eval-hash".to_string(),
                strategy_hash: crate::artifact_io::stable_json_hash(gene).unwrap(),
                temporal_scope: neoethos_core::contracts::TemporalScopeHashes {
                    temporal_contract_hash: "t".to_string(),
                    timestamp_policy_hash: "ts".to_string(),
                    feature_availability_policy_hash: "fa".to_string(),
                    label_policy_hash: "lp".to_string(),
                },
            };
            let summary = ForwardTestSummary {
                bars: 10,
                metrics: crate::eval::BacktestMetrics::from_metric_array([
                    net_profit, 1.0, 100_000.0, 0.01, 0.5, 1.2, 1.0, 0.0, 4.0, 0.8, 0.005,
                ]),
                span_days: 1.0,
            };
            ForwardTestValidationArtifactFile::new(scope, summary)
        }

        // Distinct genes so their stable hashes differ.
        let winner = gene("winner", 0.4);
        let loser = gene("loser", 0.6);
        let mut result = crate::discovery::DiscoveryResult {
            portfolio: vec![winner.clone(), loser.clone()],
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: vec!["rsi".to_string()],
            validation_gates: crate::discovery::DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: vec![
                artifact(&winner, 42.0),
                artifact(&loser, -3.0),
            ],
            prop_firm_validation_artifacts: Vec::new(),
            funnel_profile: None,
        };

        let live = LivePortfolioArtifact::from_discovery("EURUSD", "M1", &[], false, &result);
        assert_eq!(
            live.genes.iter().map(|g| g.strategy_id.as_str()).collect::<Vec<_>>(),
            vec!["winner"],
            "only the tail-profitable strategy may reach the live portfolio"
        );

        // Without forward-test evidence the gate cannot run: keep everyone
        // (the legacy/test path), never silently empty the portfolio.
        result.forward_test_validation_artifacts.clear();
        let live = LivePortfolioArtifact::from_discovery("EURUSD", "M1", &[], false, &result);
        assert_eq!(live.genes.len(), 2, "no artifacts → no gate → all members kept");
    }

    #[test]
    fn project_errors_on_missing_feature() {
        let data = ndarray::array![[1.0_f32, 2.0]];
        let raw = FeatureFrame {
            timestamps: vec![0],
            names: vec!["a".to_string(), "b".to_string()],
            data: FeatureData::InMemory(data),
        };
        let effective = vec!["a".to_string(), "missing".to_string()];
        assert!(project_features_to_effective(&raw, &effective).is_err());
    }
}
