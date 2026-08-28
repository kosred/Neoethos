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

use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use neoethos_data::{CanonicalDatasetIdentity, CanonicalTimeframe, FeatureFrame};
use neoethos_dataset_contracts::CanonicalDatasetScope;
use serde::{Deserialize, Serialize};

use crate::Gene;
use crate::data_selection::{
    CanonicalSearchArtifactEnvelopeV2, CanonicalSearchArtifactScopeV2, CanonicalSearchInput,
    CanonicalSearchWindowRoleV1, ExactCanonicalSeries,
};
use crate::discovery::DiscoveryResult;

/// Bumped when the artifact's shape changes incompatibly.
pub const LIVE_PORTFOLIO_SCHEMA_VERSION: u32 = 3;

/// Everything the autonomous trader needs to evaluate a discovered portfolio on
/// fresh data with backtest parity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LivePortfolioArtifact {
    pub schema_version: u32,
    /// Full immutable dataset/generation/manifest/Vortex/feature-plan authority
    /// for this portfolio. No neighboring file or current publication is used
    /// to reconstruct it.
    pub search_scope: CanonicalSearchArtifactScopeV2,
    /// Exact resolved search configuration used by the discovery run.
    pub search_config_hash: String,
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
    /// What the round-trip COST BAND said about each promoted gene, as
    /// `(strategy_id, verdict)` — audit #71.
    ///
    /// The band charges the same candidate at an optimistic and a pessimistic
    /// all-in cost. `cost_band_optimistic_edge_only` means profitable at the
    /// cheap end and NOT at the expensive one: a strategy whose entire result
    /// is a bet that the operator's real spread is the good one. Until
    /// 2026-08-10 the verdict was measured, counted run-level, and then dropped
    /// at the export boundary, so this file — the only artifact a live run reads
    /// — could not tell such a gene from one profitable across the whole band.
    ///
    pub cost_band: Vec<(String, crate::discovery::CostBandVerdict)>,
}

impl LivePortfolioArtifact {
    pub fn from_discovery(
        normalize_features: bool,
        result: &DiscoveryResult,
    ) -> anyhow::Result<Self> {
        result.validate_evaluated_scopes()?;
        let search_scope = result.selection_scope()?.clone();
        let (anchor, higher_tfs) = direct_timeframe_authority(&search_scope)?;
        let genes = drop_retired_rules(
            oos_surviving_genes(result)?,
            &result.effective_feature_names,
        );
        // Only the genes that actually ship, in the order they ship: a verdict
        // for a gene the OOS gate dropped is noise, and a missing verdict for a
        // gene that IS here would be a lie of omission — so every promoted gene
        // gets a row, `Unmeasured` included.
        let cost_band = genes
            .iter()
            .map(|gene| {
                (
                    gene.strategy_id.clone(),
                    result.cost_band_for_strategy(&gene.strategy_id),
                )
            })
            .collect();
        let artifact = Self {
            schema_version: LIVE_PORTFOLIO_SCHEMA_VERSION,
            search_scope,
            search_config_hash: result.search_config_hash.clone(),
            symbol: anchor.symbol_name().to_owned(),
            base_tf: anchor.timeframe().as_str().to_owned(),
            higher_tfs,
            effective_feature_names: result.effective_feature_names.clone(),
            normalize_features,
            genes,
            cost_band,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Validate the complete persisted contract. This is called both before an
    /// atomic write and after every load; a valid-looking display symbol can
    /// never override the embedded exact receipt.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema_version == LIVE_PORTFOLIO_SCHEMA_VERSION,
            "unsupported live-portfolio schema version {}; expected {}",
            self.schema_version,
            LIVE_PORTFOLIO_SCHEMA_VERSION
        );

        // Reuse the canonical authority validator instead of growing a second
        // spelling of the fnv64/scope rules in this artifact module.
        CanonicalSearchArtifactEnvelopeV2::new(
            "neoethos.live-portfolio-authority.v3",
            self.search_scope.clone(),
            self.search_config_hash.clone(),
            (),
        )
        .map_err(anyhow::Error::new)?;
        let receipt = self.search_scope.receipt();
        let anchor_id = receipt.anchor_dataset_identity();
        let anchor_bindings = receipt
            .source_bindings()
            .iter()
            .filter(|binding| binding.dataset_identity() == anchor_id)
            .collect::<Vec<_>>();
        anyhow::ensure!(
            anchor_bindings.len() == 1,
            "live portfolio scope requires exactly one receipt anchor binding; found {}",
            anchor_bindings.len()
        );
        let segments = anchor_bindings[0].segments();
        anyhow::ensure!(
            !segments.is_empty(),
            "live portfolio receipt anchor has no segments"
        );
        anyhow::ensure!(
            segments
                .windows(2)
                .all(|adjacent| adjacent[0].row_end() == adjacent[1].row_start()),
            "live portfolio selection scope cannot cover disjoint anchor segments"
        );
        let first = segments.first().expect("segments checked non-empty");
        let last = segments.last().expect("segments checked non-empty");
        let selected = self.search_scope.evaluated_window();
        anyhow::ensure!(
            selected.row_start() == first.row_start()
                && selected.timestamp_start_ms() == first.timestamp_start_ms(),
            "live portfolio selection scope must start at the receipt anchor"
        );
        match selected.role() {
            CanonicalSearchWindowRoleV1::DiscoveryInput => anyhow::ensure!(
                selected.row_end() == last.row_end()
                    && selected.timestamp_end_ms() == last.timestamp_end_ms(),
                "live portfolio DiscoveryInput scope must exactly cover the receipt anchor"
            ),
            CanonicalSearchWindowRoleV1::InSample => anyhow::ensure!(
                selected.row_end() < last.row_end()
                    && selected.timestamp_end_ms() < last.timestamp_end_ms(),
                "live portfolio InSample scope must be a strict receipt-anchor prefix"
            ),
            role => anyhow::bail!(
                "live portfolio selection scope has unsupported role {role:?}; expected discovery_input or in_sample"
            ),
        }

        let (anchor, expected_higher_tfs) = direct_timeframe_authority(&self.search_scope)?;
        anyhow::ensure!(
            self.symbol == anchor.symbol_name(),
            "live portfolio symbol {} disagrees with search-scope anchor symbol {}",
            self.symbol,
            anchor.symbol_name()
        );
        anyhow::ensure!(
            self.base_tf == anchor.timeframe().as_str(),
            "live portfolio base timeframe {} disagrees with search-scope anchor timeframe {}",
            self.base_tf,
            anchor.timeframe()
        );
        anyhow::ensure!(
            self.higher_tfs == expected_higher_tfs,
            "live portfolio direct higher-timeframe set/order {:?} disagrees with receipt {:?}",
            self.higher_tfs,
            expected_higher_tfs
        );

        anyhow::ensure!(
            !self.effective_feature_names.is_empty(),
            "live portfolio effective feature ordering is empty"
        );
        let mut feature_names = HashSet::with_capacity(self.effective_feature_names.len());
        for (index, name) in self.effective_feature_names.iter().enumerate() {
            anyhow::ensure!(
                !name.trim().is_empty(),
                "live portfolio effective feature name {index} is empty"
            );
            anyhow::ensure!(
                feature_names.insert(name.as_str()),
                "live portfolio effective feature ordering contains duplicate `{name}`"
            );
        }

        anyhow::ensure!(
            self.cost_band.len() == self.genes.len(),
            "live portfolio has {} genes but {} cost-band rows",
            self.genes.len(),
            self.cost_band.len()
        );
        let mut strategy_ids = HashSet::with_capacity(self.genes.len());
        for (position, (gene, (cost_strategy_id, _))) in
            self.genes.iter().zip(&self.cost_band).enumerate()
        {
            anyhow::ensure!(
                !gene.strategy_id.trim().is_empty(),
                "live portfolio gene {position} has an empty strategy id"
            );
            anyhow::ensure!(
                strategy_ids.insert(gene.strategy_id.as_str()),
                "live portfolio contains duplicate strategy id `{}`",
                gene.strategy_id
            );
            anyhow::ensure!(
                cost_strategy_id == &gene.strategy_id,
                "live portfolio cost-band row {position} belongs to `{cost_strategy_id}` but gene is `{}`",
                gene.strategy_id
            );
            anyhow::ensure!(
                gene.indices.len() == gene.weights.len(),
                "live portfolio gene `{}` has {} indices but {} weights",
                gene.strategy_id,
                gene.indices.len(),
                gene.weights.len()
            );
            anyhow::ensure!(
                gene.indices.windows(2).all(|pair| pair[0] < pair[1]),
                "live portfolio gene `{}` indices are not strictly ordered and unique",
                gene.strategy_id
            );
            for (term, (&feature_index, &weight)) in
                gene.indices.iter().zip(&gene.weights).enumerate()
            {
                anyhow::ensure!(
                    feature_index < self.effective_feature_names.len(),
                    "live portfolio gene `{}` term {term} references feature {feature_index}, but the exact ordering has {} columns",
                    gene.strategy_id,
                    self.effective_feature_names.len()
                );
                anyhow::ensure!(
                    weight.is_finite(),
                    "live portfolio gene `{}` term {term} has a non-finite weight",
                    gene.strategy_id
                );
            }
            for (label, value) in [
                ("long_threshold", gene.long_threshold),
                ("short_threshold", gene.short_threshold),
                ("tp_pips", gene.tp_pips),
                ("sl_pips", gene.sl_pips),
                ("stop_vol_mult", gene.stop_vol_mult),
            ] {
                anyhow::ensure!(
                    value.is_finite(),
                    "live portfolio gene `{}` has non-finite {label}",
                    gene.strategy_id
                );
            }
        }
        Ok(())
    }

    /// Reopen the exact canonical generations named by this artifact and prove
    /// the rebuilt feature plan/provenance equals the embedded search receipt.
    pub fn load_exact_search_input(
        &self,
        data_root: impl AsRef<Path>,
    ) -> anyhow::Result<CanonicalSearchInput> {
        self.validate()?;
        let anchor = self
            .search_scope
            .receipt()
            .validate()
            .map_err(anyhow::Error::new)?;
        let higher_timeframes = self
            .higher_tfs
            .iter()
            .map(|timeframe| {
                timeframe.parse::<CanonicalTimeframe>().map_err(|error| {
                    anyhow::anyhow!("invalid receipt-derived direct timeframe {timeframe}: {error}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let exact = ExactCanonicalSeries::open(data_root.as_ref().to_path_buf(), anchor)
            .map_err(anyhow::Error::new)?;
        let input = exact
            .load_search_input(&higher_timeframes)
            .map_err(anyhow::Error::new)?;
        let rebuilt_receipt = input.receipt().map_err(anyhow::Error::new)?;
        self.search_scope
            .validate_against_receipt(&rebuilt_receipt)
            .map_err(anyhow::Error::new)?;
        Ok(input)
    }

    /// Verify that a live cTrader session is the same environment/account/symbol
    /// authority captured by discovery. The app supplies values returned by the
    /// broker itself, not settings or a filename.
    pub fn validate_ctrader_runtime_binding(
        &self,
        environment: neoethos_data::CTraderEnvironment,
        account_id: i64,
        symbol_id: i64,
        symbol_name: &str,
    ) -> anyhow::Result<()> {
        self.validate()?;
        let anchor = self
            .search_scope
            .receipt()
            .validate()
            .map_err(anyhow::Error::new)?;
        let CanonicalDatasetScope::CTrader {
            environment: expected_environment,
            account_id: expected_account_id,
            symbol_id: expected_symbol_id,
            ..
        } = anchor.scope()
        else {
            anyhow::bail!(
                "live portfolio search receipt is not cTrader broker data; external research data cannot authorize live execution"
            );
        };
        anyhow::ensure!(
            *expected_environment == environment,
            "live cTrader environment {} disagrees with portfolio receipt {}",
            environment.as_str(),
            expected_environment.as_str()
        );
        anyhow::ensure!(
            *expected_account_id == account_id,
            "live cTrader account {account_id} disagrees with portfolio receipt account {expected_account_id}"
        );
        anyhow::ensure!(
            *expected_symbol_id == symbol_id && anchor.symbol_name() == symbol_name,
            "live cTrader symbol {symbol_name}/{symbol_id} disagrees with portfolio receipt {}/{}",
            anchor.symbol_name(),
            expected_symbol_id
        );
        Ok(())
    }

    /// The cost-band verdict recorded for `strategy_id` in THIS artifact.
    ///
    /// A gene with no row is [`CostBandVerdict::Unmeasured`], never
    /// `SurvivesBand`. Strict v3 validation rejects such a missing row on load;
    /// the fallback is only defensive for an in-memory value mutated after
    /// validation.
    pub fn cost_band_for(&self, strategy_id: &str) -> crate::discovery::CostBandVerdict {
        self.cost_band
            .iter()
            .find(|(id, _)| id == strategy_id)
            .map(|(_, verdict)| *verdict)
            .unwrap_or(crate::discovery::CostBandVerdict::Unmeasured)
    }
}

fn direct_timeframe_authority(
    search_scope: &CanonicalSearchArtifactScopeV2,
) -> anyhow::Result<(CanonicalDatasetIdentity, Vec<String>)> {
    search_scope.validate().map_err(anyhow::Error::new)?;
    let anchor = search_scope
        .receipt()
        .validate()
        .map_err(anyhow::Error::new)?;
    let mut direct_timeframes = BTreeSet::new();
    for binding in search_scope.receipt().source_bindings() {
        let identity = CanonicalDatasetIdentity::from_path_component(binding.dataset_identity())
            .map_err(|error| {
                anyhow::anyhow!(
                    "live portfolio source binding `{}` has an invalid dataset identity: {error}",
                    binding.source_node_id()
                )
            })?;
        if identity.scope() == anchor.scope()
            && identity.symbol_name() == anchor.symbol_name()
            && identity.bar_timestamp_convention() == anchor.bar_timestamp_convention()
        {
            anyhow::ensure!(
                direct_timeframes.insert(identity.timeframe()),
                "live portfolio receipt contains duplicate direct {} generation for the anchor series",
                identity.timeframe()
            );
        }
    }
    anyhow::ensure!(
        direct_timeframes.remove(&anchor.timeframe()),
        "live portfolio receipt has no exact direct base-timeframe binding"
    );
    let higher_tfs = direct_timeframes
        .into_iter()
        .map(|timeframe| timeframe.as_str().to_owned())
        .collect();
    Ok((anchor, higher_tfs))
}

/// Process-wide set of RETIRED trading rules, installed once at startup from
/// the operator's `Settings` (`install_search_runtime_overrides_from_settings`).
///
/// Not installed ⇒ empty ⇒ every gene is kept, which is exactly the behaviour
/// this file had before #219 and is what unit tests see.
static RETIRED_RULES: std::sync::OnceLock<neoethos_core::strategy_identity::RetiredRules> =
    std::sync::OnceLock::new();

/// Read `<data_dir>/strategy_blacklist.json` and install the retired rule set.
/// Idempotent: the first install wins, like every other runtime-override
/// boundary in this crate.
pub fn install_retired_rules_from_settings(s: &neoethos_core::Settings) {
    let retired =
        neoethos_core::strategy_identity::RetiredRules::load_from_data_dir(&s.system.data_dir);
    if retired.entries > 0 {
        tracing::info!(
            target: "neoethos_search::live_portfolio",
            blacklist_entries = retired.entries,
            retired_rules = retired.len(),
            unreadable_entries = retired.unreadable_entries,
            data_dir = %s.system.data_dir.display(),
            "auto-cull blacklist loaded — discovery will refuse to promote these rules"
        );
    }
    let _ = RETIRED_RULES.set(retired);
}

/// The installed retired-rule set, or an empty one when nothing was installed.
pub fn current_retired_rules() -> &'static neoethos_core::strategy_identity::RetiredRules {
    static EMPTY: std::sync::OnceLock<neoethos_core::strategy_identity::RetiredRules> =
        std::sync::OnceLock::new();
    RETIRED_RULES.get().unwrap_or_else(|| {
        EMPTY.get_or_init(neoethos_core::strategy_identity::RetiredRules::default)
    })
}

/// AUTO-CULL GATE — item #219, 2026-08-10.
///
/// The retirement loop was only half closed. `strategy_blacklist::is_blacklisted`
/// stops a retired artifact being SELECTED (`server::autonomous`,
/// `app_services::federation`), but `neoethos-search` held zero references to
/// the blacklist: the GA was free to re-derive the culled rule on the very run
/// `app_services::rediscovery` queued after the cull, and a portfolio pairing
/// that rule with two different genes hashed differently as an artifact, so
/// selection did not catch it either.
///
/// This filters at the ONE artifact the autonomous trader consumes, by the SAME
/// identity the blacklist stores (`neoethos_core::strategy_identity`), so a
/// retired rule cannot come back bundled with new company. The gene stays in
/// every other discovery artifact for inspection — nothing is deleted.
///
/// Silent when nothing is retired. Loud, per rule, when something is.
fn drop_retired_rules(genes: Vec<Gene>, feature_names: &[String]) -> Vec<Gene> {
    let retired = current_retired_rules();
    if retired.is_empty() || genes.is_empty() {
        return genes;
    }
    let names: Vec<&str> = feature_names.iter().map(String::as_str).collect();
    let before = genes.len();
    let mut kept = Vec::with_capacity(before);
    for gene in genes {
        let value = match serde_json::to_value(&gene) {
            Ok(v) => v,
            Err(err) => {
                // Serializing a Gene basically cannot fail. If it ever does,
                // KEEPING the member loudly beats dropping a strategy nobody
                // retired — the same direction `oos_surviving_genes` chose.
                tracing::warn!(
                    target: "neoethos_search::live_portfolio",
                    strategy_id = %gene.strategy_id,
                    error = %err,
                    "auto-cull gate: could not hash gene — keeping it WITHOUT a blacklist check"
                );
                kept.push(gene);
                continue;
            }
        };
        let fingerprint = neoethos_core::strategy_identity::gene_rule_fingerprint(&value, &names);
        if retired.contains(&fingerprint) {
            tracing::warn!(
                target: "neoethos_search::live_portfolio",
                strategy_id = %gene.strategy_id,
                rule_fingerprint = %fingerprint,
                "AUTO-CULL GATE: this rule was RETIRED by the live loop and is dropped from                  the live portfolio. The search re-derived a strategy the operator already                  stopped for losing; it stays in the discovery artifacts for inspection"
            );
            continue;
        }
        kept.push(gene);
    }
    if kept.len() < before {
        tracing::warn!(
            target: "neoethos_search::live_portfolio",
            dropped = before - kept.len(),
            kept = kept.len(),
            retired_rules = retired.len(),
            "auto-cull gate: the search rediscovered retired rules — GA time was spent              re-deriving strategies that can never trade"
        );
    }
    kept
}

/// OOS gate for LIVE trading (audit B02, 2026-07-13): only strategies that
/// made money on the never-seen held-out tail reach the live portfolio.
///
/// The full `DiscoveryResult` (portfolio JSON, quality report, walkforward
/// artifacts) is untouched — the evidence stays on disk for the operator.
/// This gate applies at the ONE artifact the autonomous trader consumes.
/// Matching is by `stable_json_hash(gene)`, the same hash
/// `compute_discovery_forward_test_artifacts` stamps into each artifact's
/// strict strategy identity, so no positional assumptions are made.
///
/// Missing, duplicated, extra, or substituted validation evidence fails closed
/// before any live artifact is constructed.
fn oos_surviving_genes(result: &DiscoveryResult) -> anyhow::Result<Vec<Gene>> {
    result.validate_complete_promotion_evidence()?;
    let passing: std::collections::HashSet<&str> = result
        .forward_test_validation_artifacts
        .iter()
        .filter(|artifact| artifact.summary().metrics.net_profit > 0.0)
        .map(|artifact| artifact.strategy_identity().exact_gene_hash())
        .collect();
    let mut kept = Vec::with_capacity(result.portfolio.len());
    for gene in &result.portfolio {
        let hash = crate::artifact_io::stable_json_hash(gene)?;
        if passing.contains(hash.as_str()) {
            kept.push(gene.clone());
        } else {
            tracing::info!(
                target: "neoethos_search::live_portfolio",
                strategy_hash = %hash,
                strategy_id = %gene.strategy_id,
                "OOS gate: dropped from LIVE portfolio — non-positive net profit on the \
                 held-out tail (it remains in the discovery artifacts for inspection)"
            );
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
    Ok(kept)
}

/// Write the live portfolio artifact as pretty JSON. Additive — does NOT touch
/// any existing discovery artifact. Reads the in-effect normalize flag from the
/// data-runtime overrides so the trader knows whether discovery normalized.
pub fn save_live_portfolio_json(
    path: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> anyhow::Result<()> {
    let normalize_features = neoethos_data::current_data_runtime_overrides().normalize_features;
    let artifact = LivePortfolioArtifact::from_discovery(normalize_features, result)?;
    artifact.validate()?;
    crate::artifact_io::write_json_atomic(path, &artifact)
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
    artifact.validate()?;
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
    raw.select_columns(&keep_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_discovery_result() -> crate::discovery::DiscoveryResult {
        let gene = Gene {
            strategy_id: "sample-live-gene".to_owned(),
            indices: vec![0],
            weights: vec![1.0],
            ..Gene::default()
        };
        sample_discovery_result_for(vec![gene])
    }

    fn sample_discovery_result_for(portfolio: Vec<Gene>) -> crate::discovery::DiscoveryResult {
        use crate::validation::{
            CanonicalBacktestArtifactFile, ForwardTestSummary, ForwardTestValidationArtifactFile,
            PropFirmRiskRules, PropFirmRiskValidationArtifactFile, PropFirmRiskValidationSummary,
            WalkforwardSummary, WalkforwardValidationArtifactFile,
        };

        const CONFIG_HASH: &str = "fnv64:0123456789abcdef";
        let (search_input_receipt, selection_scope, holdout_scope) = sample_discovery_authority();
        let canonical_backtest_artifacts = portfolio
            .iter()
            .map(|gene| {
                CanonicalBacktestArtifactFile::new(
                    selection_scope.clone(),
                    CONFIG_HASH,
                    gene,
                    crate::eval::BacktestMetrics::from_metric_array([0.0; 11]),
                )
                .expect("strict canonical live fixture")
            })
            .collect();
        let walkforward_validation_artifacts = portfolio
            .iter()
            .map(|gene| {
                WalkforwardValidationArtifactFile::new(
                    selection_scope.clone(),
                    CONFIG_HASH,
                    gene,
                    WalkforwardSummary {
                        walk_forward_splits: 1,
                        avg_pnl: 1.0,
                        avg_win_rate: 0.5,
                        avg_max_dd: 0.0,
                        avg_max_consec_losses: 0.0,
                        avg_daily_min_dd: 0.0,
                        avg_max_daily_loss: 0.0,
                        any_daily_loss_breach: false,
                        any_consistency_violation: false,
                        any_trade_limit_violation: false,
                        all_min_trading_days_ok: true,
                        splits: Vec::new(),
                    },
                )
                .expect("strict walk-forward live fixture")
            })
            .collect();
        let forward_test_validation_artifacts = portfolio
            .iter()
            .map(|gene| {
                let mut metrics = [0.0; 11];
                metrics[0] = 1.0;
                metrics[8] = 1.0;
                ForwardTestValidationArtifactFile::new(
                    holdout_scope.clone(),
                    CONFIG_HASH,
                    gene,
                    ForwardTestSummary {
                        bars: 20,
                        metrics: crate::eval::BacktestMetrics::from_metric_array(metrics),
                        span_days: 1.0,
                    },
                )
                .expect("strict forward-test live fixture")
            })
            .collect();
        let prop_firm_validation_artifacts = portfolio
            .iter()
            .map(|gene| {
                PropFirmRiskValidationArtifactFile::new(
                    holdout_scope.clone(),
                    CONFIG_HASH,
                    gene,
                    PropFirmRiskValidationSummary {
                        rules: PropFirmRiskRules::default(),
                        trades_observed: 1,
                        trading_days_observed: 1,
                        max_daily_loss_pct_observed: 0.0,
                        max_overall_drawdown_pct_observed: 0.0,
                        largest_profit_share_observed: 0.0,
                        max_trades_per_day_observed: 1,
                        net_return_pct: 0.01,
                        daily_loss_breach: false,
                        overall_drawdown_breach: false,
                        consistency_violation: false,
                        trade_limit_violation: false,
                        min_trading_days_ok: true,
                        profit_target_met: true,
                        all_rules_passed: true,
                    },
                )
                .expect("strict prop-firm live fixture")
            })
            .collect();
        let mut validation_gates = crate::discovery::DiscoveryValidationGates::pending();
        validation_gates.walkforward_passed = true;
        validation_gates.cpcv_passed = true;
        crate::discovery::DiscoveryResult {
            search_input_receipt,
            selection_scope,
            holdout_scope: Some(holdout_scope),
            search_config_hash: CONFIG_HASH.to_string(),
            cost_band_by_strategy: Vec::new(),
            cost_band_census: crate::discovery::CostBandCensus::default(),
            portfolio,
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: vec!["close_minus_open".to_string()],
            validation_gates,
            canonical_backtest_artifacts,
            walkforward_validation_artifacts,
            forward_test_validation_artifacts,
            prop_firm_validation_artifacts,
            funnel_profile: None,
            effective_smc_gate_threshold: f64::NAN,
        }
    }

    fn legacy_v1_artifact() -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "symbol": "EURUSD",
            "base_tf": "M1",
            "higher_tfs": [],
            "effective_feature_names": ["close_minus_open"],
            "normalize_features": false,
            "genes": [],
            "cost_band": []
        })
    }

    fn valid_v3_artifact() -> LivePortfolioArtifact {
        LivePortfolioArtifact::from_discovery(false, &sample_discovery_result())
            .expect("valid v3 artifact")
    }

    fn write_test_json(label: &str, value: &serde_json::Value) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "neoethos_live_portfolio_{label}_{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            serde_json::to_vec(value).expect("serialize test JSON"),
        )
        .expect("write test live portfolio");
        path
    }

    fn sample_search_input_receipt() -> crate::data_selection::CanonicalSearchInputReceiptV2 {
        let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let anchor = features.provenance().bindings()[0]
            .dataset_identity()
            .clone();
        crate::data_selection::CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &features)
            .expect("canonical search test receipt")
    }

    fn sample_discovery_authority() -> (
        crate::data_selection::CanonicalSearchInputReceiptV2,
        CanonicalSearchArtifactScopeV2,
        CanonicalSearchArtifactScopeV2,
    ) {
        let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let ohlcv = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
        let receipt = sample_search_input_receipt();
        let input = crate::data_selection::CanonicalSearchRunInputV2::new_for_test_values(
            receipt.clone(),
            &features,
            &ohlcv,
        )
        .expect("canonical live test input");
        let selection_scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
            CanonicalSearchWindowRoleV1::InSample,
            &input,
            0..80,
        )
        .expect("canonical InSample live test scope");
        let holdout_scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
            CanonicalSearchWindowRoleV1::Holdout,
            &input,
            80..100,
        )
        .expect("canonical Holdout live test scope");
        (receipt, selection_scope, holdout_scope)
    }

    #[test]
    fn artifact_round_trips_through_json() {
        let mut gene = Gene::default();
        gene.indices = vec![0];
        gene.weights = vec![0.5];
        gene.long_threshold = 0.1;
        gene.short_threshold = -0.1;
        gene.strategy_id = "test-gene".to_string();

        let mut result = sample_discovery_result_for(vec![gene]);
        result.cost_band_by_strategy = vec![(
            "test-gene".to_string(),
            crate::discovery::CostBandVerdict::OptimisticEdgeOnly,
        )];
        let artifact = LivePortfolioArtifact::from_discovery(false, &result).unwrap();

        let json = serde_json::to_string(&artifact).unwrap();
        let back: LivePortfolioArtifact = serde_json::from_str(&json).unwrap();
        back.validate().unwrap();
        assert_eq!(artifact, back, "artifact must survive a JSON round-trip");
        // The verdict has to survive the round trip too — it is the whole point
        // of carrying it (audit #71), and a silently dropped field would look
        // exactly like the pre-2026-08-10 behaviour it replaces.
        assert_eq!(
            back.cost_band_for("test-gene"),
            crate::discovery::CostBandVerdict::OptimisticEdgeOnly
        );
        // An unknown strategy reads as Unmeasured, never as a pass.
        assert_eq!(
            back.cost_band_for("no-such-gene"),
            crate::discovery::CostBandVerdict::Unmeasured
        );
    }

    #[test]
    fn load_rejects_receipt_free_v1_even_when_the_old_payload_is_well_formed() {
        let value = legacy_v1_artifact();
        let path = write_test_json("reject_v1", &value);

        let error = load_live_portfolio_json(&path)
            .expect_err("v1 has no immutable receipt/config authority and must fail closed");
        assert!(
            error.to_string().contains("schema")
                || error.to_string().contains("authority")
                || error.to_string().contains("search_scope"),
            "failure should name the unsupported schema/authority boundary: {error}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_unknown_fields_instead_of_ignoring_them() {
        let mut value = serde_json::to_value(valid_v3_artifact()).expect("serialize valid v3");
        value
            .as_object_mut()
            .expect("artifact is an object")
            .insert("future_semantics".to_string(), serde_json::json!(true));
        let path = write_test_json("reject_unknown", &value);

        let error = load_live_portfolio_json(&path)
            .expect_err("unknown persisted semantics must fail closed");
        assert!(
            error.to_string().contains("unknown field") || error.to_string().contains("schema"),
            "failure should name the unknown/unsupported contract: {error}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_display_identity_that_disagrees_with_embedded_scope() {
        let mut value = serde_json::to_value(valid_v3_artifact()).expect("serialize valid v3");
        let object = value.as_object_mut().expect("artifact is an object");
        object.insert("symbol".to_string(), serde_json::json!("GBPUSD"));
        let path = write_test_json("reject_scope_mismatch", &value);

        let error = load_live_portfolio_json(&path)
            .expect_err("display symbol must not override the exact receipt anchor");
        assert!(
            error.to_string().contains("symbol") || error.to_string().contains("scope"),
            "failure should name the scope/display mismatch: {error}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn from_discovery_derives_identity_from_the_result_receipt() {
        let result = sample_discovery_result();
        let artifact = LivePortfolioArtifact::from_discovery(false, &result).unwrap();

        assert_eq!(artifact.symbol, "EURUSD");
        assert_eq!(artifact.base_tf, "M1");
        assert!(artifact.higher_tfs.is_empty());
    }

    #[test]
    fn live_broker_binding_rejects_an_external_research_receipt() {
        let artifact = valid_v3_artifact();
        let error = artifact
            .validate_ctrader_runtime_binding(
                neoethos_data::CTraderEnvironment::Demo,
                42,
                1,
                "EURUSD",
            )
            .expect_err("an external research receipt cannot authorize cTrader execution");
        assert!(error.to_string().contains("cTrader"));
    }

    #[test]
    fn project_selects_and_reorders_by_name() {
        // raw frame: 3 cols [a, b, c]; effective wants [c, a] (subset + reorder).
        let data = ndarray::array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0],];
        let raw = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_matrix(
            neoethos_data::test_fixtures::canonical_test_timestamps(2),
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            data,
        )
        .expect("valid f64 test frame");
        let effective = vec!["c".to_string(), "a".to_string()];
        let projected = project_features_to_effective(&raw, &effective).unwrap();
        assert_eq!(projected.names, effective);
        assert_eq!(projected.n_features(), 2);
        // column 0 == raw "c" == [3, 6]; column 1 == raw "a" == [1, 4]
        assert_eq!(projected.cell(0, 0).unwrap().value, 3.0);
        assert_eq!(projected.cell(1, 0).unwrap().value, 6.0);
        assert_eq!(projected.cell(0, 1).unwrap().value, 1.0);
        assert_eq!(projected.cell(1, 1).unwrap().value, 4.0);
    }

    #[test]
    fn oos_gate_drops_tail_losers_and_keeps_tail_winners() {
        use crate::validation::{ForwardTestSummary, ForwardTestValidationArtifactFile};

        fn gene(id: &str, long_threshold: f64) -> Gene {
            Gene {
                strategy_id: id.to_string(),
                indices: vec![0],
                weights: vec![1.0],
                long_threshold,
                short_threshold: -0.5,
                ..Gene::default()
            }
        }
        fn artifact(
            gene: &Gene,
            holdout_scope: &CanonicalSearchArtifactScopeV2,
            net_profit: f64,
        ) -> ForwardTestValidationArtifactFile {
            let summary = ForwardTestSummary {
                bars: 20,
                metrics: crate::eval::BacktestMetrics::from_metric_array([
                    net_profit, 1.0, 100_000.0, 0.01, 0.5, 1.2, 1.0, 0.0, 4.0, 0.8, 0.005,
                ]),
                span_days: 1.0,
            };
            ForwardTestValidationArtifactFile::new(
                holdout_scope.clone(),
                "fnv64:0123456789abcdef",
                gene,
                summary,
            )
            .expect("strict forward-test live fixture")
        }

        // Distinct genes so their stable hashes differ.
        let winner = gene("winner", 0.4);
        let loser = gene("loser", 0.6);
        let mut result = sample_discovery_result_for(vec![winner.clone(), loser.clone()]);
        let holdout_scope = result
            .holdout_scope
            .as_ref()
            .expect("strict live fixture holdout")
            .clone();
        result.forward_test_validation_artifacts = vec![
            artifact(&winner, &holdout_scope, 42.0),
            artifact(&loser, &holdout_scope, -3.0),
        ];

        let live = LivePortfolioArtifact::from_discovery(false, &result).unwrap();
        assert_eq!(
            live.genes
                .iter()
                .map(|g| g.strategy_id.as_str())
                .collect::<Vec<_>>(),
            vec!["winner"],
            "only the tail-profitable strategy may reach the live portfolio"
        );

        // Live is an authority boundary: missing held-out evidence must refuse
        // the artifact, never silently keep every in-sample winner.
        result.forward_test_validation_artifacts.clear();
        let error = LivePortfolioArtifact::from_discovery(false, &result)
            .expect_err("missing forward-test evidence must fail closed for live");
        assert!(
            error.to_string().contains("forward_test") && error.to_string().contains("missing"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn project_errors_on_missing_feature() {
        let data = ndarray::array![[1.0_f64, 2.0]];
        let raw = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_matrix(
            neoethos_data::test_fixtures::canonical_test_timestamps(1),
            vec!["a".to_string(), "b".to_string()],
            data,
        )
        .expect("valid f64 test frame");
        let effective = vec!["a".to_string(), "missing".to_string()];
        assert!(project_features_to_effective(&raw, &effective).is_err());
    }
}
