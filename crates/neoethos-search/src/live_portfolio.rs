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
//! built from, the `normalize_features` flag, and the exact live trading policy
//! resolved from the immutable search authority — everything the trader needs
//! to rebuild the EXACT signals and exits the genes were evolved against.
//!
//! Discovery writes it (`save_live_portfolio_json`, called next to
//! `save_portfolio_json`); the trader reads it (`load_live_portfolio_json`) and
//! projects its freshly-computed features onto `effective_feature_names` with
//! [`project_features_to_effective`] (the same by-name selection discovery's
//! forward-test path uses).
//!
//! File persistence uses an explicit shared-receipt envelope: one complete
//! receipt plus hash-bound scope references. V6 explicitly separates calibration
//! from reserved final evaluation; old V5 is not silently upgraded.

use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use neoethos_data::{CanonicalDatasetIdentity, CanonicalTimeframe, FeatureFrame};
use neoethos_dataset_contracts::CanonicalDatasetScope;
use serde::{Deserialize, Serialize};

use crate::Gene;
use crate::data_selection::{
    CanonicalSearchArtifactEnvelopeV2, CanonicalSearchArtifactScopeRefV1,
    CanonicalSearchArtifactScopeV2, CanonicalSearchInput, CanonicalSearchInputReceiptV2,
    CanonicalSearchWindowRoleV1,
};
use crate::discovery::DiscoveryResult;

mod sealed_evaluation;
use sealed_evaluation::SealedEvaluationPolicyV1;

/// Bumped when the artifact's shape changes incompatibly.
pub const LIVE_PORTFOLIO_SCHEMA_VERSION: u32 = 6;

const LIVE_TRADING_POLICY_SCHEMA_VERSION_V1: u16 = 1;
const LIVE_TRADING_POLICY_SCHEMA_VERSION_V2: u16 = 2;
const LIVE_TRADING_POLICY_IDENTITY_KIND_V1: &str = "neoethos.live-trading-policy-identity.v1";
const LIVE_TRADING_POLICY_IDENTITY_KIND_V2: &str = "neoethos.live-trading-policy-identity.v2";

#[derive(Serialize)]
struct LiveTradingPolicyIdentityBodyV1<'a> {
    kind: &'static str,
    schema_version: u16,
    source_search_config_hash: &'a str,
    source_resolved_config_hash: &'a str,
    trailing_enabled: bool,
    trailing_be_trigger_r: f64,
    trailing_stop_multiplier: f64,
    trailing_min_lock_pips: f64,
    kill_zones_enabled: bool,
    baseline_spread_pips: f64,
    session_spread_pips: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sealed_evaluation_v1: Option<&'a SealedEvaluationPolicyV1>,
}

/// Position-management policy that was actually priced by discovery.
///
/// This is sealed from `PopulationAutoSearchAuthorityV1`, never reconstructed
/// from the config file that happens to exist when live trading starts. The
/// source hashes and the policy identity make accidental drift or partial edits
/// fail closed at artifact load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveTradingPolicyV1 {
    pub schema_version: u16,
    pub source_search_config_hash: String,
    pub source_resolved_config_hash: String,
    pub identity_hash: String,
    pub trailing_enabled: bool,
    pub trailing_be_trigger_r: f64,
    pub trailing_stop_multiplier: f64,
    pub trailing_min_lock_pips: f64,
    pub kill_zones_enabled: bool,
    pub baseline_spread_pips: f64,
    pub session_spread_pips: Option<[f64; 3]>,
    /// Added only by the actual Search producer. Absence preserves historical
    /// V1 bytes but cannot authorize archived signal/account-risk replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sealed_evaluation_v1: Option<SealedEvaluationPolicyV1>,
}

impl LiveTradingPolicyV1 {
    pub(crate) fn from_search_authority(
        authority: &crate::run_identity::PopulationAutoSearchAuthorityV1,
        evaluation: &crate::genetic::EvaluationConfig,
        smc_gate_disabled: bool,
        adaptive_stops: &crate::stop_target::ResolvedAdaptiveStopsPolicyV1,
    ) -> anyhow::Result<Self> {
        authority.validate()?;
        anyhow::ensure!(
            evaluation.growth_goal == authority.growth_goal(),
            "actual stage-1 growth goal disagrees with its resolved Search authority"
        );
        let stamp = authority.resolved_config_stamp();
        let mut policy = Self::seal(
            authority.search_config_hash(),
            &stamp.config_hash,
            neoethos_core::config::ExitPolicyConfig {
                trailing_enabled: stamp.trailing_enabled,
                trailing_be_trigger_r: stamp.trailing_be_trigger_r,
                trailing_stop_multiplier: stamp.trailing_give_back_r,
                trailing_min_lock_pips: stamp.trailing_min_lock_pips,
            },
            stamp.kill_zones_enabled,
            stamp.spread_pips,
            stamp.session_spread_pips,
        )?;
        anyhow::ensure!(
            evaluation.symbol == stamp.symbol
                && evaluation.initial_equity == stamp.initial_balance
                && evaluation.risk_per_trade_min == stamp.risk_per_trade_min
                && evaluation.risk_per_trade_max == stamp.risk_per_trade_max
                && evaluation.commission_per_trade == stamp.commission_per_trade
                && evaluation.pip_value_per_lot == stamp.pip_value_per_lot
                && evaluation.swap_long_pips_per_day == stamp.swap_long_pips_per_day
                && evaluation.swap_short_pips_per_day == stamp.swap_short_pips_per_day
                && evaluation.growth_objective == (stamp.mode == "risky"),
            "actual stage-1 evaluation disagrees with its resolved Search authority"
        );
        policy.sealed_evaluation_v1 = Some(SealedEvaluationPolicyV1::from_evaluation(
            evaluation,
            smc_gate_disabled,
            adaptive_stops,
        )?);
        policy.schema_version = LIVE_TRADING_POLICY_SCHEMA_VERSION_V2;
        policy.identity_hash = policy.computed_identity_hash()?;
        policy.validate()?;
        Ok(policy)
    }

    fn seal(
        source_search_config_hash: &str,
        source_resolved_config_hash: &str,
        exit_policy: neoethos_core::config::ExitPolicyConfig,
        kill_zones_enabled: bool,
        baseline_spread_pips: f64,
        session_spread_pips: Option<[f64; 3]>,
    ) -> anyhow::Result<Self> {
        let mut policy = Self {
            schema_version: LIVE_TRADING_POLICY_SCHEMA_VERSION_V1,
            source_search_config_hash: source_search_config_hash.to_owned(),
            source_resolved_config_hash: source_resolved_config_hash.to_owned(),
            identity_hash: String::new(),
            trailing_enabled: exit_policy.trailing_enabled,
            trailing_be_trigger_r: exit_policy.trailing_be_trigger_r,
            trailing_stop_multiplier: exit_policy.trailing_stop_multiplier,
            trailing_min_lock_pips: exit_policy.trailing_min_lock_pips,
            kill_zones_enabled,
            baseline_spread_pips,
            session_spread_pips,
            sealed_evaluation_v1: None,
        };
        policy.validate_value_domains()?;
        policy.identity_hash = policy.computed_identity_hash()?;
        policy.validate()?;
        Ok(policy)
    }

    fn identity_body(&self) -> LiveTradingPolicyIdentityBodyV1<'_> {
        LiveTradingPolicyIdentityBodyV1 {
            kind: if self.schema_version == LIVE_TRADING_POLICY_SCHEMA_VERSION_V2 {
                LIVE_TRADING_POLICY_IDENTITY_KIND_V2
            } else {
                LIVE_TRADING_POLICY_IDENTITY_KIND_V1
            },
            schema_version: self.schema_version,
            source_search_config_hash: &self.source_search_config_hash,
            source_resolved_config_hash: &self.source_resolved_config_hash,
            trailing_enabled: self.trailing_enabled,
            trailing_be_trigger_r: self.trailing_be_trigger_r,
            trailing_stop_multiplier: self.trailing_stop_multiplier,
            trailing_min_lock_pips: self.trailing_min_lock_pips,
            kill_zones_enabled: self.kill_zones_enabled,
            baseline_spread_pips: self.baseline_spread_pips,
            session_spread_pips: self.session_spread_pips,
            sealed_evaluation_v1: self.sealed_evaluation_v1.as_ref(),
        }
    }

    fn computed_identity_hash(&self) -> anyhow::Result<String> {
        crate::artifact_io::stable_json_hash(&self.identity_body())
    }

    fn validate_value_domains(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            is_fnv64_hash(&self.source_search_config_hash),
            "live trading policy source search-config hash is not canonical"
        );
        anyhow::ensure!(
            is_fnv64_hash(&self.source_resolved_config_hash),
            "live trading policy source resolved-config hash is not canonical"
        );
        for (name, value) in [
            ("trailing_be_trigger_r", self.trailing_be_trigger_r),
            ("trailing_stop_multiplier", self.trailing_stop_multiplier),
            ("trailing_min_lock_pips", self.trailing_min_lock_pips),
            ("baseline_spread_pips", self.baseline_spread_pips),
        ] {
            anyhow::ensure!(
                value.is_finite() && value >= 0.0,
                "live trading policy {name} must be finite and non-negative"
            );
        }
        if self.trailing_enabled {
            anyhow::ensure!(
                self.trailing_be_trigger_r > 0.0 && self.trailing_stop_multiplier > 0.0,
                "enabled live trailing requires positive trigger and stop multiplier"
            );
        }
        if let Some(curve) = self.session_spread_pips {
            anyhow::ensure!(
                curve.iter().all(|value| value.is_finite() && *value >= 0.0),
                "live trading policy session spread curve must be finite and non-negative"
            );
        }
        Ok(())
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            matches!(
                (self.schema_version, self.sealed_evaluation_v1.is_some()),
                (LIVE_TRADING_POLICY_SCHEMA_VERSION_V1, false)
                    | (LIVE_TRADING_POLICY_SCHEMA_VERSION_V2, true)
            ),
            "unsupported or incomplete live trading policy schema {}",
            self.schema_version,
        );
        self.validate_value_domains()?;
        if let Some(evaluation) = &self.sealed_evaluation_v1 {
            evaluation.validate_against_policy(self)?;
        }
        anyhow::ensure!(
            self.identity_hash == self.computed_identity_hash()?,
            "live trading policy identity hash mismatch"
        );
        Ok(())
    }

    /// Return the exact archived Search evaluator, never ambient Settings or
    /// runtime defaults. V1 files remain readable but lack this authority.
    pub fn sealed_evaluation_config(&self) -> anyhow::Result<crate::genetic::EvaluationConfig> {
        self.validate()?;
        self.sealed_evaluation_v1
            .as_ref()
            .map(SealedEvaluationPolicyV1::to_evaluation)
            .ok_or_else(|| {
                anyhow::anyhow!("legacy live policy lacks sealed signal/account evaluation")
            })
    }

    pub fn sealed_smc_gate_disabled(&self) -> anyhow::Result<bool> {
        self.validate()?;
        self.sealed_evaluation_v1
            .as_ref()
            .map(SealedEvaluationPolicyV1::smc_gate_disabled)
            .ok_or_else(|| {
                anyhow::anyhow!("legacy live policy lacks its effective SMC bypass decision")
            })
    }

    pub fn sealed_adaptive_stops_policy(
        &self,
    ) -> anyhow::Result<&crate::stop_target::ResolvedAdaptiveStopsPolicyV1> {
        self.validate()?;
        self.sealed_evaluation_v1
            .as_ref()
            .map(SealedEvaluationPolicyV1::adaptive_stops)
            .ok_or_else(|| {
                anyhow::anyhow!("legacy live policy lacks its exact adaptive-stop recipe")
            })
    }

    fn with_final_smc_gate(mut self, effective_gate: f64) -> anyhow::Result<Self> {
        self.validate()?;
        if let Some(evaluation) = &mut self.sealed_evaluation_v1 {
            evaluation.set_final_gate(effective_gate)?;
            self.identity_hash = self.computed_identity_hash()?;
            self.validate()?;
        }
        Ok(self)
    }

    pub const fn exit_policy(&self) -> neoethos_core::config::ExitPolicyConfig {
        neoethos_core::config::ExitPolicyConfig {
            trailing_enabled: self.trailing_enabled,
            trailing_be_trigger_r: self.trailing_be_trigger_r,
            trailing_stop_multiplier: self.trailing_stop_multiplier,
            trailing_min_lock_pips: self.trailing_min_lock_pips,
        }
    }

    pub fn expected_spread_pips_at(&self, timestamp_ms: i64) -> f64 {
        match self.session_spread_pips {
            Some(curve) => match crate::eval::SessionSpreadProfile::bucket_index(timestamp_ms) {
                1 => curve[1],
                2 => curve[2],
                _ => curve[0],
            },
            None => self.baseline_spread_pips,
        }
    }
}

fn is_fnv64_hash(value: &str) -> bool {
    value.len() == 22
        && value.starts_with("fnv64:")
        && value[6..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Held-out evidence that determines live Risky-mode sizing for one promoted
/// gene. These are the selection-validation calibration metrics, not the gene's
/// in-sample fields. Reuse the complete forward-test artifact so the exact
/// genome, dataset, holdout window and configuration travel with the metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveSizingEvidenceV1 {
    pub forward_test: crate::validation::ForwardTestValidationArtifactFile,
}

impl LiveSizingEvidenceV1 {
    fn from_forward_test(
        gene: &Gene,
        artifact: &crate::validation::ForwardTestValidationArtifactFile,
    ) -> anyhow::Result<Self> {
        artifact.strategy_identity().validate_against(gene)?;
        let evidence = Self {
            forward_test: artifact.clone(),
        };
        evidence.validate_against(gene)?;
        Ok(evidence)
    }

    fn validate_against(&self, gene: &Gene) -> anyhow::Result<()> {
        self.forward_test.validate_against(
            self.forward_test.scope(),
            self.forward_test.search_config_hash(),
            gene,
        )?;
        Self::validate_calibration_metrics(&gene.strategy_id, &self.oos_metrics())
    }

    /// Reuse the existing numerical selection/sizing gate before portfolio
    /// capacity is applied. This checks metrics only and grants no artifact,
    /// final-evaluation, or trading authority.
    pub(crate) fn validate_calibration_metrics(
        strategy_id: &str,
        metrics: &crate::eval::BacktestMetrics,
    ) -> anyhow::Result<()> {
        for (name, value) in [
            ("net_profit", metrics.net_profit),
            ("sharpe", metrics.sharpe),
            ("peak_equity", metrics.peak_equity),
            ("max_drawdown", metrics.max_drawdown),
            ("win_rate", metrics.win_rate),
            ("profit_factor", metrics.profit_factor),
            ("expectancy", metrics.expectancy),
            ("monthly_target_hit_rate", metrics.monthly_target_hit_rate),
            ("consistency", metrics.consistency),
            ("max_daily_drawdown", metrics.max_daily_drawdown),
        ] {
            anyhow::ensure!(
                value.is_finite(),
                "live sizing evidence for `{}` has non-finite OOS {name}",
                strategy_id
            );
        }
        anyhow::ensure!(
            metrics.trade_count > 0,
            "live sizing evidence for `{}` has no OOS trades",
            strategy_id
        );
        anyhow::ensure!(
            metrics.net_profit > 0.0 && metrics.expectancy > 0.0,
            "live sizing evidence for `{}` is not profitable after OOS costs",
            strategy_id
        );
        anyhow::ensure!(
            metrics.win_rate > 0.0 && metrics.win_rate <= 1.0,
            "live sizing evidence for `{}` has invalid OOS win rate {}",
            strategy_id,
            metrics.win_rate
        );
        anyhow::ensure!(
            metrics.profit_factor > 1.0,
            "live sizing evidence for `{}` has no positive OOS profit-factor edge ({})",
            strategy_id,
            metrics.profit_factor
        );
        anyhow::ensure!(
            neoethos_gpu_contracts::resident_search_scoring_v2::risky_growth_v5_half_kelly_fraction(
                metrics.win_rate,
                metrics.profit_factor,
            ) > 0.0,
            "live sizing evidence for `{}` yields no positive half-Kelly fraction",
            strategy_id
        );
        Ok(())
    }

    pub fn oos_metrics(&self) -> crate::eval::BacktestMetrics {
        self.forward_test.summary().metrics
    }

    /// Same bounded half-Kelly fraction used by Risky search scoring v5.
    pub fn half_kelly_risk_fraction(&self) -> f64 {
        neoethos_gpu_contracts::resident_search_scoring_v2::risky_growth_v5_half_kelly_fraction(
            self.oos_metrics().win_rate,
            self.oos_metrics().profit_factor,
        )
    }
}

/// Everything the autonomous trader needs to evaluate a discovered portfolio on
/// fresh data with backtest parity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LivePortfolioArtifact {
    pub schema_version: u32,
    /// Full immutable dataset/generation/manifest/Vortex/feature-plan authority
    /// for this portfolio. No neighboring file or current publication is used
    /// to reconstruct it.
    pub search_scope: CanonicalSearchArtifactScopeV2,
    /// Reserved final evaluation tail, never used for gene selection or sizing.
    /// A scope reservation alone is not evidence of first use or a passed test.
    pub final_holdout_scope: CanonicalSearchArtifactScopeV2,
    /// Exact resolved search configuration used by the discovery run.
    pub search_config_hash: String,
    /// Exact position-management and session-cost policy priced by that search.
    /// Live execution must use this, never today's mutable Settings value.
    pub live_trading_policy: LiveTradingPolicyV1,
    pub symbol: String,
    pub base_tf: String,
    pub higher_tfs: Vec<String>,
    /// Feature names AFTER discovery's prefilter, in the exact column order the
    /// gene `indices` reference.
    pub effective_feature_names: Vec<String>,
    /// Whether discovery normalized features. The exact fitted state travels
    /// inside `search_scope.receipt()`: live applies that state without fitting
    /// on incoming bars. This flag must agree with the embedded receipt.
    pub normalize_features: bool,
    /// The promoted portfolio — FULL genes, including SMC flags + SL/TP.
    pub genes: Vec<Gene>,
    /// Selection-validation calibration metrics used to size each gene. Entries are
    /// ordered one-to-one with `genes`; live must never size from in-sample
    /// `Gene` metrics or from an unrelated fixed bankroll ladder.
    pub sizing_evidence: Vec<LiveSizingEvidenceV1>,
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

const LIVE_PORTFOLIO_SHARED_RECEIPT_KIND_V1: &str = "neoethos.live-portfolio-shared-receipt.v1";

/// Owned transport body for an enclosing artifact that stores the receipt ONCE.
/// It is separate from the V6 in-memory/default serde shape. The embedded
/// portfolio schema binds the new calibration/final-scope semantics explicitly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LivePortfolioSharedReceiptBodyV1 {
    schema_version: u16,
    portfolio_schema_version: u32,
    portfolio_identity_sha256: String,
    search_scope: CanonicalSearchArtifactScopeRefV1,
    final_holdout_scope: CanonicalSearchArtifactScopeRefV1,
    search_config_hash: String,
    live_trading_policy: LiveTradingPolicyV1,
    symbol: String,
    base_tf: String,
    higher_tfs: Vec<String>,
    effective_feature_names: Vec<String>,
    normalize_features: bool,
    genes: Vec<Gene>,
    sizing_evidence: Vec<LiveSizingEvidenceSharedReceiptV1>,
    cost_band: Vec<(String, crate::discovery::CostBandVerdict)>,
}

/// Borrowing encoder: only the small scope references are owned. No receipt,
/// gene array, feature-name array or fitted-state tree is cloned to serialize.
#[derive(Serialize)]
pub struct LivePortfolioSharedReceiptBodyRefV1<'a> {
    schema_version: u16,
    portfolio_schema_version: u32,
    portfolio_identity_sha256: String,
    search_scope: CanonicalSearchArtifactScopeRefV1,
    final_holdout_scope: CanonicalSearchArtifactScopeRefV1,
    search_config_hash: &'a str,
    live_trading_policy: &'a LiveTradingPolicyV1,
    symbol: &'a str,
    base_tf: &'a str,
    higher_tfs: &'a [String],
    effective_feature_names: &'a [String],
    normalize_features: bool,
    genes: &'a [Gene],
    sizing_evidence: Vec<LiveSizingEvidenceSharedReceiptRefV1<'a>>,
    cost_band: &'a [(String, crate::discovery::CostBandVerdict)],
}

impl LivePortfolioSharedReceiptBodyRefV1<'_> {
    /// Identity already computed while constructing this immutable borrowed
    /// body. Consumers can bind its encoded bytes without hashing the same
    /// expanded portfolio again. This does not cache a mutable artifact.
    pub fn portfolio_identity_sha256(&self) -> &str {
        &self.portfolio_identity_sha256
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveSizingEvidenceSharedReceiptV1 {
    schema_version: u16,
    forward_test_artifact_kind: String,
    forward_test_schema_version: u32,
    scope: CanonicalSearchArtifactScopeRefV1,
    search_config_hash: String,
    strategy_identity: crate::validation::ValidationStrategyIdentityV2,
    summary: crate::validation::ForwardTestSummary,
}

#[derive(Serialize)]
struct LiveSizingEvidenceSharedReceiptRefV1<'a> {
    schema_version: u16,
    forward_test_artifact_kind: &'static str,
    forward_test_schema_version: u32,
    scope: CanonicalSearchArtifactScopeRefV1,
    search_config_hash: &'a str,
    strategy_identity: &'a crate::validation::ValidationStrategyIdentityV2,
    summary: &'a crate::validation::ForwardTestSummary,
}

impl LiveSizingEvidenceSharedReceiptV1 {
    fn attach(
        self,
        receipt: &CanonicalSearchInputReceiptV2,
        gene: &Gene,
    ) -> anyhow::Result<LiveSizingEvidenceV1> {
        anyhow::ensure!(
            self.schema_version == 1,
            "unsupported shared live-sizing schema"
        );
        anyhow::ensure!(
            self.forward_test_artifact_kind
                == crate::validation::FORWARD_TEST_VALIDATION_ARTIFACT_KIND
                && self.forward_test_schema_version
                    == crate::validation::FORWARD_TEST_VALIDATION_SCHEMA_VERSION,
            "shared live-sizing evidence has unsupported forward-test semantics"
        );
        // Do not regenerate an identity from the gene before checking the one
        // actually stored in this evidence; that would conceal substitution.
        self.strategy_identity.validate_against(gene)?;
        let scope = self.scope.attach(receipt).map_err(anyhow::Error::new)?;
        let forward_test = crate::validation::ForwardTestValidationArtifactFile::new(
            scope,
            self.search_config_hash,
            gene,
            self.summary,
        )?;
        let evidence = LiveSizingEvidenceV1 { forward_test };
        evidence.validate_against(gene)?;
        Ok(evidence)
    }
}

impl LivePortfolioSharedReceiptBodyV1 {
    pub fn portfolio_identity_sha256(&self) -> &str {
        &self.portfolio_identity_sha256
    }

    pub fn search_scope(&self) -> &CanonicalSearchArtifactScopeRefV1 {
        &self.search_scope
    }

    pub fn attach(
        self,
        receipt: &CanonicalSearchInputReceiptV2,
    ) -> anyhow::Result<LivePortfolioArtifact> {
        anyhow::ensure!(
            self.schema_version == 1,
            "unsupported shared live-portfolio body schema"
        );
        anyhow::ensure!(
            self.genes.len() == self.sizing_evidence.len(),
            "shared live portfolio gene/sizing-evidence counts disagree"
        );
        let search_scope = self
            .search_scope
            .attach(receipt)
            .map_err(anyhow::Error::new)?;
        let final_holdout_scope = self
            .final_holdout_scope
            .attach(receipt)
            .map_err(anyhow::Error::new)?;
        let sizing_evidence = self
            .sizing_evidence
            .into_iter()
            .zip(&self.genes)
            .map(|(evidence, gene)| evidence.attach(receipt, gene))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let artifact = LivePortfolioArtifact {
            schema_version: self.portfolio_schema_version,
            search_scope,
            final_holdout_scope,
            search_config_hash: self.search_config_hash,
            live_trading_policy: self.live_trading_policy,
            symbol: self.symbol,
            base_tf: self.base_tf,
            higher_tfs: self.higher_tfs,
            effective_feature_names: self.effective_feature_names,
            normalize_features: self.normalize_features,
            genes: self.genes,
            sizing_evidence,
            cost_band: self.cost_band,
        };
        artifact.validate()?;
        anyhow::ensure!(
            crate::canonical_locked_portfolio_identity_sha256_v1(&artifact)?
                == self.portfolio_identity_sha256,
            "shared live portfolio changed its original V6 canonical identity"
        );
        Ok(artifact)
    }
}

#[derive(Serialize)]
struct LivePortfolioSharedReceiptEnvelopeRefV1<'a> {
    shared_receipt_schema_version: u16,
    artifact_kind: &'static str,
    input_receipt: &'a CanonicalSearchInputReceiptV2,
    portfolio: LivePortfolioSharedReceiptBodyRefV1<'a>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LivePortfolioSharedReceiptEnvelopeV1 {
    shared_receipt_schema_version: u16,
    artifact_kind: String,
    input_receipt: CanonicalSearchInputReceiptV2,
    portfolio: LivePortfolioSharedReceiptBodyV1,
}

impl LivePortfolioSharedReceiptEnvelopeV1 {
    fn attach(self) -> anyhow::Result<LivePortfolioArtifact> {
        anyhow::ensure!(
            self.shared_receipt_schema_version == 1
                && self.artifact_kind == LIVE_PORTFOLIO_SHARED_RECEIPT_KIND_V1,
            "unsupported shared live-portfolio envelope schema/kind"
        );
        self.portfolio.attach(&self.input_receipt)
    }
}

impl LivePortfolioArtifact {
    pub fn shared_receipt_body_v1(
        &self,
    ) -> anyhow::Result<LivePortfolioSharedReceiptBodyRefV1<'_>> {
        self.validate()?;
        let sizing_evidence = self
            .sizing_evidence
            .iter()
            .map(|evidence| {
                let forward = &evidence.forward_test;
                Ok(LiveSizingEvidenceSharedReceiptRefV1 {
                    schema_version: 1,
                    forward_test_artifact_kind:
                        crate::validation::FORWARD_TEST_VALIDATION_ARTIFACT_KIND,
                    forward_test_schema_version:
                        crate::validation::FORWARD_TEST_VALIDATION_SCHEMA_VERSION,
                    scope: CanonicalSearchArtifactScopeRefV1::from_scope(forward.scope())
                        .map_err(anyhow::Error::new)?,
                    search_config_hash: forward.search_config_hash(),
                    strategy_identity: forward.strategy_identity(),
                    summary: forward.summary(),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(LivePortfolioSharedReceiptBodyRefV1 {
            schema_version: 1,
            portfolio_schema_version: self.schema_version,
            portfolio_identity_sha256: crate::canonical_locked_portfolio_identity_sha256_v1(self)?,
            search_scope: CanonicalSearchArtifactScopeRefV1::from_scope(&self.search_scope)
                .map_err(anyhow::Error::new)?,
            final_holdout_scope: CanonicalSearchArtifactScopeRefV1::from_scope(
                &self.final_holdout_scope,
            )
            .map_err(anyhow::Error::new)?,
            search_config_hash: &self.search_config_hash,
            live_trading_policy: &self.live_trading_policy,
            symbol: &self.symbol,
            base_tf: &self.base_tf,
            higher_tfs: &self.higher_tfs,
            effective_feature_names: &self.effective_feature_names,
            normalize_features: self.normalize_features,
            genes: &self.genes,
            sizing_evidence,
            cost_band: &self.cost_band,
        })
    }

    fn shared_receipt_envelope_v1(
        &self,
    ) -> anyhow::Result<LivePortfolioSharedReceiptEnvelopeRefV1<'_>> {
        Ok(LivePortfolioSharedReceiptEnvelopeRefV1 {
            shared_receipt_schema_version: 1,
            artifact_kind: LIVE_PORTFOLIO_SHARED_RECEIPT_KIND_V1,
            input_receipt: self.search_scope.receipt(),
            portfolio: self.shared_receipt_body_v1()?,
        })
    }

    /// Explicit new standalone wire form. Default Serialize remains the original
    /// V6 form; obsolete V5 evidence is not reinterpreted as a reserved final test.
    pub fn to_shared_receipt_json_bytes_v1(&self) -> anyhow::Result<Vec<u8>> {
        Ok(serde_json::to_vec(&self.shared_receipt_envelope_v1()?)?)
    }

    pub fn from_persisted_json_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        // This discriminator skips other values without materializing a JSON
        // tree. The selected strict typed decoder rejects unknown/duplicate
        // fields. No untagged fallback can reinterpret a malformed new envelope.
        #[derive(Deserialize)]
        struct WireVersion {
            shared_receipt_schema_version: Option<u16>,
        }
        let version: WireVersion = serde_json::from_slice(bytes)?;
        let artifact = if version.shared_receipt_schema_version.is_some() {
            serde_json::from_slice::<LivePortfolioSharedReceiptEnvelopeV1>(bytes)?.attach()?
        } else {
            serde_json::from_slice::<Self>(bytes)?
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn from_discovery(
        normalize_features: bool,
        result: &DiscoveryResult,
    ) -> anyhow::Result<Self> {
        result.validate_evaluated_scopes()?;
        let search_scope = result.selection_scope()?.clone();
        let calibration_scope = result.calibration_scope.as_ref().ok_or_else(|| {
            anyhow::anyhow!("V6 live portfolio requires an explicit selection-validation calibration scope; legacy two-way evidence cannot reserve a new final test")
        })?;
        let final_holdout_scope = result.holdout_scope.clone().ok_or_else(|| {
            anyhow::anyhow!("V6 live portfolio requires an explicit reserved final holdout scope")
        })?;
        anyhow::ensure!(
            result
                .forward_test_validation_artifacts
                .iter()
                .all(|artifact| artifact.scope() == calibration_scope),
            "live sizing inputs must all use the declared calibration scope, not the final test"
        );
        let (anchor, higher_tfs) = direct_timeframe_authority(&search_scope)?;
        let genes = drop_retired_rules(
            oos_surviving_genes(result)?,
            &result.effective_feature_names,
        );
        let sizing_evidence = oos_sizing_evidence(result, &genes)?;
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
        let live_trading_policy = result
            .funnel_profile
            .as_ref()
            .and_then(|funnel| funnel.live_trading_policy_v1())
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "discovery result has no run-authority-derived live trading policy; refusing to reconstruct exits from ambient Settings"
                )
            })?
            .with_final_smc_gate(result.effective_smc_gate_threshold)?;
        let artifact = Self {
            schema_version: LIVE_PORTFOLIO_SCHEMA_VERSION,
            search_scope,
            final_holdout_scope,
            search_config_hash: result.search_config_hash.clone(),
            live_trading_policy,
            symbol: anchor.symbol_name().to_owned(),
            base_tf: anchor.timeframe().as_str().to_owned(),
            higher_tfs,
            effective_feature_names: result.effective_feature_names.clone(),
            normalize_features,
            genes,
            sizing_evidence,
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
            "neoethos.live-portfolio-authority.v6",
            self.search_scope.clone(),
            self.search_config_hash.clone(),
            (),
        )
        .map_err(anyhow::Error::new)?;
        self.live_trading_policy.validate()?;
        if let Some(evaluation) = &self.live_trading_policy.sealed_evaluation_v1 {
            anyhow::ensure!(
                evaluation.symbol() == self.symbol,
                "sealed evaluation symbol differs from the exact portfolio symbol"
            );
        }
        anyhow::ensure!(
            self.live_trading_policy.source_search_config_hash == self.search_config_hash,
            "live trading policy belongs to search config {}, but portfolio belongs to {}",
            self.live_trading_policy.source_search_config_hash,
            self.search_config_hash
        );
        let receipt = self.search_scope.receipt();
        anyhow::ensure!(
            self.normalize_features == receipt.normalization_fitted_state().is_some(),
            "live portfolio normalization flag disagrees with its persisted training fit; normalized legacy artifacts without fitted parameters must be regenerated"
        );
        if let Some(fitted) = receipt.normalization_fitted_state() {
            fitted.validate_feature_names(&self.effective_feature_names)?;
        }
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
        crate::discovery::validate_normalization_training_scope(receipt, selected)?;
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

        self.final_holdout_scope
            .validate_against_receipt(receipt)
            .map_err(anyhow::Error::new)?;
        let final_window = self.final_holdout_scope.evaluated_window();
        anyhow::ensure!(
            selected.role() == CanonicalSearchWindowRoleV1::InSample
                && final_window.role() == CanonicalSearchWindowRoleV1::Holdout
                && final_window.row_start() > selected.row_end()
                && final_window.timestamp_start_ms() > selected.timestamp_end_ms()
                && final_window.row_end() == last.row_end()
                && final_window.timestamp_end_ms() == last.timestamp_end_ms(),
            "V6 final holdout must be an exact receipt tail after a nonempty separate calibration interval"
        );

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
        anyhow::ensure!(
            self.sizing_evidence.len() == self.genes.len(),
            "live portfolio has {} genes but {} OOS sizing-evidence rows",
            self.genes.len(),
            self.sizing_evidence.len()
        );
        let mut strategy_ids = HashSet::with_capacity(self.genes.len());
        for (position, ((gene, sizing), (cost_strategy_id, _))) in self
            .genes
            .iter()
            .zip(&self.sizing_evidence)
            .zip(&self.cost_band)
            .enumerate()
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
            sizing.validate_against(gene)?;
            let holdout_scope = sizing.forward_test.scope();
            holdout_scope
                .validate_against_receipt(receipt)
                .map_err(anyhow::Error::new)?;
            let held_out = holdout_scope.evaluated_window();
            anyhow::ensure!(
                sizing.forward_test.search_config_hash() == self.search_config_hash
                    && selected.role() == CanonicalSearchWindowRoleV1::InSample
                    && held_out.role() == CanonicalSearchWindowRoleV1::SelectionValidation
                    && selected.row_end() == held_out.row_start()
                    && selected.timestamp_end_ms() < held_out.timestamp_start_ms()
                    && held_out.row_end() == final_window.row_start()
                    && held_out.timestamp_end_ms() < final_window.timestamp_start_ms()
                    && self
                        .sizing_evidence
                        .first()
                        .is_some_and(|first| first.forward_test.scope() == holdout_scope),
                "live sizing evidence must use this search config and its exact shared calibration interval before the final holdout"
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

    /// Build with the persisted recipe and fitted state, never the process's
    /// current normalization flag. The producer applies the saved transform
    /// inside its admitted batches without learning from incoming bars.
    pub fn prepare_live_features(
        &self,
        dataset: &neoethos_data::SymbolDataset,
    ) -> anyhow::Result<FeatureFrame> {
        self.validate()?;
        let receipt = self.search_scope.receipt();
        let mut options = receipt.feature_build_options().cloned().unwrap_or_else(|| {
            neoethos_data::FeatureBuildOptions {
                higher_tfs: self.higher_tfs.clone(),
                ..Default::default()
            }
        });
        match receipt.normalization_fitted_state() {
            Some(fitted) => {
                neoethos_data::prepare_multitimeframe_features_with_fitted_normalization(
                    dataset,
                    &self.base_tf,
                    &options,
                    fitted,
                )
            }
            None => {
                options.normalization_training_rows = None;
                options.drop_columns_without_normalization_training_support = false;
                neoethos_data::prepare_multitimeframe_features_raw_with_options(
                    dataset,
                    &self.base_tf,
                    &options,
                )
            }
        }
    }

    /// Project the producer's live frame only after verifying that it used this
    /// artifact's exact fitted state. Matching names alone cannot prove matching
    /// numeric inputs; an unrelated/refitted or raw frame is refused.
    pub fn project_live_features(&self, features: &FeatureFrame) -> anyhow::Result<FeatureFrame> {
        self.validate()?;
        let expected = self.search_scope.receipt().normalization_fitted_state();
        match (expected, features.normalization_fitted_state()) {
            (None, None) => {}
            (Some(expected), Some(actual)) => anyhow::ensure!(
                expected.fitted_state_hash()? == actual.fitted_state_hash()?,
                "live feature normalization fit differs from this portfolio's historical training fit"
            ),
            _ => anyhow::bail!(
                "live feature normalization presence differs from the portfolio's persisted training fit"
            ),
        }
        self.search_scope
            .receipt()
            .validate_live_feature_plan(features)
            .map_err(anyhow::Error::new)?;
        project_features_to_effective(features, &self.effective_feature_names)
    }

    /// Pin the canonical generations named by this artifact before feature work
    /// and prove the rebuilt input equals its saved receipt. The current store
    /// refuses a generation that became stale before pinning; it must never be
    /// replaced with CURRENT or reconstructed from guessed historical metadata.
    pub fn load_exact_search_input(
        &self,
        data_root: impl AsRef<Path>,
    ) -> anyhow::Result<CanonicalSearchInput> {
        self.validate()?;
        let higher_timeframes = self
            .higher_tfs
            .iter()
            .map(|timeframe| {
                timeframe.parse::<CanonicalTimeframe>().map_err(|error| {
                    anyhow::anyhow!("invalid receipt-derived direct timeframe {timeframe}: {error}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let receipt = self.search_scope.receipt();
        let options = receipt.feature_build_options().cloned().unwrap_or_else(|| {
            neoethos_data::FeatureBuildOptions {
                higher_tfs: higher_timeframes
                    .iter()
                    .map(|tf| tf.as_str().to_owned())
                    .collect(),
                ..Default::default()
            }
        });
        let input = CanonicalSearchInput::from_recorded_receipt(
            data_root.as_ref(),
            receipt.clone(),
            &options,
        )
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
    /// A gene with no row is [`crate::discovery::CostBandVerdict::Unmeasured`], never
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

    /// OOS sizing evidence for an exact strategy identity.
    pub fn sizing_evidence_for(&self, strategy_id: &str) -> Option<&LiveSizingEvidenceV1> {
        self.sizing_evidence
            .iter()
            .find(|evidence| evidence.forward_test.strategy_identity().strategy_id() == strategy_id)
    }

    /// Per-member Risky sizing ceiling. The current live signal is
    /// a net vote of all genes, while the discovery result stores held-out
    /// metrics per gene rather than for that combined vote. Until a combined
    /// OOS ledger is persisted, the weakest positive half-Kelly value is a
    /// sizing heuristic, NOT the Kelly optimum or a drawdown guarantee for the
    /// combined vote. Runtime account and position limits still apply.
    pub fn portfolio_half_kelly_risk_fraction(&self) -> anyhow::Result<f64> {
        self.validate()?;
        self.sizing_evidence
            .iter()
            .map(LiveSizingEvidenceV1::half_kelly_risk_fraction)
            .reduce(f64::min)
            .filter(|fraction| fraction.is_finite() && *fraction > 0.0)
            .ok_or_else(|| anyhow::anyhow!("live portfolio has no positive OOS sizing edge"))
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

/// Candidate selection filter: retain strategies profitable on the explicitly
/// designated calibration interval. This is part of fitting/selection, not a
/// result on the reserved final holdout and not permission to trade.
///
/// The full `DiscoveryResult` (portfolio JSON, quality report, walkforward
/// artifacts) is untouched — the evidence stays on disk for the operator.
/// V6 also reserves the later final scope for the frozen candidate evaluation.
/// Its presence does not prove that evaluation completed or passed. Live
/// admission remains a separate boundary; this candidate may enter training.
/// Matching is by `stable_json_hash(gene)`, the same hash
/// `compute_discovery_forward_test_artifacts` stamps into each artifact's
/// strict strategy identity, so no positional assumptions are made.
///
/// Missing, duplicated, extra, or substituted validation evidence fails closed
/// before any candidate artifact is constructed.
fn oos_surviving_genes(result: &DiscoveryResult) -> anyhow::Result<Vec<Gene>> {
    result.validate_complete_selection_evidence()?;
    filter_oos_survivors_from_validated_diagnostics(result)
}

fn oos_sizing_evidence(
    result: &DiscoveryResult,
    genes: &[Gene],
) -> anyhow::Result<Vec<LiveSizingEvidenceV1>> {
    let mut evidence = Vec::with_capacity(genes.len());
    for gene in genes {
        let exact_hash = crate::artifact_io::stable_json_hash(gene)?;
        let artifact = result
            .forward_test_validation_artifacts
            .iter()
            .find(|candidate| {
                candidate.strategy_identity().exact_gene_hash() == exact_hash.as_str()
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "promoted strategy `{}` has no exact OOS sizing evidence",
                    gene.strategy_id
                )
            })?;
        evidence.push(LiveSizingEvidenceV1::from_forward_test(gene, artifact)?);
    }
    Ok(evidence)
}

/// Apply the deterministic calibration profitability filter after the caller
/// has checked complete selection evidence. This does not manufacture final
/// evaluation or live-promotion authority from the calibration diagnostics.
fn filter_oos_survivors_from_validated_diagnostics(
    result: &DiscoveryResult,
) -> anyhow::Result<Vec<Gene>> {
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
/// any existing discovery artifact. The completed search receipt determines
/// normalization; mutable process settings cannot relabel the result.
pub fn save_live_portfolio_json(
    path: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> anyhow::Result<()> {
    let normalize_features = result
        .search_input_receipt
        .normalization_fitted_state()
        .is_some();
    let artifact = LivePortfolioArtifact::from_discovery(normalize_features, result)?;
    artifact.validate()?;
    crate::artifact_io::write_json_atomic(path, &artifact.shared_receipt_envelope_v1()?)
}

/// Load a live portfolio artifact written by [`save_live_portfolio_json`].
pub fn load_live_portfolio_json(path: impl AsRef<Path>) -> anyhow::Result<LivePortfolioArtifact> {
    let raw = std::fs::read_to_string(&path).map_err(|e| {
        anyhow::anyhow!(
            "live portfolio artifact {} not readable: {e}",
            path.as_ref().display()
        )
    })?;
    let artifact =
        LivePortfolioArtifact::from_persisted_json_bytes(raw.as_bytes()).map_err(|e| {
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

    include!("live_portfolio_report_fixture_tests.rs");

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
        sample_discovery_result_for_features(
            portfolio,
            &neoethos_data::test_fixtures::ctrader_sample_feature_frame(),
        )
    }

    fn sample_discovery_result_for_features(
        portfolio: Vec<Gene>,
        features: &FeatureFrame,
    ) -> crate::discovery::DiscoveryResult {
        sample_discovery_result_for_authority(portfolio, sample_discovery_authority(features))
    }

    type SampleDiscoveryAuthority = (
        crate::data_selection::CanonicalSearchInputReceiptV2,
        CanonicalSearchArtifactScopeV2,
        CanonicalSearchArtifactScopeV2,
        CanonicalSearchArtifactScopeV2,
    );

    fn sample_discovery_result_for_authority(
        portfolio: Vec<Gene>,
        authority: SampleDiscoveryAuthority,
    ) -> crate::discovery::DiscoveryResult {
        use crate::validation::{
            CanonicalBacktestArtifactFile, ForwardTestSummary, ForwardTestValidationArtifactFile,
            PropFirmRiskRules, PropFirmRiskValidationArtifactFile, PropFirmRiskValidationSummary,
            WalkforwardSummary, WalkforwardValidationArtifactFile,
        };

        const CONFIG_HASH: &str = "fnv64:0123456789abcdef";
        let (search_input_receipt, selection_scope, calibration_scope, holdout_scope) = authority;
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
                let metrics = [
                    1.0, 1.0, 100_001.0, 0.01, 0.55, 1.5, 1.0, 0.5, 1.0, 0.8, 0.005,
                ];
                ForwardTestValidationArtifactFile::new(
                    calibration_scope.clone(),
                    CONFIG_HASH,
                    gene,
                    ForwardTestSummary {
                        bars: 10,
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
                    calibration_scope.clone(),
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
        let mut result = crate::discovery::DiscoveryResult {
            search_input_receipt,
            selection_scope,
            calibration_scope: Some(calibration_scope),
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
        };
        let mut funnel = crate::funnel_profile::FunnelProfile::new("EURUSD", "M1");
        funnel
            .attach_live_trading_policy_v1(sample_live_trading_policy(CONFIG_HASH))
            .expect("attach live trading policy fixture");
        result.funnel_profile = Some(funnel);
        result
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

    fn sample_live_trading_policy(search_config_hash: &str) -> LiveTradingPolicyV1 {
        LiveTradingPolicyV1::seal(
            search_config_hash,
            "fnv64:fedcba9876543210",
            neoethos_core::config::ExitPolicyConfig {
                trailing_enabled: true,
                trailing_be_trigger_r: 1.25,
                trailing_stop_multiplier: 0.75,
                trailing_min_lock_pips: 3.0,
            },
            false,
            1.5,
            Some([1.8, 0.6, 1.1]),
        )
        .expect("valid sealed live trading policy")
    }

    fn sample_sealed_evaluation_policy() -> LiveTradingPolicyV1 {
        let mut policy = sample_live_trading_policy("fnv64:0123456789abcdef");
        let evaluation = crate::genetic::EvaluationConfig {
            symbol: "EURUSD".to_owned(),
            account_currency: "USD".to_owned(),
            initial_equity: 12_345.67,
            max_hold_bars: 37,
            trailing_enabled: policy.trailing_enabled,
            trailing_atr_multiplier: policy.trailing_stop_multiplier,
            trailing_be_trigger_r: policy.trailing_be_trigger_r,
            trailing_min_lock_pips: policy.trailing_min_lock_pips,
            pip_value: 0.0001,
            spread_pips: policy.baseline_spread_pips,
            commission_per_trade: 7.1,
            pip_value_per_lot: 9.8,
            swap_long_pips_per_day: -0.9,
            swap_short_pips_per_day: 0.4,
            pnl_conversion_fee_rate: 0.002,
            kill_zones_enabled: policy.kill_zones_enabled,
            session_spread_pips: policy.session_spread_pips,
            risk_per_trade_min: 0.003,
            risk_per_trade_max: 0.023,
            high_quality_confidence: 0.73,
            smc_gate_threshold: 0.81,
            smc_weight_ob: 0.11,
            smc_weight_fvg: 0.22,
            smc_weight_liq: 0.33,
            smc_weight_mtf: 0.44,
            smc_weight_premium: 0.55,
            smc_weight_inducement: 0.66,
            smc_weight_bos: 0.77,
            smc_weight_choch: 0.88,
            smc_weight_eqh: 0.99,
            smc_weight_eql: 1.1,
            smc_weight_displacement: 1.21,
            growth_objective: true,
            growth_goal: None,
        };
        policy.sealed_evaluation_v1 = Some(
            SealedEvaluationPolicyV1::from_evaluation(
                &evaluation,
                true,
                &crate::stop_target::ResolvedAdaptiveStopsPolicyV1::capture_current().unwrap(),
            )
            .unwrap(),
        );
        policy.schema_version = LIVE_TRADING_POLICY_SCHEMA_VERSION_V2;
        policy.identity_hash = policy.computed_identity_hash().unwrap();
        policy.validate().unwrap();
        policy
    }

    #[test]
    fn sealed_live_evaluation_roundtrip_retains_every_field_and_final_gate_without_defaults() {
        let original = sample_sealed_evaluation_policy();
        let bytes = serde_json::to_vec(&original).unwrap();
        let restored: LiveTradingPolicyV1 = serde_json::from_slice(&bytes).unwrap();
        let replay = restored.sealed_evaluation_config().unwrap();
        assert_eq!(
            SealedEvaluationPolicyV1::from_evaluation(
                &replay,
                restored.sealed_smc_gate_disabled().unwrap(),
                restored.sealed_adaptive_stops_policy().unwrap(),
            )
            .unwrap(),
            *original.sealed_evaluation_v1.as_ref().unwrap()
        );
        assert_eq!(serde_json::to_vec(&restored).unwrap(), bytes);
        let final_policy = restored.with_final_smc_gate(0.31).unwrap();
        assert_ne!(final_policy.identity_hash, original.identity_hash);
        assert_eq!(
            final_policy
                .sealed_evaluation_config()
                .unwrap()
                .smc_gate_threshold,
            0.31
        );
        let mut expected = original.sealed_evaluation_v1.clone().unwrap();
        expected.set_final_gate(0.31).unwrap();
        assert_eq!(final_policy.sealed_evaluation_v1.as_ref(), Some(&expected));
        assert!(original.clone().with_final_smc_gate(f64::NAN).is_err());

        let mut result = sample_discovery_result();
        result.effective_smc_gate_threshold = 0.31;
        let mut funnel = crate::funnel_profile::FunnelProfile::new("EURUSD", "M1");
        funnel.attach_live_trading_policy_v1(original).unwrap();
        result.funnel_profile = Some(funnel);
        let artifact = LivePortfolioArtifact::from_discovery(false, &result).unwrap();
        assert_eq!(artifact.live_trading_policy, final_policy);
        let compact = artifact.to_shared_receipt_json_bytes_v1().unwrap();
        let restored = LivePortfolioArtifact::from_persisted_json_bytes(&compact).unwrap();
        assert_eq!(restored.live_trading_policy, final_policy);
        result.effective_smc_gate_threshold = f64::NAN;
        assert!(LivePortfolioArtifact::from_discovery(false, &result).is_err());
    }

    #[test]
    fn sealed_live_evaluation_rejects_each_mutated_field_and_legacy_authority_fallback() {
        let legacy = sample_live_trading_policy("fnv64:0123456789abcdef");
        let legacy_bytes = serde_json::to_vec(&legacy).unwrap();
        assert!(!String::from_utf8_lossy(&legacy_bytes).contains("sealed_evaluation"));
        let restored: LiveTradingPolicyV1 = serde_json::from_slice(&legacy_bytes).unwrap();
        restored.validate().unwrap();
        assert!(restored.sealed_evaluation_config().is_err());
        assert!(restored.sealed_smc_gate_disabled().is_err());
        assert!(restored.sealed_adaptive_stops_policy().is_err());
        assert_eq!(serde_json::to_vec(&restored).unwrap(), legacy_bytes);

        let policy = sample_sealed_evaluation_policy();
        let original = serde_json::to_value(&policy).unwrap();
        for (field, value) in original["sealed_evaluation_v1"].as_object().unwrap() {
            let mut changed = original.clone();
            changed["sealed_evaluation_v1"][field] = match value {
                serde_json::Value::Bool(value) => (!value).into(),
                serde_json::Value::String(_) => "OTHER".into(),
                serde_json::Value::Array(_) => serde_json::json!([2.0, 2.0, 2.0]),
                serde_json::Value::Object(value) => {
                    let mut changed = value.clone();
                    let enabled = changed["enabled"].as_bool().unwrap();
                    changed.insert("enabled".to_owned(), (!enabled).into());
                    changed.into()
                }
                serde_json::Value::Number(value) if value.is_u64() => {
                    (value.as_u64().unwrap() + 1).into()
                }
                serde_json::Value::Number(value) => (value.as_f64().unwrap() + 0.01).into(),
                _ => panic!("uncovered sealed snapshot field {field}"),
            };
            let changed: LiveTradingPolicyV1 = serde_json::from_value(changed).unwrap();
            assert!(
                changed.validate().is_err(),
                "changed sealed {field} admitted"
            );
        }
        for mutation in 0..3 {
            let mut changed = original.clone();
            match mutation {
                0 => {
                    changed
                        .as_object_mut()
                        .unwrap()
                        .remove("sealed_evaluation_v1");
                }
                1 => changed["schema_version"] = 1.into(),
                2 => changed["sealed_evaluation_v1"]["unknown_override"] = true.into(),
                _ => unreachable!(),
            }
            assert!(
                serde_json::from_value::<LiveTradingPolicyV1>(changed)
                    .map(|policy| policy.validate().is_err())
                    .unwrap_or(true)
            );
        }
        for (field, value) in original["sealed_evaluation_v1"]["adaptive_stops_v1"]["settings"]
            .as_object()
            .unwrap()
        {
            let mut changed = original.clone();
            changed["sealed_evaluation_v1"]["adaptive_stops_v1"]["settings"][field] = match value {
                serde_json::Value::String(value) => format!("{value}_changed").into(),
                serde_json::Value::Number(value) if value.is_u64() => {
                    (value.as_u64().unwrap() + 1).into()
                }
                serde_json::Value::Number(value) => (value.as_f64().unwrap() + 0.01).into(),
                serde_json::Value::Null => {
                    serde_json::json!({"yang_zhang":0.1,"garman_klass":0.2,"rogers_satchell":0.3,"parkinson":0.4})
                }
                _ => panic!("uncovered adaptive recipe field {field}"),
            };
            let changed: LiveTradingPolicyV1 = serde_json::from_value(changed).unwrap();
            assert!(
                changed.validate().is_err(),
                "changed archived adaptive {field} admitted"
            );
        }
    }

    fn artifact_shape_for_contract_test(
        normalize_features: bool,
        result: &crate::discovery::DiscoveryResult,
    ) -> LivePortfolioArtifact {
        result
            .validate_evaluated_scopes()
            .expect("valid test discovery scopes");
        let search_scope = result
            .selection_scope()
            .expect("valid test selection scope")
            .clone();
        let (anchor, higher_tfs) =
            direct_timeframe_authority(&search_scope).expect("direct timeframe authority");
        let genes = result.portfolio.clone();
        let cost_band = genes
            .iter()
            .map(|gene| {
                (
                    gene.strategy_id.clone(),
                    result.cost_band_for_strategy(&gene.strategy_id),
                )
            })
            .collect();
        let sizing_evidence =
            oos_sizing_evidence(result, &genes).expect("valid OOS sizing evidence fixture");
        let artifact = LivePortfolioArtifact {
            schema_version: LIVE_PORTFOLIO_SCHEMA_VERSION,
            search_scope,
            final_holdout_scope: result
                .holdout_scope
                .clone()
                .expect("reserved final fixture"),
            search_config_hash: result.search_config_hash.clone(),
            live_trading_policy: sample_live_trading_policy(&result.search_config_hash),
            symbol: anchor.symbol_name().to_owned(),
            base_tf: anchor.timeframe().as_str().to_owned(),
            higher_tfs,
            effective_feature_names: result.effective_feature_names.clone(),
            normalize_features,
            genes,
            sizing_evidence,
            cost_band,
        };
        artifact.validate().expect("valid v6 artifact shape");
        artifact
    }

    fn valid_v6_artifact() -> LivePortfolioArtifact {
        artifact_shape_for_contract_test(false, &sample_discovery_result())
    }

    fn shared_receipt_fixture(normalized: bool, members: usize) -> LivePortfolioArtifact {
        let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(100);
        let columns = (0..96)
            .map(|column| {
                let name = if column == 0 {
                    "close_minus_open".to_owned()
                } else {
                    format!("wide_receipt_feature_{column}_{}", "x".repeat(96))
                };
                neoethos_data::FeatureColumnF64::new(
                    name,
                    (0..100)
                        .map(|row| row as f64 + column as f64 / 8.0)
                        .collect(),
                    vec![neoethos_data::FeatureCellValidity::Valid; 100],
                )
                .unwrap()
            })
            .collect();
        let raw = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            timestamps, columns,
        )
        .unwrap();
        let frame = if normalized {
            neoethos_data::test_fixtures::ctrader_test_feature_frame_with_normalization(
                &raw,
                0..80,
                None,
            )
            .unwrap()
        } else {
            raw
        };
        let genes = (0..members)
            .map(|member| Gene {
                strategy_id: format!("shared-receipt-gene-{member}"),
                indices: vec![0],
                weights: vec![1.0],
                ..Gene::default()
            })
            .collect();
        let result = sample_discovery_result_for_features(genes, &frame);
        LivePortfolioArtifact::from_discovery(normalized, &result).unwrap()
    }

    #[test]
    fn shared_portfolio_wire_stores_one_receipt_and_preserves_complete_v6_identity() {
        for normalized in [false, true] {
            let one = shared_receipt_fixture(normalized, 1);
            let many = shared_receipt_fixture(normalized, 8);
            assert_eq!(one.search_scope.receipt(), many.search_scope.receipt());
            let receipt_bytes = serde_json::to_vec(many.search_scope.receipt())
                .unwrap()
                .len();
            let one_bytes = one.to_shared_receipt_json_bytes_v1().unwrap();
            let compact_bytes = many.to_shared_receipt_json_bytes_v1().unwrap();
            let legacy_bytes = serde_json::to_vec(&many).unwrap();
            let legacy_fingerprint =
                neoethos_core::strategy_identity::portfolio_gene_fingerprint(&legacy_bytes)
                    .expect("actual V6 portfolio gene identity");
            assert_eq!(
                neoethos_core::strategy_identity::portfolio_gene_fingerprint(&compact_bytes),
                Some(legacy_fingerprint),
                "compact transport must preserve the existing blacklist identity"
            );
            let legacy_rules =
                neoethos_core::strategy_identity::gene_rule_fingerprints(&legacy_bytes);
            assert_eq!(legacy_rules.len(), many.genes.len());
            assert_eq!(
                neoethos_core::strategy_identity::gene_rule_fingerprints(&compact_bytes),
                legacy_rules,
                "compact portfolios must remain visible to per-gene retirement filtering"
            );
            assert_eq!(
                String::from_utf8_lossy(&compact_bytes)
                    .matches("\"feature_plan_canonical_bytes\"")
                    .count(),
                1
            );
            assert_eq!(
                String::from_utf8_lossy(&legacy_bytes)
                    .matches("\"feature_plan_canonical_bytes\"")
                    .count(),
                10
            );
            assert!(compact_bytes.len() < legacy_bytes.len() / 4);
            assert!(
                compact_bytes.len() - one_bytes.len() < 7 * receipt_bytes / 2,
                "adding genes must add their compact evidence, not repeat the large feature plan"
            );
            let restored =
                LivePortfolioArtifact::from_persisted_json_bytes(&compact_bytes).unwrap();
            assert_eq!(
                serde_json::to_vec(&restored).unwrap(),
                legacy_bytes,
                "every V6 field and f64 value must survive compact transport unchanged"
            );
            assert_eq!(
                crate::canonical_locked_portfolio_identity_sha256_v1(&restored).unwrap(),
                crate::canonical_locked_portfolio_identity_sha256_v1(&many).unwrap()
            );
            assert_eq!(
                restored.search_scope.receipt().normalization_fitted_state(),
                many.search_scope.receipt().normalization_fitted_state()
            );
            let borrowed_body = many.shared_receipt_body_v1().unwrap();
            assert_eq!(
                borrowed_body.portfolio_identity_sha256(),
                crate::canonical_locked_portfolio_identity_sha256_v1(&many).unwrap()
            );
            let body_bytes = serde_json::to_vec(&borrowed_body).unwrap();
            let body: LivePortfolioSharedReceiptBodyV1 =
                serde_json::from_slice(&body_bytes).unwrap();
            assert_eq!(
                serde_json::to_vec(&body).unwrap(),
                body_bytes,
                "borrowed and owned wire encoders must have identical canonical ordering"
            );
            assert!(!String::from_utf8_lossy(&body_bytes).contains("feature_plan_canonical_bytes"));
            assert_eq!(
                serde_json::to_vec(&body.attach(many.search_scope.receipt()).unwrap()).unwrap(),
                legacy_bytes
            );
            let legacy = LivePortfolioArtifact::from_persisted_json_bytes(&legacy_bytes).unwrap();
            assert_eq!(
                serde_json::to_vec(&legacy).unwrap(),
                legacy_bytes,
                "plain V6 loading must preserve its canonical identity"
            );
            eprintln!(
                "shared live portfolio normalized={normalized}: receipt={receipt_bytes}, one={}, eight={}, legacy_eight={}",
                one_bytes.len(),
                compact_bytes.len(),
                legacy_bytes.len()
            );
        }
    }

    #[test]
    fn shared_portfolio_wire_refuses_tampering_without_rebinding_evidence() {
        let artifact = valid_v6_artifact();
        let encoded = artifact.to_shared_receipt_json_bytes_v1().unwrap();
        let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        let mutations: &[(&str, fn(&mut serde_json::Value))] = &[
            ("final scope role", |v| {
                v["portfolio"]["final_holdout_scope"]["evaluated_window"]["role"] =
                    serde_json::json!("selection_validation")
            }),
            ("final scope receipt", |v| {
                v["portfolio"]["final_holdout_scope"]["receipt_sha256"] =
                    serde_json::json!("0".repeat(64))
            }),
            ("receipt payload", |v| {
                v["input_receipt"]["feature_content_sha256"] = serde_json::json!("0".repeat(64))
            }),
            ("receipt proof", |v| {
                let old = v["input_receipt"]["feature_plan_canonical_bytes"][0]
                    .as_u64()
                    .unwrap();
                v["input_receipt"]["feature_plan_canonical_bytes"][0] = serde_json::json!(old ^ 1)
            }),
            ("selection role", |v| {
                v["portfolio"]["search_scope"]["evaluated_window"]["role"] =
                    serde_json::json!("holdout")
            }),
            ("holdout role", |v| {
                v["portfolio"]["sizing_evidence"][0]["scope"]["evaluated_window"]["role"] =
                    serde_json::json!("in_sample")
            }),
            ("holdout time", |v| {
                v["portfolio"]["sizing_evidence"][0]["scope"]["evaluated_window"]["timestamp_start_ms"] =
                    serde_json::json!(0)
            }),
            ("receipt reference", |v| {
                v["portfolio"]["sizing_evidence"][0]["scope"]["receipt_sha256"] =
                    serde_json::json!("0".repeat(64))
            }),
            ("strategy identity", |v| {
                v["portfolio"]["sizing_evidence"][0]["strategy_identity"]["exact_gene_hash"] =
                    serde_json::json!("fnv64:0000000000000000")
            }),
            ("summary", |v| {
                v["portfolio"]["sizing_evidence"][0]["summary"]["metrics"]["net_profit"] =
                    serde_json::json!(9000.0)
            }),
            ("cost band", |v| {
                v["portfolio"]["cost_band"][0][1] = serde_json::json!("cost_band_survives")
            }),
            ("unknown body field", |v| {
                v["portfolio"]["unbound_semantics"] = serde_json::json!(true)
            }),
            ("body version", |v| {
                v["portfolio"]["schema_version"] = serde_json::json!(2)
            }),
            ("envelope version", |v| {
                v["shared_receipt_schema_version"] = serde_json::json!(2)
            }),
            ("envelope kind", |v| {
                v["artifact_kind"] = serde_json::json!("another.artifact.v1")
            }),
        ];
        for (label, mutate) in mutations {
            let mut changed = value.clone();
            mutate(&mut changed);
            assert!(
                LivePortfolioArtifact::from_persisted_json_bytes(
                    &serde_json::to_vec(&changed).unwrap()
                )
                .is_err(),
                "accepted changed {label}"
            );
        }
        let json = String::from_utf8(encoded).unwrap();
        let duplicate = json.replacen('{', "{\"shared_receipt_schema_version\":1,", 1);
        assert!(LivePortfolioArtifact::from_persisted_json_bytes(duplicate.as_bytes()).is_err());
        let missing = json.replacen("\"shared_receipt_schema_version\":1,", "", 1);
        assert!(
            LivePortfolioArtifact::from_persisted_json_bytes(missing.as_bytes()).is_err(),
            "new envelopes cannot fall back to legacy parsing when their discriminator is removed"
        );
    }

    #[test]
    fn shared_portfolio_file_roundtrip_uses_one_embedded_receipt() {
        let result = sample_discovery_result();
        let expected = LivePortfolioArtifact::from_discovery(false, &result).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("live_portfolio.json");
        save_live_portfolio_json(&path, &result).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            String::from_utf8_lossy(&bytes)
                .matches("\"feature_plan_canonical_bytes\"")
                .count(),
            1
        );
        assert!(String::from_utf8_lossy(&bytes).contains("shared_receipt_schema_version"));
        let loaded = load_live_portfolio_json(&path).unwrap();
        assert_eq!(
            serde_json::to_vec(&loaded).unwrap(),
            serde_json::to_vec(&expected).unwrap()
        );
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

    fn sample_discovery_authority(
        features: &FeatureFrame,
    ) -> (
        crate::data_selection::CanonicalSearchInputReceiptV2,
        CanonicalSearchArtifactScopeV2,
        CanonicalSearchArtifactScopeV2,
        CanonicalSearchArtifactScopeV2,
    ) {
        let ohlcv = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
        let anchor = features.provenance().bindings()[0].dataset_identity();
        let receipt = crate::data_selection::CanonicalSearchInputReceiptV2::from_feature_frame(
            anchor, features,
        )
        .expect("canonical search test receipt");
        let input = crate::data_selection::CanonicalSearchRunInputV2::new_for_test_values(
            receipt.clone(),
            features,
            &ohlcv,
        )
        .expect("canonical live test input");
        let selection_scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
            CanonicalSearchWindowRoleV1::InSample,
            &input,
            0..80,
        )
        .expect("canonical InSample live test scope");
        let calibration_scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
            CanonicalSearchWindowRoleV1::SelectionValidation,
            &input,
            80..90,
        )
        .expect("canonical calibration live test scope");
        let holdout_scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
            CanonicalSearchWindowRoleV1::Holdout,
            &input,
            90..100,
        )
        .expect("canonical Holdout live test scope");
        (receipt, selection_scope, calibration_scope, holdout_scope)
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
        let artifact = artifact_shape_for_contract_test(false, &result);

        let json = serde_json::to_string(&artifact).unwrap();
        let back: LivePortfolioArtifact = serde_json::from_str(&json).unwrap();
        back.validate().unwrap();
        assert_eq!(
            serde_json::to_value(&artifact).unwrap(),
            serde_json::to_value(&back).unwrap(),
            "every artifact field, including OOS evidence, must survive a JSON round-trip"
        );
        assert_eq!(
            back.live_trading_policy.exit_policy(),
            neoethos_core::config::ExitPolicyConfig {
                trailing_enabled: true,
                trailing_be_trigger_r: 1.25,
                trailing_stop_multiplier: 0.75,
                trailing_min_lock_pips: 3.0,
            },
            "the exact exit policy priced by discovery must survive the artifact round-trip"
        );
        assert!(!back.live_trading_policy.kill_zones_enabled);
        assert_eq!(
            back.live_trading_policy
                .expected_spread_pips_at(8 * 3_600_000),
            0.6
        );
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
        let mut value = serde_json::to_value(valid_v6_artifact()).expect("serialize valid v6");
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
        let mut value = serde_json::to_value(valid_v6_artifact()).expect("serialize valid v6");
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
    fn from_discovery_mints_demo_artifact_from_complete_bar_oos_evidence() {
        let result = sample_discovery_result();
        let artifact = LivePortfolioArtifact::from_discovery(false, &result)
            .expect("complete canonical bar OOS evidence must mint a demo-runnable artifact");
        assert_eq!(artifact.genes.len(), 1);
        assert_eq!(artifact.sizing_evidence.len(), 1);
        assert!(artifact.portfolio_half_kelly_risk_fraction().unwrap() > 0.0);
    }

    #[test]
    fn v6_separates_training_calibration_and_reserved_final_without_claiming_a_test_pass() {
        let result = sample_discovery_result();
        let artifact = LivePortfolioArtifact::from_discovery(false, &result).unwrap();
        let training = artifact.search_scope.evaluated_window();
        let calibration = artifact.sizing_evidence[0]
            .forward_test
            .scope()
            .evaluated_window();
        let final_window = artifact.final_holdout_scope.evaluated_window();
        assert_eq!(artifact.schema_version, 6);
        assert_eq!((training.row_start(), training.row_end()), (0, 80));
        assert_eq!((calibration.row_start(), calibration.row_end()), (80, 90));
        assert_eq!(
            (final_window.row_start(), final_window.row_end()),
            (90, 100)
        );
        assert_eq!(
            calibration.role(),
            CanonicalSearchWindowRoleV1::SelectionValidation
        );
        assert_eq!(final_window.role(), CanonicalSearchWindowRoleV1::Holdout);
        assert_eq!(artifact.final_holdout_scope, result.holdout_scope.unwrap());
        let mut legacy_result = sample_discovery_result();
        legacy_result.calibration_scope = None;
        assert!(LivePortfolioArtifact::from_discovery(false, &legacy_result).is_err());
        let mut legacy = serde_json::to_value(&artifact).unwrap();
        legacy["schema_version"] = 5.into();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("final_holdout_scope");
        assert!(
            LivePortfolioArtifact::from_persisted_json_bytes(&serde_json::to_vec(&legacy).unwrap())
                .is_err()
        );
        let mut disguised = serde_json::to_value(&artifact).unwrap();
        disguised["schema_version"] = 5.into();
        assert!(
            LivePortfolioArtifact::from_persisted_json_bytes(
                &serde_json::to_vec(&disguised).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn v6_refuses_final_scope_overlap_gaps_and_calibration_evidence_on_final_rows() {
        let original = valid_v6_artifact();
        for start in [89, 91] {
            let mut changed = serde_json::to_value(&original).unwrap();
            changed["final_holdout_scope"]["evaluated_window"]["row_start"] = start.into();
            let changed: LivePortfolioArtifact = serde_json::from_value(changed).unwrap();
            assert!(
                changed.validate().is_err(),
                "accepted final row start {start}"
            );
        }
        let mut changed = original.clone();
        changed.final_holdout_scope = changed.sizing_evidence[0].forward_test.scope().clone();
        assert!(changed.validate().is_err());
        let mut result = sample_discovery_result();
        let source = &result.forward_test_validation_artifacts[0];
        result.forward_test_validation_artifacts[0] =
            crate::validation::ForwardTestValidationArtifactFile::new(
                result.holdout_scope.clone().unwrap(),
                source.search_config_hash(),
                &result.portfolio[0],
                source.summary().clone(),
            )
            .unwrap();
        assert!(LivePortfolioArtifact::from_discovery(false, &result).is_err());
    }

    #[test]
    fn normalized_legacy_artifact_cannot_silently_become_a_raw_portfolio() {
        let mut artifact = valid_v6_artifact();
        artifact.normalize_features = true;
        let decoded: LivePortfolioArtifact =
            serde_json::from_slice(&serde_json::to_vec(&artifact).unwrap()).unwrap();
        let error = decoded.validate().unwrap_err();
        assert!(error.to_string().contains("persisted training fit"));
    }

    #[test]
    fn discovery_cannot_relabel_a_raw_receipt_using_an_ambient_normalize_flag() {
        let result = sample_discovery_result();
        let error = LivePortfolioArtifact::from_discovery(true, &result).unwrap_err();
        assert!(error.to_string().contains("normalization flag"));
    }

    #[test]
    fn raw_live_projection_preserves_the_exact_values_and_validity() {
        let artifact = valid_v6_artifact();
        let raw = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let projected = artifact.project_live_features(&raw).unwrap();
        assert_eq!(projected.names, artifact.effective_feature_names);
        let source_column = raw
            .names
            .iter()
            .position(|name| name == &artifact.effective_feature_names[0])
            .unwrap();
        for row in 0..raw.n_samples() {
            let expected = raw.cell(row, source_column).unwrap();
            let actual = projected.cell(row, 0).unwrap();
            assert_eq!(actual.validity, expected.validity);
            assert_eq!(actual.value.to_bits(), expected.value.to_bits());
        }
    }

    #[test]
    fn normalized_portfolio_roundtrip_uses_saved_fit_and_rejects_raw_or_refitted_inputs() {
        use neoethos_data::test_fixtures::{
            ctrader_sample_feature_frame, ctrader_test_feature_frame_with_normalization,
        };
        let raw = ctrader_sample_feature_frame();
        let fitted = ctrader_test_feature_frame_with_normalization(&raw, 0..80, None).unwrap();
        let genes = sample_discovery_result().portfolio;
        let result = sample_discovery_result_for_features(genes, &fitted);
        let artifact = LivePortfolioArtifact::from_discovery(true, &result).unwrap();
        let encoded = serde_json::to_vec(&artifact).unwrap();
        let decoded: LivePortfolioArtifact = serde_json::from_slice(&encoded).unwrap();
        decoded.validate().unwrap();
        let saved = decoded
            .search_scope
            .receipt()
            .normalization_fitted_state()
            .unwrap();
        assert_eq!(saved, fitted.normalization_fitted_state().unwrap());
        // Simulate a short unseen tail; its 20 rows cannot fit the original
        // 80-row training scope. The only allowed operation is replay.
        let tail = raw.row_window(80, 100).unwrap();
        let live =
            ctrader_test_feature_frame_with_normalization(&tail, 0..80, Some(saved)).unwrap();
        let projected = decoded.project_live_features(&live).unwrap();
        for row in 0..20 {
            let actual = projected.cell(row, 0).unwrap();
            let expected = fitted.cell(row + 80, 0).unwrap();
            assert_eq!(actual.validity, expected.validity);
            assert_eq!(actual.value.to_bits(), expected.value.to_bits());
        }
        assert!(
            decoded.project_live_features(&tail).is_err(),
            "raw values must not pass as normalized"
        );
        let refitted = ctrader_test_feature_frame_with_normalization(&tail, 0..20, None).unwrap();
        assert!(
            decoded.project_live_features(&refitted).is_err(),
            "live refitting must not substitute another fit"
        );
        let raw_artifact = valid_v6_artifact();
        assert!(
            raw_artifact.project_live_features(&live).is_err(),
            "raw artifact must not accept normalized inputs"
        );
    }

    #[test]
    fn normalized_portfolio_refuses_a_fit_that_consumed_its_holdout() {
        let raw = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let fitted = neoethos_data::test_fixtures::ctrader_test_feature_frame_with_normalization(
            &raw,
            0..100,
            None,
        )
        .unwrap();
        let result =
            sample_discovery_result_for_features(sample_discovery_result().portfolio, &fitted);
        let error = LivePortfolioArtifact::from_discovery(true, &result).unwrap_err();
        assert!(error.to_string().contains("held-out"), "{error}");
    }

    #[test]
    fn live_sizing_persists_full_holdout_evidence_and_does_not_use_training_metrics() {
        let mut result = sample_discovery_result();
        let source = result.forward_test_validation_artifacts[0].clone();
        let mut summary = source.summary().clone();
        summary.metrics.win_rate = 0.4;
        summary.metrics.profit_factor = 1.2;
        result.forward_test_validation_artifacts[0] =
            crate::validation::ForwardTestValidationArtifactFile::new(
                source.scope().clone(),
                source.search_config_hash(),
                &result.portfolio[0],
                summary,
            )
            .unwrap();
        let artifact = LivePortfolioArtifact::from_discovery(false, &result).unwrap();
        // Independent arithmetic: .5 * .4 * (1.2 - 1) / 1.2 = 1/30.
        assert!(
            (artifact.portfolio_half_kelly_risk_fraction().unwrap() - 1.0 / 30.0).abs() < 1e-12
        );
        let decoded: LivePortfolioArtifact =
            serde_json::from_slice(&serde_json::to_vec(&artifact).unwrap()).unwrap();
        decoded.validate().unwrap();
        let evidence = &decoded.sizing_evidence[0].forward_test;
        assert_eq!(evidence.scope(), result.calibration_scope.as_ref().unwrap());
        assert_eq!(evidence.search_config_hash(), result.search_config_hash);
        assert_eq!(evidence.summary().metrics.win_rate, 0.4);
    }

    #[test]
    fn runnable_portfolio_requires_passed_selection_gates_not_just_present_files() {
        for failed_gate in ["walkforward", "cpcv", "pbo"] {
            let mut result = sample_discovery_result();
            match failed_gate {
                "walkforward" => result.validation_gates.walkforward_passed = false,
                "cpcv" => result.validation_gates.cpcv_passed = false,
                _ => result.validation_gates.pbo_passed = false,
            }
            let error = LivePortfolioArtifact::from_discovery(false, &result).unwrap_err();
            assert!(
                error.to_string().contains("requires passed"),
                "{failed_gate}: {error}"
            );
        }
    }

    #[test]
    fn live_sizing_refuses_missing_evidence_and_a_modified_genome() {
        let mut artifact = valid_v6_artifact();
        artifact.genes[0].long_threshold += 0.01;
        assert!(
            artifact
                .validate()
                .unwrap_err()
                .to_string()
                .contains("exact_gene_hash")
        );
        let mut artifact = valid_v6_artifact();
        artifact.sizing_evidence.clear();
        assert!(
            artifact
                .validate()
                .unwrap_err()
                .to_string()
                .contains("sizing-evidence")
        );
    }

    #[test]
    fn calibration_metrics_gate_retains_all_existing_numerical_sizing_checks() {
        let valid = crate::eval::BacktestMetrics::from_metric_array([
            20.0, 1.0, 10_020.0, 0.01, 0.6, 1.5, 2.0, 0.5, 10.0, 0.8, 0.005,
        ]);
        LiveSizingEvidenceV1::validate_calibration_metrics("positive", &valid).unwrap();
        for field in 0..10 {
            let mut metrics = valid;
            match field {
                0 => metrics.net_profit = 0.0,
                1 => metrics.expectancy = 0.0,
                2 => metrics.trade_count = 0,
                3 => metrics.win_rate = 0.0,
                4 => metrics.win_rate = 1.1,
                5 => metrics.profit_factor = 1.0,
                6 => metrics.sharpe = f64::NEG_INFINITY,
                7 => metrics.max_drawdown = f64::NAN,
                8 => metrics.consistency = f64::INFINITY,
                _ => metrics.max_daily_drawdown = f64::NAN,
            }
            assert!(
                LiveSizingEvidenceV1::validate_calibration_metrics("invalid", &metrics).is_err(),
                "accepted case {field}"
            );
        }
    }

    #[test]
    fn live_sizing_refuses_evidence_from_another_search_configuration() {
        let mut artifact = valid_v6_artifact();
        let source = artifact.sizing_evidence[0].forward_test.clone();
        artifact.sizing_evidence[0].forward_test =
            crate::validation::ForwardTestValidationArtifactFile::new(
                source.scope().clone(),
                "fnv64:ffffffffffffffff",
                &artifact.genes[0],
                source.summary().clone(),
            )
            .unwrap();
        assert!(
            artifact
                .validate()
                .unwrap_err()
                .to_string()
                .contains("this search config")
        );
    }

    #[test]
    fn live_broker_binding_rejects_an_external_research_receipt() {
        let artifact = valid_v6_artifact();
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
    fn load_rejects_v3_without_a_pinned_live_trading_policy() {
        let mut value = serde_json::to_value(valid_v6_artifact()).expect("serialize valid v6");
        let object = value.as_object_mut().expect("artifact is an object");
        object.insert("schema_version".to_string(), serde_json::json!(3));
        object.remove("live_trading_policy");
        let path = write_test_json("reject_v3_without_policy", &value);

        let error = load_live_portfolio_json(&path)
            .expect_err("v3 reconstructed live exits from mutable Settings and must fail closed");
        assert!(
            error.to_string().contains("schema")
                || error.to_string().contains("live_trading_policy"),
            "failure should name the obsolete schema or missing policy: {error}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn policy_value_tampering_breaks_its_identity() {
        let mut artifact = valid_v6_artifact();
        artifact.live_trading_policy.trailing_be_trigger_r = 9.0;
        let error = artifact
            .validate()
            .expect_err("changed exit geometry must not retain authority");
        assert!(error.to_string().contains("identity hash mismatch"));
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
                bars: 10,
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
            .calibration_scope
            .as_ref()
            .expect("strict live fixture holdout")
            .clone();
        result.forward_test_validation_artifacts = vec![
            artifact(&winner, &holdout_scope, 42.0),
            artifact(&loser, &holdout_scope, -3.0),
        ];

        let survivors = filter_oos_survivors_from_validated_diagnostics(&result)
            .expect("validated diagnostic filter");
        assert_eq!(
            survivors
                .iter()
                .map(|g| g.strategy_id.as_str())
                .collect::<Vec<_>>(),
            vec!["winner"],
            "only the tail-profitable strategy survives the OOS filter"
        );

        let promoted = LivePortfolioArtifact::from_discovery(false, &result)
            .expect("complete bar OOS evidence must be runnable on demo");
        assert_eq!(
            promoted
                .genes
                .iter()
                .map(|gene| gene.strategy_id.as_str())
                .collect::<Vec<_>>(),
            vec!["winner"]
        );
        assert_eq!(promoted.sizing_evidence.len(), 1);

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
