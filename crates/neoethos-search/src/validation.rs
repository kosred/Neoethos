use crate::artifact_io::{read_json, stable_json_hash, write_json_atomic};
use crate::eval::{
    BacktestMetrics, BacktestSettings, fast_evaluate_strategy_core, simulate_trades_core,
};
use anyhow::{Result, bail};
use itertools::Itertools;
use rayon::prelude::*;
use neoethos_core::contracts::{TemporalFeatureContract, TemporalScopeHashes};
use neoethos_core::domain::prop_firm::{PropFirmChallengeDefaults, PropFirmConstraints};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkforwardSplitResult {
    pub split: usize,
    pub trades: usize,
    pub pnl: f64,
    pub win_rate: f64,
    pub max_dd: f64,
    pub max_consec_losses: usize,
    pub daily_min_dd: f64,
    pub max_daily_loss: f64,
    pub daily_loss_breach: bool,
    pub consistency_violation: bool,
    pub trade_limit_violation: bool,
    pub min_trading_days_ok: bool,
    pub daily_returns: Vec<f64>,
    pub max_daily_dd_pct: f64,
    pub prop_compliant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkforwardSummary {
    pub walk_forward_splits: usize,
    pub avg_pnl: f64,
    pub avg_win_rate: f64,
    pub avg_max_dd: f64,
    pub avg_max_consec_losses: f64,
    pub avg_daily_min_dd: f64,
    pub avg_max_daily_loss: f64,
    pub any_daily_loss_breach: bool,
    pub any_consistency_violation: bool,
    pub any_trade_limit_violation: bool,
    pub all_min_trading_days_ok: bool,
    pub splits: Vec<WalkforwardSplitResult>,
}

pub const WALKFORWARD_VALIDATION_ARTIFACT_KIND: &str = "walkforward_validation_artifact";
pub const WALKFORWARD_VALIDATION_SCHEMA_VERSION: u32 = 1;
pub const CANONICAL_BACKTEST_ARTIFACT_KIND: &str = "canonical_strategy_backtest_artifact";
pub const CANONICAL_BACKTEST_SCHEMA_VERSION: u32 = 1;
pub const FORWARD_TEST_VALIDATION_ARTIFACT_KIND: &str = "forward_test_validation_artifact";
pub const FORWARD_TEST_VALIDATION_SCHEMA_VERSION: u32 = 1;
pub const LIVE_EXECUTION_SIMULATION_ARTIFACT_KIND: &str = "live_execution_simulation_artifact";
pub const LIVE_EXECUTION_SIMULATION_SCHEMA_VERSION: u32 = 1;
pub const PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND: &str = "prop_firm_risk_validation_artifact";
pub const PROP_FIRM_RISK_VALIDATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalBacktestScope {
    pub dataset_hash: String,
    pub evaluation_config_hash: String,
    pub strategy_hash: String,
    pub temporal_scope: TemporalScopeHashes,
}

impl CanonicalBacktestScope {
    pub fn new(
        dataset_hash: impl Into<String>,
        evaluation_config_hash: impl Into<String>,
        strategy_hash: impl Into<String>,
        temporal_contract: &TemporalFeatureContract,
    ) -> Self {
        Self {
            dataset_hash: dataset_hash.into(),
            evaluation_config_hash: evaluation_config_hash.into(),
            strategy_hash: strategy_hash.into(),
            temporal_scope: TemporalScopeHashes::from_contract(temporal_contract),
        }
    }

    pub fn from_parts<T: Serialize, U: Serialize, V: Serialize>(
        dataset: &T,
        evaluation_config: &U,
        strategy: &V,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<Self> {
        Ok(Self::new(
            stable_json_hash(dataset)?,
            stable_json_hash(evaluation_config)?,
            stable_json_hash(strategy)?,
            temporal_contract,
        ))
    }

    pub fn validate_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        self.temporal_scope
            .validate_contract(temporal_contract)
            .map_err(|err| anyhow::anyhow!("canonical backtest {err}"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalBacktestArtifactFile {
    pub artifact_kind: String,
    pub artifact_schema_version: u32,
    pub scope: CanonicalBacktestScope,
    pub metrics: BacktestMetrics,
}

impl CanonicalBacktestArtifactFile {
    pub fn new(scope: CanonicalBacktestScope, metrics: BacktestMetrics) -> Self {
        Self {
            artifact_kind: CANONICAL_BACKTEST_ARTIFACT_KIND.to_string(),
            artifact_schema_version: CANONICAL_BACKTEST_SCHEMA_VERSION,
            scope,
            metrics,
        }
    }

    pub fn validate_for_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        if self.artifact_kind != CANONICAL_BACKTEST_ARTIFACT_KIND {
            bail!(
                "artifact kind {} cannot be used as a canonical backtest artifact",
                self.artifact_kind
            );
        }
        if self.artifact_schema_version != CANONICAL_BACKTEST_SCHEMA_VERSION {
            bail!(
                "unsupported canonical backtest schema version {}",
                self.artifact_schema_version
            );
        }
        self.scope.validate_temporal_contract(temporal_contract)
    }
}

pub fn write_canonical_backtest_artifact_atomic(
    path: impl AsRef<Path>,
    artifact: &CanonicalBacktestArtifactFile,
) -> Result<()> {
    write_json_atomic(path, artifact)
}

pub fn read_canonical_backtest_artifact(
    path: impl AsRef<Path>,
    temporal_contract: &TemporalFeatureContract,
) -> Result<CanonicalBacktestArtifactFile> {
    let artifact: CanonicalBacktestArtifactFile = read_json(path, "canonical backtest")?;
    artifact.validate_for_temporal_contract(temporal_contract)?;
    Ok(artifact)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WalkforwardValidationScope {
    pub dataset_hash: String,
    pub evaluation_config_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy_hash: Option<String>,
    pub temporal_scope: TemporalScopeHashes,
}

impl WalkforwardValidationScope {
    pub fn new(
        dataset_hash: impl Into<String>,
        evaluation_config_hash: impl Into<String>,
        temporal_contract: &TemporalFeatureContract,
    ) -> Self {
        Self {
            dataset_hash: dataset_hash.into(),
            evaluation_config_hash: evaluation_config_hash.into(),
            strategy_hash: None,
            temporal_scope: TemporalScopeHashes::from_contract(temporal_contract),
        }
    }

    pub fn for_strategy(
        dataset_hash: impl Into<String>,
        evaluation_config_hash: impl Into<String>,
        strategy_hash: impl Into<String>,
        temporal_contract: &TemporalFeatureContract,
    ) -> Self {
        Self {
            dataset_hash: dataset_hash.into(),
            evaluation_config_hash: evaluation_config_hash.into(),
            strategy_hash: Some(strategy_hash.into()),
            temporal_scope: TemporalScopeHashes::from_contract(temporal_contract),
        }
    }

    pub fn from_parts<T: Serialize, U: Serialize>(
        dataset: &T,
        evaluation_config: &U,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<Self> {
        Ok(Self::new(
            stable_json_hash(dataset)?,
            stable_json_hash(evaluation_config)?,
            temporal_contract,
        ))
    }

    pub fn validate_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        self.temporal_scope
            .validate_contract(temporal_contract)
            .map_err(|err| anyhow::anyhow!("walk-forward validation {err}"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkforwardValidationArtifactFile {
    pub artifact_kind: String,
    pub artifact_schema_version: u32,
    pub scope: WalkforwardValidationScope,
    pub summary: WalkforwardSummary,
}

impl WalkforwardValidationArtifactFile {
    pub fn new(scope: WalkforwardValidationScope, summary: WalkforwardSummary) -> Self {
        Self {
            artifact_kind: WALKFORWARD_VALIDATION_ARTIFACT_KIND.to_string(),
            artifact_schema_version: WALKFORWARD_VALIDATION_SCHEMA_VERSION,
            scope,
            summary,
        }
    }

    pub fn validate_for_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        if self.artifact_kind != WALKFORWARD_VALIDATION_ARTIFACT_KIND {
            bail!(
                "artifact kind {} cannot be used as a walk-forward validation artifact",
                self.artifact_kind
            );
        }
        if self.artifact_schema_version != WALKFORWARD_VALIDATION_SCHEMA_VERSION {
            bail!(
                "unsupported walk-forward validation schema version {}",
                self.artifact_schema_version
            );
        }
        self.scope.validate_temporal_contract(temporal_contract)
    }
}

pub fn write_walkforward_validation_artifact_atomic(
    path: impl AsRef<Path>,
    artifact: &WalkforwardValidationArtifactFile,
) -> Result<()> {
    write_json_atomic(path, artifact)
}

pub fn read_walkforward_validation_artifact(
    path: impl AsRef<Path>,
    temporal_contract: &TemporalFeatureContract,
) -> Result<WalkforwardValidationArtifactFile> {
    let artifact: WalkforwardValidationArtifactFile = read_json(path, "walk-forward validation")?;
    artifact.validate_for_temporal_contract(temporal_contract)?;
    Ok(artifact)
}

/// Forward-test validation summary: a single backtest pass over a tail
/// window that was withheld from both training and walk-forward CV. The
/// summary is intentionally flat (no `splits`) because forward testing
/// produces one unbiased OOS estimate, not a folded distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardTestSummary {
    /// Number of bars in the held-out tail window.
    pub bars: usize,
    /// Canonical metrics computed on the held-out tail.
    pub metrics: BacktestMetrics,
    /// Wall-clock span of the tail window in days (`exit_time - entry_time`
    /// of the first/last bar). `0.0` when the tail has fewer than two bars.
    pub span_days: f64,
}

/// Forward-test validation scope. The dataset hash binds the *tail*
/// dataset (not the full discovery dataset) so the artifact cannot be
/// confused with a canonical backtest produced from in-sample data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ForwardTestValidationScope {
    pub dataset_hash: String,
    pub evaluation_config_hash: String,
    pub strategy_hash: String,
    pub temporal_scope: TemporalScopeHashes,
}

impl ForwardTestValidationScope {
    pub fn new(
        dataset_hash: impl Into<String>,
        evaluation_config_hash: impl Into<String>,
        strategy_hash: impl Into<String>,
        temporal_contract: &TemporalFeatureContract,
    ) -> Self {
        Self {
            dataset_hash: dataset_hash.into(),
            evaluation_config_hash: evaluation_config_hash.into(),
            strategy_hash: strategy_hash.into(),
            temporal_scope: TemporalScopeHashes::from_contract(temporal_contract),
        }
    }

    pub fn from_parts<T: Serialize, U: Serialize, V: Serialize>(
        dataset: &T,
        evaluation_config: &U,
        strategy: &V,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<Self> {
        Ok(Self::new(
            stable_json_hash(dataset)?,
            stable_json_hash(evaluation_config)?,
            stable_json_hash(strategy)?,
            temporal_contract,
        ))
    }

    pub fn validate_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        self.temporal_scope
            .validate_contract(temporal_contract)
            .map_err(|err| anyhow::anyhow!("forward test {err}"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardTestValidationArtifactFile {
    pub artifact_kind: String,
    pub artifact_schema_version: u32,
    pub scope: ForwardTestValidationScope,
    pub summary: ForwardTestSummary,
}

impl ForwardTestValidationArtifactFile {
    pub fn new(scope: ForwardTestValidationScope, summary: ForwardTestSummary) -> Self {
        Self {
            artifact_kind: FORWARD_TEST_VALIDATION_ARTIFACT_KIND.to_string(),
            artifact_schema_version: FORWARD_TEST_VALIDATION_SCHEMA_VERSION,
            scope,
            summary,
        }
    }

    pub fn validate_for_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        if self.artifact_kind != FORWARD_TEST_VALIDATION_ARTIFACT_KIND {
            bail!(
                "artifact kind {} cannot be used as a forward-test validation artifact",
                self.artifact_kind
            );
        }
        if self.artifact_schema_version != FORWARD_TEST_VALIDATION_SCHEMA_VERSION {
            bail!(
                "unsupported forward-test validation schema version {}",
                self.artifact_schema_version
            );
        }
        self.scope.validate_temporal_contract(temporal_contract)
    }
}

pub fn write_forward_test_validation_artifact_atomic(
    path: impl AsRef<Path>,
    artifact: &ForwardTestValidationArtifactFile,
) -> Result<()> {
    write_json_atomic(path, artifact)
}

pub fn read_forward_test_validation_artifact(
    path: impl AsRef<Path>,
    temporal_contract: &TemporalFeatureContract,
) -> Result<ForwardTestValidationArtifactFile> {
    let artifact: ForwardTestValidationArtifactFile = read_json(path, "forward-test validation")?;
    artifact.validate_for_temporal_contract(temporal_contract)?;
    Ok(artifact)
}

/// Inputs for [`compute_forward_test_summary`] — a single tail-window
/// replay using the same evaluation core as canonical backtests.
pub struct ForwardTestInput<'a> {
    pub close: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub signals: &'a [i8],
    pub months: &'a [i64],
    pub days: &'a [i64],
    pub timestamps: &'a [i64],
    pub settings: &'a BacktestSettings,
}

/// Run a single canonical backtest pass over the held-out tail and
/// package the result as a [`ForwardTestSummary`]. Callers are responsible
/// for slicing `close`/`high`/`low`/`signals`/`months`/`days`/`timestamps`
/// to the tail window before calling this helper — the function does no
/// internal partitioning.
pub fn compute_forward_test_summary(input: ForwardTestInput<'_>) -> Result<ForwardTestSummary> {
    let bars = input.close.len();
    if bars == 0 {
        bail!("forward-test tail must contain at least one bar");
    }
    if input.high.len() != bars
        || input.low.len() != bars
        || input.signals.len() != bars
        || input.months.len() != bars
        || input.days.len() != bars
    {
        bail!("forward-test tail length mismatch across input arrays");
    }
    let timestamps_len = input.timestamps.len();
    if timestamps_len != 0 && timestamps_len != bars {
        bail!("forward-test timestamps must be empty or match the tail length");
    }
    let metrics = BacktestMetrics::from_metric_array(fast_evaluate_strategy_core(
        input.close,
        input.high,
        input.low,
        input.signals,
        // Phase 1: legacy fixed-1-lot for the forward-test summary (no
        // confidence threaded here yet) — `&[]` forces pos_lots = 1.0.
        &[],
        input.months,
        input.days,
        input.timestamps,
        input.settings,
    ));
    let span_days = if timestamps_len >= 2 {
        let first = input.timestamps[0];
        let last = input.timestamps[timestamps_len - 1];
        let delta = (last - first) as f64;
        if delta > 0.0 {
            // `simulate_trades_core` accepts ms timestamps; convert to
            // days so the artifact is self-describing without leaking the
            // unit assumption.
            delta / 86_400_000.0
        } else {
            0.0
        }
    } else {
        0.0
    };
    Ok(ForwardTestSummary {
        bars,
        metrics,
        span_days,
    })
}

/// Runtime model used by a live-execution simulation. The artifact
/// records which slippage / latency / spread / commission assumptions
/// produced the metrics so a downstream live bridge can reject artifacts
/// whose execution semantics do not match its current configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LiveExecutionRuntimeModel {
    pub avg_slippage_pips: f64,
    pub avg_latency_ms: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub partial_fill_rate: f64,
    pub kill_zone_blocking: bool,
    pub backend_kind: String,
}

/// Live-execution simulation summary — canonical metrics under live-like
/// execution assumptions, plus the simulator-observed counters that
/// distinguish a live-sim from a canonical backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveExecutionSimulationSummary {
    pub bars_simulated: usize,
    pub trades_simulated: usize,
    pub trades_blocked_by_kill_zone: usize,
    pub trades_partially_filled: usize,
    pub metrics: BacktestMetrics,
    pub runtime_model: LiveExecutionRuntimeModel,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LiveExecutionSimulationScope {
    pub dataset_hash: String,
    pub evaluation_config_hash: String,
    pub strategy_hash: String,
    pub runtime_model_hash: String,
    pub temporal_scope: TemporalScopeHashes,
}

impl LiveExecutionSimulationScope {
    pub fn new(
        dataset_hash: impl Into<String>,
        evaluation_config_hash: impl Into<String>,
        strategy_hash: impl Into<String>,
        runtime_model: &LiveExecutionRuntimeModel,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<Self> {
        Ok(Self {
            dataset_hash: dataset_hash.into(),
            evaluation_config_hash: evaluation_config_hash.into(),
            strategy_hash: strategy_hash.into(),
            runtime_model_hash: stable_json_hash(runtime_model)?,
            temporal_scope: TemporalScopeHashes::from_contract(temporal_contract),
        })
    }

    pub fn validate_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        self.temporal_scope
            .validate_contract(temporal_contract)
            .map_err(|err| anyhow::anyhow!("live execution simulation {err}"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveExecutionSimulationArtifactFile {
    pub artifact_kind: String,
    pub artifact_schema_version: u32,
    pub scope: LiveExecutionSimulationScope,
    pub summary: LiveExecutionSimulationSummary,
}

impl LiveExecutionSimulationArtifactFile {
    pub fn new(
        scope: LiveExecutionSimulationScope,
        summary: LiveExecutionSimulationSummary,
    ) -> Self {
        Self {
            artifact_kind: LIVE_EXECUTION_SIMULATION_ARTIFACT_KIND.to_string(),
            artifact_schema_version: LIVE_EXECUTION_SIMULATION_SCHEMA_VERSION,
            scope,
            summary,
        }
    }

    pub fn validate_for_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        if self.artifact_kind != LIVE_EXECUTION_SIMULATION_ARTIFACT_KIND {
            bail!(
                "artifact kind {} cannot be used as a live execution simulation artifact",
                self.artifact_kind
            );
        }
        if self.artifact_schema_version != LIVE_EXECUTION_SIMULATION_SCHEMA_VERSION {
            bail!(
                "unsupported live execution simulation schema version {}",
                self.artifact_schema_version
            );
        }
        self.scope.validate_temporal_contract(temporal_contract)
    }
}

pub fn write_live_execution_simulation_artifact_atomic(
    path: impl AsRef<Path>,
    artifact: &LiveExecutionSimulationArtifactFile,
) -> Result<()> {
    write_json_atomic(path, artifact)
}

pub fn read_live_execution_simulation_artifact(
    path: impl AsRef<Path>,
    temporal_contract: &TemporalFeatureContract,
) -> Result<LiveExecutionSimulationArtifactFile> {
    let artifact: LiveExecutionSimulationArtifactFile =
        read_json(path, "live execution simulation")?;
    artifact.validate_for_temporal_contract(temporal_contract)?;
    Ok(artifact)
}

/// Prop-firm rule set applied to observed trade outcomes. Each numeric
/// field is a pass threshold (`<= 0.0` means "rule disabled" so callers
/// can opt out per-field); booleans toggle structural rules.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct PropFirmRiskRules {
    pub max_daily_loss_pct: f64,
    pub max_overall_drawdown_pct: f64,
    pub max_profit_consistency_ratio: f64,
    pub min_trading_days: usize,
    pub max_trades_per_day: usize,
    pub require_profit_target: bool,
    pub min_profit_target_pct: f64,
}

impl Default for PropFirmRiskRules {
    fn default() -> Self {
        // FTMO-style baseline; callers should override per challenge.
        // Numeric defaults come from `PropFirmConstraints::FTMO_STANDARD`
        // per operator directive 2026-05-14 — they are the only
        // hardcoded prop-firm numbers allowed in production code.
        let ftmo = PropFirmConstraints::FTMO_STANDARD;
        let challenge_defaults = PropFirmChallengeDefaults::FTMO_STANDARD;
        Self {
            max_daily_loss_pct: ftmo.max_daily_loss_pct as f64,
            max_overall_drawdown_pct: ftmo.max_overall_drawdown_pct as f64,
            // FIXME(hardcoded): config-extract — internal consistency-ratio cap.
            max_profit_consistency_ratio: 0.50,
            min_trading_days: challenge_defaults.relaxed_min_trading_days as usize,
            max_trades_per_day: 0,
            require_profit_target: false,
            min_profit_target_pct: ftmo.challenge_profit_target_pct as f64,
        }
    }
}

/// Prop-firm validation summary — explicit per-rule pass/fail flags plus
/// the worst observed values, so a downstream challenge gate can reject
/// the artifact without re-running the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropFirmRiskValidationSummary {
    pub rules: PropFirmRiskRules,
    pub trades_observed: usize,
    pub trading_days_observed: usize,
    pub max_daily_loss_pct_observed: f64,
    pub max_overall_drawdown_pct_observed: f64,
    pub largest_profit_share_observed: f64,
    pub max_trades_per_day_observed: usize,
    pub net_return_pct: f64,
    pub daily_loss_breach: bool,
    pub overall_drawdown_breach: bool,
    pub consistency_violation: bool,
    pub trade_limit_violation: bool,
    pub min_trading_days_ok: bool,
    pub profit_target_met: bool,
    pub all_rules_passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PropFirmRiskValidationScope {
    pub dataset_hash: String,
    pub evaluation_config_hash: String,
    pub strategy_hash: String,
    pub rules_hash: String,
    pub temporal_scope: TemporalScopeHashes,
}

impl PropFirmRiskValidationScope {
    pub fn new(
        dataset_hash: impl Into<String>,
        evaluation_config_hash: impl Into<String>,
        strategy_hash: impl Into<String>,
        rules: &PropFirmRiskRules,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<Self> {
        Ok(Self {
            dataset_hash: dataset_hash.into(),
            evaluation_config_hash: evaluation_config_hash.into(),
            strategy_hash: strategy_hash.into(),
            rules_hash: stable_json_hash(rules)?,
            temporal_scope: TemporalScopeHashes::from_contract(temporal_contract),
        })
    }

    pub fn validate_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        self.temporal_scope
            .validate_contract(temporal_contract)
            .map_err(|err| anyhow::anyhow!("prop firm risk validation {err}"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropFirmRiskValidationArtifactFile {
    pub artifact_kind: String,
    pub artifact_schema_version: u32,
    pub scope: PropFirmRiskValidationScope,
    pub summary: PropFirmRiskValidationSummary,
}

impl PropFirmRiskValidationArtifactFile {
    pub fn new(scope: PropFirmRiskValidationScope, summary: PropFirmRiskValidationSummary) -> Self {
        Self {
            artifact_kind: PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND.to_string(),
            artifact_schema_version: PROP_FIRM_RISK_VALIDATION_SCHEMA_VERSION,
            scope,
            summary,
        }
    }

    pub fn validate_for_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        if self.artifact_kind != PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND {
            bail!(
                "artifact kind {} cannot be used as a prop-firm risk validation artifact",
                self.artifact_kind
            );
        }
        if self.artifact_schema_version != PROP_FIRM_RISK_VALIDATION_SCHEMA_VERSION {
            bail!(
                "unsupported prop-firm risk validation schema version {}",
                self.artifact_schema_version
            );
        }
        self.scope.validate_temporal_contract(temporal_contract)
    }
}

pub fn write_prop_firm_risk_validation_artifact_atomic(
    path: impl AsRef<Path>,
    artifact: &PropFirmRiskValidationArtifactFile,
) -> Result<()> {
    write_json_atomic(path, artifact)
}

pub fn read_prop_firm_risk_validation_artifact(
    path: impl AsRef<Path>,
    temporal_contract: &TemporalFeatureContract,
) -> Result<PropFirmRiskValidationArtifactFile> {
    let artifact: PropFirmRiskValidationArtifactFile =
        read_json(path, "prop-firm risk validation")?;
    artifact.validate_for_temporal_contract(temporal_contract)?;
    Ok(artifact)
}

/// Inputs for [`compute_prop_firm_risk_summary`]. Callers pass observed
/// trades plus the rule set; the helper aggregates daily PnL, applies
/// the rules, and returns a summary with explicit pass/fail flags.
pub struct PropFirmRiskInput<'a> {
    pub trades: &'a [crate::quality::Trade],
    pub initial_balance: f64,
    pub rules: PropFirmRiskRules,
}

/// Aggregate observed trades against [`PropFirmRiskRules`] and produce a
/// validation summary. The function is deterministic and contains no
/// simulation — callers feed trades from a canonical backtest,
/// walk-forward, forward-test, or live-execution simulation.
pub fn compute_prop_firm_risk_summary(
    input: PropFirmRiskInput<'_>,
) -> PropFirmRiskValidationSummary {
    let initial_balance = if input.initial_balance.is_finite() && input.initial_balance > 0.0 {
        input.initial_balance
    } else {
        100_000.0
    };

    let mut day_pnl: BTreeMap<i64, f64> = BTreeMap::new();
    let mut day_trade_count: BTreeMap<i64, usize> = BTreeMap::new();
    let mut total_pnl = 0.0_f64;
    for trade in input.trades {
        total_pnl += trade.pnl;
        let day_key = trade.exit_time.unwrap_or(trade.entry_time) / 86_400_000;
        *day_pnl.entry(day_key).or_insert(0.0) += trade.pnl;
        *day_trade_count.entry(day_key).or_insert(0) += 1;
    }

    let trading_days_observed = day_trade_count.values().filter(|&&n| n > 0).count();
    let max_trades_per_day_observed = day_trade_count.values().copied().max().unwrap_or(0);

    let max_daily_loss_pct_observed = day_pnl
        .values()
        .copied()
        .filter(|pnl| *pnl < 0.0)
        .map(|pnl| pnl.abs() / initial_balance)
        .fold(0.0, f64::max);

    let mut equity = initial_balance;
    let mut peak = initial_balance;
    let mut max_overall_drawdown_pct_observed = 0.0_f64;
    for (_day, pnl) in &day_pnl {
        equity += pnl;
        peak = peak.max(equity);
        if peak > 0.0 {
            let dd = (peak - equity) / peak;
            if dd > max_overall_drawdown_pct_observed {
                max_overall_drawdown_pct_observed = dd;
            }
        }
    }

    let total_positive: f64 = day_pnl.values().copied().filter(|pnl| *pnl > 0.0).sum();
    let largest_positive: f64 = day_pnl.values().copied().fold(0.0, f64::max);
    let largest_profit_share_observed = if total_positive > f64::EPSILON {
        largest_positive / total_positive
    } else {
        0.0
    };

    let net_return_pct = if initial_balance > 0.0 {
        total_pnl / initial_balance
    } else {
        0.0
    };

    let rules = input.rules;
    let daily_loss_breach =
        rules.max_daily_loss_pct > 0.0 && max_daily_loss_pct_observed >= rules.max_daily_loss_pct;
    let overall_drawdown_breach = rules.max_overall_drawdown_pct > 0.0
        && max_overall_drawdown_pct_observed >= rules.max_overall_drawdown_pct;
    let consistency_violation = rules.max_profit_consistency_ratio > 0.0
        && largest_profit_share_observed > rules.max_profit_consistency_ratio;
    let trade_limit_violation =
        rules.max_trades_per_day > 0 && max_trades_per_day_observed > rules.max_trades_per_day;
    let min_trading_days_ok =
        rules.min_trading_days == 0 || trading_days_observed >= rules.min_trading_days;
    let profit_target_met = !rules.require_profit_target
        || (rules.min_profit_target_pct > 0.0 && net_return_pct >= rules.min_profit_target_pct);
    let all_rules_passed = !daily_loss_breach
        && !overall_drawdown_breach
        && !consistency_violation
        && !trade_limit_violation
        && min_trading_days_ok
        && profit_target_met;

    PropFirmRiskValidationSummary {
        rules,
        trades_observed: input.trades.len(),
        trading_days_observed,
        max_daily_loss_pct_observed,
        max_overall_drawdown_pct_observed,
        largest_profit_share_observed,
        max_trades_per_day_observed,
        net_return_pct,
        daily_loss_breach,
        overall_drawdown_breach,
        consistency_violation,
        trade_limit_violation,
        min_trading_days_ok,
        profit_target_met,
        all_rules_passed,
    }
}

pub struct WalkforwardBacktestInput<'a> {
    pub close: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub signals: &'a [i8],
    pub months: &'a [i64],
    pub days: &'a [i64],
    /// Real bar timestamps (ms or ns, same unit as `simulate_trades_core` expects).
    /// Used for gap detection, kill-zone rules, and day/week/month boundaries.
    pub timestamps: &'a [i64],
    pub train_ratio: f64,
    pub n_splits: usize,
    pub embargo_bars: usize,
    pub settings: &'a BacktestSettings,
    pub max_daily_loss_pct: f64,
    pub max_daily_profit_pct: f64,
    pub min_trading_days: usize,
    pub max_trades_per_day: usize,
    /// Starting account balance used to convert absolute PnL into daily return %.
    pub initial_balance: f64,
}

#[derive(Debug, Clone, Default)]
struct WalkforwardRiskDiagnostics {
    max_consec_losses: usize,
    daily_min_dd: f64,
    max_daily_loss: f64,
    daily_loss_breach: bool,
    consistency_violation: bool,
    trade_limit_violation: bool,
    min_trading_days_ok: bool,
    daily_returns: Vec<f64>,
    max_daily_dd_pct: f64,
    prop_compliant: bool,
}

/// Normalises a percentage value to a fraction in `[0, 1]`.
///
/// **F-022 documentation (2026-05-25)** — the boundary at `1.0` is
/// **inclusive** on the FRACTION side: `value == 1.0` is treated as
/// "100% as a fraction", NOT "1% as a percentage". This matters
/// because operator configs that pass `1.0` mean different things:
///
/// - `daily_drawdown_limit: 1.0` → 100% drawdown (sentinel: never trips)
/// - `daily_drawdown_limit: 5` → 5% (gets normalised to 0.05)
///
/// The 1.0 cutoff was chosen because real prop-firm caps are always
/// `< 1.0` (FTMO 5% / 10% are 0.05 / 0.10). A literal `1.0` is
/// always intended as the unit-fraction representation. Operators
/// who need exactly 1% must write `0.01` or use the typed
/// `RiskConfig::risk_per_trade` field which has explicit semantics.
///
/// - Non-finite / non-positive → `0.0` (gate disabled).
/// - `(0.0, 1.0]` → unchanged (already a fraction).
/// - `(1.0, ∞)` → divide by 100 (caller meant percent).
fn normalized_pct_threshold(value: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 {
        0.0
    } else if value > 1.0 {
        value / 100.0
    } else {
        value
    }
}

#[allow(clippy::too_many_arguments)]
fn walkforward_risk_diagnostics(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    signals: &[i8],
    days: &[i64],
    timestamps: &[i64],
    settings: &BacktestSettings,
    evaluator_max_daily_dd: f64,
    max_daily_loss_pct: f64,
    max_daily_profit_pct: f64,
    min_trading_days: usize,
    max_trades_per_day: usize,
    initial_balance: f64,
) -> WalkforwardRiskDiagnostics {
    if close.is_empty() || days.is_empty() {
        return WalkforwardRiskDiagnostics::default();
    }
    let initial_balance = if initial_balance.is_finite() && initial_balance > 0.0 {
        initial_balance
    } else {
        100_000.0
    };

    let mut day_offsets = BTreeMap::<i64, usize>::new();
    let mut daily_pnl = Vec::<f64>::new();
    let mut daily_trade_counts = Vec::<usize>::new();
    for &day in days {
        day_offsets.entry(day).or_insert_with(|| {
            let offset = daily_pnl.len();
            daily_pnl.push(0.0);
            daily_trade_counts.push(0);
            offset
        });
    }

    // Use real timestamps so simulate_trades_core applies correct gap/session/kill-zone logic.
    let ts = if timestamps.len() == close.len() {
        timestamps
    } else {
        days
    };
    let trades = simulate_trades_core(close, high, low, ts, signals, settings);
    let mut max_consec_losses = 0usize;
    let mut current_consec_losses = 0usize;

    for trade in &trades {
        if trade.pnl < 0.0 {
            current_consec_losses += 1;
            max_consec_losses = max_consec_losses.max(current_consec_losses);
        } else if trade.pnl > 0.0 {
            current_consec_losses = 0;
        }

        let exit_day = trade.exit_time.unwrap_or(trade.entry_time);
        let offset = if let Some(&offset) = day_offsets.get(&exit_day) {
            offset
        } else {
            let offset = daily_pnl.len();
            day_offsets.insert(exit_day, offset);
            daily_pnl.push(0.0);
            daily_trade_counts.push(0);
            offset
        };
        daily_pnl[offset] += trade.pnl;
        daily_trade_counts[offset] += 1;
    }

    let daily_returns: Vec<f64> = daily_pnl.iter().map(|pnl| pnl / initial_balance).collect();
    let daily_min_return = daily_returns.iter().copied().fold(0.0, f64::min);
    let closed_trade_daily_loss = daily_returns
        .iter()
        .filter(|ret| **ret < 0.0)
        .map(|ret| ret.abs())
        .fold(0.0, f64::max);
    let evaluator_max_daily_dd = if evaluator_max_daily_dd.is_finite() {
        evaluator_max_daily_dd.max(0.0)
    } else {
        0.0
    };
    let max_daily_loss = closed_trade_daily_loss.max(evaluator_max_daily_dd);
    let daily_min_dd = daily_min_return.min(-evaluator_max_daily_dd);

    let max_daily_loss_limit = normalized_pct_threshold(max_daily_loss_pct);
    let daily_loss_breach = max_daily_loss_limit > 0.0 && max_daily_loss >= max_daily_loss_limit;

    let profit_consistency_limit = normalized_pct_threshold(max_daily_profit_pct);
    let total_positive_daily_pnl: f64 = daily_pnl.iter().filter(|pnl| **pnl > 0.0).sum();
    let largest_positive_daily_pnl = daily_pnl.iter().copied().fold(0.0, f64::max);
    let largest_profit_share = if total_positive_daily_pnl > f64::EPSILON {
        largest_positive_daily_pnl / total_positive_daily_pnl
    } else {
        0.0
    };
    let consistency_violation =
        profit_consistency_limit > 0.0 && largest_profit_share > profit_consistency_limit;

    let trade_limit_violation = max_trades_per_day > 0
        && daily_trade_counts
            .iter()
            .any(|&count| count > max_trades_per_day);
    let trading_days = daily_trade_counts
        .iter()
        .filter(|&&count| count > 0)
        .count();
    let min_trading_days_ok = min_trading_days == 0 || trading_days >= min_trading_days;
    let prop_compliant = !daily_loss_breach
        && !consistency_violation
        && !trade_limit_violation
        && min_trading_days_ok;

    WalkforwardRiskDiagnostics {
        max_consec_losses,
        daily_min_dd,
        max_daily_loss,
        daily_loss_breach,
        consistency_violation,
        trade_limit_violation,
        min_trading_days_ok,
        daily_returns,
        max_daily_dd_pct: max_daily_loss,
        prop_compliant,
    }
}

pub fn embargoed_walkforward_backtest(
    input: WalkforwardBacktestInput<'_>,
) -> Result<WalkforwardSummary> {
    let WalkforwardBacktestInput {
        close,
        high,
        low,
        signals,
        months,
        days,
        timestamps,
        train_ratio,
        n_splits,
        embargo_bars,
        settings,
        max_daily_loss_pct,
        max_daily_profit_pct,
        min_trading_days,
        max_trades_per_day,
        initial_balance,
    } = input;
    let n = close.len();
    if n == 0
        || high.len() != n
        || low.len() != n
        || signals.len() != n
        || months.len() != n
        || days.len() != n
    {
        bail!("empty data or length mismatch");
    }
    if n_splits == 0 {
        bail!("n_splits must be greater than zero");
    }
    if !train_ratio.is_finite() || !(0.0..1.0).contains(&train_ratio) {
        bail!("train_ratio must be finite and in the open interval (0, 1)");
    }

    let window = (n / n_splits).max(1);

    // `window` is constant across splits: with floor division
    // n_splits*window <= n, so `end` == (i+1)*window for every split (the
    // `.min(n)` clamp never bites) and `end - start` == window for ALL splits.
    // The 80-bar floor and the train/embargo validity checks below are
    // therefore split-INDEPENDENT — either every split qualifies or none does.
    // Each split's backtest reads disjoint slices with NO RNG, so the
    // qualifying splits evaluate in parallel bit-identically to the old serial
    // loop. This saturates idle cores when the outer candidate axis has shrunk
    // below the core count (the validation-tail idle-core leak).
    //
    // F-020 + F-021: the 80-bar floor is timeframe-AGNOSTIC by design (80
    // M1-bars = 80 min, 80 D1-bars = 80 days). Phase B (deferred): replace
    // `< 80` with a calendar-day minimum from the timestamps array via an
    // operator-tunable `min_window_days` knob.
    if window < 80 {
        tracing::warn!(
            target: "neoethos_search::validation",
            bars_in_window = window,
            n_splits,
            "walkforward window below 80-bar floor; dropping all splits. \
             Consider reducing n_splits or expanding the input window."
        );
    }
    let mut split_results: Vec<WalkforwardSplitResult> = if window < 80 {
        Vec::new()
    } else {
        (0..n_splits)
            .into_par_iter()
            .filter_map(|i| {
                let start = i * window;
                let end = ((i + 1) * window).min(n);

                let train_end = start + ((window as f64) * train_ratio) as usize;
                let test_start = train_end + embargo_bars;

                if test_start >= end || (train_end - start) < 40 || (end - test_start) < 40 {
                    return None;
                }

                let slice_close = &close[test_start..end];
                let slice_high = &high[test_start..end];
                let slice_low = &low[test_start..end];
                let slice_sig = &signals[test_start..end];
                let slice_months = &months[test_start..end];
                let slice_days = &days[test_start..end];
                let slice_ts = if timestamps.len() == n {
                    &timestamps[test_start..end]
                } else {
                    slice_days
                };

                let metrics = fast_evaluate_strategy_core(
                    slice_close,
                    slice_high,
                    slice_low,
                    slice_sig,
                    // Phase 1: legacy fixed-1-lot for the walk-forward slice eval.
                    &[],
                    slice_months,
                    slice_days,
                    &[],
                    settings,
                );

                // Map metrics [net_profit, 0.0, peak_equity, max_dd, win_rate, pf, expectancy, 0.0, trade_count, consistency, max_daily_dd]
                let net_profit = metrics[0];
                let max_dd = metrics[3];
                let win_rate = metrics[4];
                let trade_count = metrics[8] as usize;
                let max_daily_dd = metrics[10];
                let risk = walkforward_risk_diagnostics(
                    slice_close,
                    slice_high,
                    slice_low,
                    slice_sig,
                    slice_days,
                    slice_ts,
                    settings,
                    max_daily_dd,
                    max_daily_loss_pct,
                    max_daily_profit_pct,
                    min_trading_days,
                    max_trades_per_day,
                    initial_balance,
                );

                Some(WalkforwardSplitResult {
                    split: i + 1,
                    trades: trade_count,
                    pnl: net_profit,
                    win_rate,
                    max_dd,
                    max_consec_losses: risk.max_consec_losses,
                    daily_min_dd: risk.daily_min_dd,
                    max_daily_loss: risk.max_daily_loss,
                    daily_loss_breach: risk.daily_loss_breach,
                    consistency_violation: risk.consistency_violation,
                    trade_limit_violation: risk.trade_limit_violation,
                    min_trading_days_ok: risk.min_trading_days_ok,
                    daily_returns: risk.daily_returns,
                    max_daily_dd_pct: risk.max_daily_dd_pct,
                    prop_compliant: risk.prop_compliant,
                })
            })
            .collect()
    };
    // par collect preserves range order, but make the ascending-split
    // invariant explicit for downstream consumers.
    split_results.sort_by_key(|r| r.split);

    Ok(summarize_walkforward_splits(split_results))
}

/// Reduce a per-gene list of qualifying [`WalkforwardSplitResult`]s into a
/// [`WalkforwardSummary`]. SINGLE source of truth for the avg/any/all
/// reductions, shared by the single-gene [`embargoed_walkforward_backtest`] and
/// the GPU-routed [`embargoed_walkforward_population`] so both produce a
/// **byte-identical** summary (same averaging divisor, same any/all booleans,
/// same empty-splits sentinel). Callers MUST pass the splits already sorted by
/// `split` ascending (both call sites do).
fn summarize_walkforward_splits(split_results: Vec<WalkforwardSplitResult>) -> WalkforwardSummary {
    if split_results.is_empty() {
        return WalkforwardSummary {
            walk_forward_splits: 0,
            avg_pnl: 0.0,
            avg_win_rate: 0.0,
            avg_max_dd: 0.0,
            avg_max_consec_losses: 0.0,
            avg_daily_min_dd: 0.0,
            avg_max_daily_loss: 0.0,
            any_daily_loss_breach: false,
            any_consistency_violation: false,
            any_trade_limit_violation: false,
            all_min_trading_days_ok: false,
            splits: Vec::new(),
        };
    }

    let n_res = split_results.len() as f64;
    let avg_pnl = split_results.iter().map(|r| r.pnl).sum::<f64>() / n_res;
    let avg_win = split_results.iter().map(|r| r.win_rate).sum::<f64>() / n_res;
    let avg_dd = split_results.iter().map(|r| r.max_dd).sum::<f64>() / n_res;
    let avg_max_consec_losses = split_results
        .iter()
        .map(|r| r.max_consec_losses as f64)
        .sum::<f64>()
        / n_res;
    let avg_daily_min_dd = split_results.iter().map(|r| r.daily_min_dd).sum::<f64>() / n_res;
    let avg_max_daily_loss = split_results.iter().map(|r| r.max_daily_loss).sum::<f64>() / n_res;

    WalkforwardSummary {
        walk_forward_splits: split_results.len(),
        avg_pnl,
        avg_win_rate: avg_win,
        avg_max_dd: avg_dd,
        avg_max_consec_losses,
        avg_daily_min_dd,
        avg_max_daily_loss,
        any_daily_loss_breach: split_results.iter().any(|r| r.daily_loss_breach),
        any_consistency_violation: split_results.iter().any(|r| r.consistency_violation),
        any_trade_limit_violation: split_results.iter().any(|r| r.trade_limit_violation),
        all_min_trading_days_ok: split_results.iter().all(|r| r.min_trading_days_ok),
        splits: split_results,
    }
}

/// Shared (gene-INDEPENDENT) inputs for the GPU-routed population walk-forward.
///
/// Everything here is identical across the whole survivor portfolio: the
/// full-series OHLCV / calendar arrays, the split geometry, and the prop-firm
/// risk knobs. The per-gene axis (precomputed signals + the GPU metrics) is
/// supplied separately to [`embargoed_walkforward_population`].
pub struct WalkforwardPopulationInput<'a> {
    pub close: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub months: &'a [i64],
    pub days: &'a [i64],
    /// Real bar timestamps (same unit as `simulate_trades_core` expects).
    pub timestamps: &'a [i64],
    pub train_ratio: f64,
    pub n_splits: usize,
    pub embargo_bars: usize,
    /// PER-GENE backtest settings (one per gene, aligned to `signals_per_gene`),
    /// used by the CPU risk-diagnostic half (`walkforward_risk_diagnostics` →
    /// `simulate_trades_core`). These MUST be the SAME per-gene settings the
    /// single-gene path built (`discovery_backtest_settings`), in particular the
    /// gene's own SL/TP, so the risk-diagnostic half is byte-identical to
    /// `embargoed_walkforward_backtest`. (The GPU **metrics** half gets the
    /// gene's SL/TP separately via the metrics provider's own per-gene arrays.)
    pub gene_settings: &'a [BacktestSettings],
    pub max_daily_loss_pct: f64,
    pub max_daily_profit_pct: f64,
    pub min_trading_days: usize,
    pub max_trades_per_day: usize,
    pub initial_balance: f64,
}

/// AREA 2 / Stage C (2026-06-09) — GPU-routed **population** walk-forward.
///
/// This is the population twin of [`embargoed_walkforward_backtest`]. The split
/// loop in the single-gene path runs `fast_evaluate_strategy_core` PER GENE PER
/// SPLIT on the contiguous test slice `[test_start..end]` (validation.rs ~:1124).
/// For a portfolio of `n_genes` survivors that is `n_genes × n_splits` tiny CPU
/// backtests. This helper **transposes** the loop: it walks the qualifying split
/// windows ONCE and, per window, calls `metrics_fn(test_start, end)` — wired by
/// the caller to ONE GPU population launch over ALL survivor genes on that
/// contiguous slice — collapsing the launch count to `n_splits`.
///
/// ## HYBRID: GPU for backtest metrics, CPU for risk diagnostics
/// The GPU population kernel emits ONLY the 11-wide metric array
/// (net_profit/max_dd/win_rate/trade_count, …). It does NOT produce the
/// walk-forward risk diagnostics (max_consec_losses, daily_returns,
/// prop_compliant, …) which need the per-trade list. So per split this helper:
///  - takes the **metrics half** (slots 0/3/4/8/10) from `metrics_fn`'s GPU rows
///    (one per gene), and
///  - computes [`walkforward_risk_diagnostics`] **on the CPU** per gene on the
///    sliced precomputed `signals` — EXACTLY as the single-gene path does.
///
/// The resulting per-gene `WalkforwardSplitResult` is field-for-field identical
/// to the single-gene path's (the metric slots are read the SAME way; the risk
/// fields come from the SAME CPU function on the SAME sliced signals), and each
/// gene's `WalkforwardSummary` is built through the SHARED
/// [`summarize_walkforward_splits`] reducer — so the avg/any/all aggregation is
/// byte-identical to `embargoed_walkforward_backtest`.
///
/// ## Fixed-1-lot
/// The metrics half MUST be produced with `risk_based_sizing == false` (fixed
/// 1-lot) and empty (`&[]`) confidence, matching the single-gene WF call at
/// validation.rs:1129-1130. The caller wires `metrics_fn` to
/// `validation_genes_population`, which FORCES `risk_based_sizing = false`.
///
/// ## Split-qualification parity
/// The window size, the 80-bar floor, and the train/embargo validity checks are
/// COPIED VERBATIM from the single-gene path (which proved them
/// split-INDEPENDENT): either every split qualifies or none does, and each
/// qualifying split is the SAME contiguous `[test_start..end]` slice. The set of
/// qualifying splits is therefore identical to the single-gene loop's.
///
/// Returns one [`WalkforwardSummary`] per gene, in `genes` order.
#[allow(clippy::too_many_arguments)]
pub fn embargoed_walkforward_population<F>(
    input: WalkforwardPopulationInput<'_>,
    // Full-series precomputed signals, one per gene (aligned to the shared
    // per-bar arrays). Sliced per window for the CPU risk diagnostics — the
    // single source of truth for the per-gene signal direction, identical to
    // what the single-gene path slices.
    signals_per_gene: &[Vec<i8>],
    // Per-window GPU metrics provider: `metrics_fn(test_start, end)` returns one
    // `[f64; 11]` row per gene (same order as `signals_per_gene`) for the
    // contiguous slice `[test_start..end]`. The caller wires this to a single
    // GPU population launch (fixed-1-lot). Errors propagate (fail-loud).
    mut metrics_fn: F,
) -> Result<Vec<WalkforwardSummary>>
where
    F: FnMut(usize, usize) -> Result<Vec<[f64; 11]>>,
{
    let WalkforwardPopulationInput {
        close,
        high,
        low,
        months,
        days,
        timestamps,
        train_ratio,
        n_splits,
        embargo_bars,
        gene_settings,
        max_daily_loss_pct,
        max_daily_profit_pct,
        min_trading_days,
        max_trades_per_day,
        initial_balance,
    } = input;

    let n = close.len();
    let n_genes = signals_per_gene.len();
    if n == 0 || high.len() != n || low.len() != n || months.len() != n || days.len() != n {
        bail!("empty data or length mismatch");
    }
    if gene_settings.len() != n_genes {
        bail!(
            "walk-forward population gene_settings.len()={} != {} genes",
            gene_settings.len(),
            n_genes
        );
    }
    if let Some((g, s)) = signals_per_gene
        .iter()
        .enumerate()
        .find(|(_, s)| s.len() != n)
    {
        bail!(
            "walk-forward population signals[{}].len()={} != {} bars",
            g,
            s.len(),
            n
        );
    }
    if n_splits == 0 {
        bail!("n_splits must be greater than zero");
    }
    if !train_ratio.is_finite() || !(0.0..1.0).contains(&train_ratio) {
        bail!("train_ratio must be finite and in the open interval (0, 1)");
    }

    // Empty portfolio: nothing to evaluate.
    if n_genes == 0 {
        return Ok(Vec::new());
    }

    // ── Window geometry — COPIED VERBATIM from `embargoed_walkforward_backtest`
    //    so the set of qualifying splits is byte-identical. ───────────────────
    let window = (n / n_splits).max(1);
    if window < 80 {
        tracing::warn!(
            target: "neoethos_search::validation",
            bars_in_window = window,
            n_splits,
            "walkforward window below 80-bar floor; dropping all splits. \
             Consider reducing n_splits or expanding the input window."
        );
        // No qualifying splits → every gene gets the empty-splits summary, exactly
        // like the single-gene path returns when `window < 80`.
        return Ok((0..n_genes)
            .map(|_| summarize_walkforward_splits(Vec::new()))
            .collect());
    }

    // Per-gene accumulator of split results, filled split-by-split.
    let mut per_gene_splits: Vec<Vec<WalkforwardSplitResult>> =
        (0..n_genes).map(|_| Vec::new()).collect();

    for i in 0..n_splits {
        let start = i * window;
        let end = ((i + 1) * window).min(n);

        let train_end = start + ((window as f64) * train_ratio) as usize;
        let test_start = train_end + embargo_bars;

        // SAME qualification predicate as the single-gene path.
        if test_start >= end || (train_end - start) < 40 || (end - test_start) < 40 {
            continue;
        }

        // ── GPU half: ONE population launch over all genes on this contiguous
        //    slice. The caller forces fixed-1-lot / risk_based_sizing=false. ──
        let gpu_metrics = metrics_fn(test_start, end)?;
        if gpu_metrics.len() != n_genes {
            bail!(
                "walk-forward split {} metrics provider returned {} rows for {} genes",
                i + 1,
                gpu_metrics.len(),
                n_genes
            );
        }

        // Contiguous per-bar slices, shared across genes.
        let slice_close = &close[test_start..end];
        let slice_high = &high[test_start..end];
        let slice_low = &low[test_start..end];
        let slice_days = &days[test_start..end];
        let slice_ts = if timestamps.len() == n {
            &timestamps[test_start..end]
        } else {
            slice_days
        };

        // ── CPU half (per gene): risk diagnostics on the sliced precomputed
        //    signals — IDENTICAL to the single-gene path. ────────────────────
        let split_results: Vec<WalkforwardSplitResult> = (0..n_genes)
            .into_par_iter()
            .map(|g| {
                let m = gpu_metrics[g];
                // Metric slots read EXACTLY as the single-gene path
                // (validation.rs:1138-1142): trade_count via `as usize`, NOT the
                // `BacktestMetrics::from_metric_array` rounding, so the population
                // path stays byte-identical to `embargoed_walkforward_backtest`.
                let net_profit = m[0];
                let max_dd = m[3];
                let win_rate = m[4];
                let trade_count = m[8] as usize;
                let max_daily_dd = m[10];

                let slice_sig = &signals_per_gene[g][test_start..end];
                // Per-gene settings (the gene's own SL/TP) so `simulate_trades_core`
                // inside the diagnostics applies the SAME SL/TP exits the single-gene
                // path did — byte-identical risk-diagnostic half.
                let risk = walkforward_risk_diagnostics(
                    slice_close,
                    slice_high,
                    slice_low,
                    slice_sig,
                    slice_days,
                    slice_ts,
                    &gene_settings[g],
                    max_daily_dd,
                    max_daily_loss_pct,
                    max_daily_profit_pct,
                    min_trading_days,
                    max_trades_per_day,
                    initial_balance,
                );

                WalkforwardSplitResult {
                    split: i + 1,
                    trades: trade_count,
                    pnl: net_profit,
                    win_rate,
                    max_dd,
                    max_consec_losses: risk.max_consec_losses,
                    daily_min_dd: risk.daily_min_dd,
                    max_daily_loss: risk.max_daily_loss,
                    daily_loss_breach: risk.daily_loss_breach,
                    consistency_violation: risk.consistency_violation,
                    trade_limit_violation: risk.trade_limit_violation,
                    min_trading_days_ok: risk.min_trading_days_ok,
                    daily_returns: risk.daily_returns,
                    max_daily_dd_pct: risk.max_daily_dd_pct,
                    prop_compliant: risk.prop_compliant,
                }
            })
            .collect();

        for (g, r) in split_results.into_iter().enumerate() {
            per_gene_splits[g].push(r);
        }
    }

    // Each gene's splits are pushed in ascending split order (the `for i` loop is
    // sequential), matching the single-gene path's post-sort invariant. Reduce
    // through the SHARED summarizer so the aggregation is byte-identical.
    Ok(per_gene_splits
        .into_iter()
        .map(summarize_walkforward_splits)
        .collect())
}

pub struct CombinatorialPurgedCV {
    pub n_splits: usize,
    pub n_test_groups: usize,
    pub embargo_pct: f64,
    pub purge_pct: f64,
}

impl CombinatorialPurgedCV {
    pub fn new(n_splits: usize, n_test_groups: usize, embargo_pct: f64, purge_pct: f64) -> Self {
        Self {
            n_splits,
            n_test_groups,
            embargo_pct,
            purge_pct,
        }
    }

    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        if n_samples == 0 || self.n_splits < 2 {
            return Vec::new();
        }

        // Divide n_samples into S groups
        let group_size = n_samples / self.n_splits;
        if group_size == 0 {
            return Vec::new();
        }

        let mut groups = Vec::with_capacity(self.n_splits);
        for i in 0..self.n_splits {
            let start = i * group_size;
            let end = if i == self.n_splits - 1 {
                n_samples
            } else {
                (i + 1) * group_size
            };
            groups.push(start..end);
        }

        let purge_size = (n_samples as f64 * self.purge_pct).ceil() as usize;
        let embargo_size = (n_samples as f64 * self.embargo_pct).ceil() as usize;

        let mut results = Vec::new();

        // Form all combinations of k test groups
        for combination in (0..self.n_splits).combinations(self.n_test_groups) {
            let mut test_idx = Vec::new();
            let mut candidate_train_groups = Vec::new();

            for (i, group) in groups.iter().enumerate().take(self.n_splits) {
                if combination.contains(&i) {
                    test_idx.extend(group.clone());
                } else {
                    candidate_train_groups.push(i);
                }
            }

            let mut train_idx = Vec::new();

            // For each training group, apply purging and embargoing relative to ALL test groups
            for &g_idx in &candidate_train_groups {
                let group_range = groups[g_idx].clone();
                let group_start = group_range.start;
                let group_end = group_range.end;

                let mut group_valid_start = group_start;
                let mut group_valid_end = group_end;

                for &t_idx in &combination {
                    let test_range = groups[t_idx].clone();

                    // 1. Purge: if training group is BEFORE a test group,
                    // remove samples at the end of training group that look into the test group.
                    if group_end <= test_range.start {
                        let potential_end = test_range.start.saturating_sub(purge_size);
                        if potential_end < group_valid_end && potential_end >= group_start {
                            group_valid_end = potential_end;
                        }
                    }

                    // 2. Embargo: if training group is AFTER a test group,
                    // remove samples at the beginning of training group that are serially correlated.
                    if group_start >= test_range.end {
                        let potential_start = test_range.end + embargo_size;
                        if potential_start > group_valid_start && potential_start <= group_end {
                            group_valid_start = potential_start;
                        }
                    }
                }

                if group_valid_start < group_valid_end {
                    train_idx.extend(group_valid_start..group_valid_end);
                }
            }

            if !test_idx.is_empty() && !train_idx.is_empty() {
                results.push((train_idx, test_idx));
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporal_contract(label_policy_hash: &str) -> TemporalFeatureContract {
        TemporalFeatureContract::strict_live(
            "UTC",
            "alignment-policy-v1",
            label_policy_hash,
            "walk-forward-policy-v1",
            "live-readiness-policy-v1",
        )
        .expect("strict temporal contract should be valid")
    }

    fn sample_summary() -> WalkforwardSummary {
        WalkforwardSummary {
            walk_forward_splits: 1,
            avg_pnl: 12.0,
            avg_win_rate: 0.5,
            avg_max_dd: 0.1,
            avg_max_consec_losses: 1.0,
            avg_daily_min_dd: -0.01,
            avg_max_daily_loss: 0.01,
            any_daily_loss_breach: false,
            any_consistency_violation: false,
            any_trade_limit_violation: false,
            all_min_trading_days_ok: true,
            splits: Vec::new(),
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("forex-validation-{name}-{unique}.json"))
    }

    fn flat_settings() -> BacktestSettings {
        BacktestSettings {
            sl_pips: 1_000_000.0,
            tp_pips: 1_000_000.0,
            max_hold_bars: 1,
            min_hold_bars: 1,
            max_trades_per_day: 0,
            gap_threshold_ms: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            pip_value: 1.0,
            spread_pips: 0.0,
            commission_per_trade: 0.0,
            pip_value_per_lot: 10_000.0,
            kill_zones_enabled: false,
            session_spread_profile: None,
            // Phase C — flat-test fixture: no swap, no conversion fee.
            // These are deliberately zeroed so the existing test
            // assertions (which assume only commission + spread costs)
            // continue to hold.
            swap_long_pips_per_day: 0.0,
            swap_short_pips_per_day: 0.0,
            pnl_conversion_fee_rate: 0.0,
            // Risk-based sizing OFF for the flat-test fixture so the existing
            // PnL assertions (pips × pip_value_per_lot) keep holding.
            risk_based_sizing: false,
            risk_per_trade_min: 0.005,
            risk_per_trade_max: 0.03,
            high_quality_confidence: 0.65,
            adaptive_base_pips: None,
            adaptive_vol_mult: 0.0,
            adaptive_rr: 2.0,
        }
    }

    #[test]
    fn risk_diagnostics_enforce_prop_constraints_from_simulated_trades() {
        let close = [100.0, 101.0, 103.0, 102.0, 100.0, 99.0, 98.0];
        let high = close;
        let low = close;
        let signals = [1, 0, 1, 0, 1, 0, 0];
        let days = [1, 1, 1, 2, 2, 2, 2];

        let risk = walkforward_risk_diagnostics(
            &close,
            &high,
            &low,
            &signals,
            &days,
            &days,
            &flat_settings(),
            0.0,
            0.01,
            0.50,
            3,
            1,
            100_000.0,
        );

        assert_eq!(risk.max_consec_losses, 2);
        assert!(risk.daily_loss_breach);
        assert!(risk.consistency_violation);
        assert!(risk.trade_limit_violation);
        assert!(!risk.min_trading_days_ok);
        assert!(!risk.prop_compliant);
        assert_eq!(risk.daily_returns.len(), 2);
    }

    #[test]
    fn walkforward_validation_artifact_binds_temporal_scope() {
        let contract = temporal_contract("label-policy-v1");
        let scope = WalkforwardValidationScope::new("dataset-a", "eval-a", &contract);
        let artifact = WalkforwardValidationArtifactFile::new(scope.clone(), sample_summary());

        assert_eq!(artifact.artifact_kind, WALKFORWARD_VALIDATION_ARTIFACT_KIND);
        assert_eq!(artifact.scope, scope);
        artifact
            .validate_for_temporal_contract(&contract)
            .expect("matching temporal contract should validate");
    }

    #[test]
    fn walkforward_validation_artifact_rejects_temporal_drift_and_wrong_kind() {
        let contract = temporal_contract("label-policy-v1");
        let changed_contract = temporal_contract("label-policy-v2");
        let scope = WalkforwardValidationScope::new("dataset-a", "eval-a", &contract);
        let mut artifact = WalkforwardValidationArtifactFile::new(scope, sample_summary());

        let err = artifact
            .validate_for_temporal_contract(&changed_contract)
            .expect_err("changed temporal contract must not validate");
        assert!(err.to_string().contains("temporal_contract_hash"));

        artifact.artifact_kind = "search_checkpoint_artifact".to_string();
        let err = artifact
            .validate_for_temporal_contract(&contract)
            .expect_err("wrong artifact kind must not validate");
        assert!(err.to_string().contains("cannot be used as a walk-forward"));
    }

    #[test]
    fn walkforward_validation_artifact_uses_shared_atomic_io() {
        let contract = temporal_contract("label-policy-v1");
        let scope = WalkforwardValidationScope::new("dataset-a", "eval-a", &contract);
        let artifact = WalkforwardValidationArtifactFile::new(scope, sample_summary());
        let path = temp_path("artifact");

        write_walkforward_validation_artifact_atomic(&path, &artifact)
            .expect("atomic validation artifact write should succeed");
        let loaded = read_walkforward_validation_artifact(&path, &contract)
            .expect("matching validation artifact should load");
        assert_eq!(loaded.artifact_kind, WALKFORWARD_VALIDATION_ARTIFACT_KIND);
        assert_eq!(loaded.summary.walk_forward_splits, 1);

        let changed_contract = temporal_contract("label-policy-v2");
        let err = read_walkforward_validation_artifact(&path, &changed_contract)
            .expect_err("temporal drift must reject persisted validation artifact");
        assert!(err.to_string().contains("temporal_contract_hash"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn backtest_metrics_preserve_canonical_metric_layout() {
        let raw = [
            12.0, 1.5, 100_012.0, 0.02, 0.60, 1.8, 4.0, 0.0, 7.0, 0.9, 0.01,
        ];
        let metrics = BacktestMetrics::from_metric_array(raw);

        assert_eq!(metrics.net_profit, 12.0);
        assert_eq!(metrics.sharpe, 1.5);
        assert_eq!(metrics.trade_count, 7);
        assert_eq!(metrics.to_metric_array(), raw);
    }

    #[test]
    fn canonical_backtest_artifact_uses_shared_atomic_io_and_temporal_scope() {
        let contract = temporal_contract("label-policy-v1");
        let scope = CanonicalBacktestScope::new("dataset-a", "eval-a", "strategy-a", &contract);
        let artifact = CanonicalBacktestArtifactFile::new(
            scope,
            BacktestMetrics::from_metric_array([
                12.0, 1.5, 100_012.0, 0.02, 0.60, 1.8, 4.0, 0.0, 7.0, 0.9, 0.01,
            ]),
        );
        let path = temp_path("canonical-backtest");

        write_canonical_backtest_artifact_atomic(&path, &artifact)
            .expect("atomic canonical backtest artifact write should succeed");
        let loaded = read_canonical_backtest_artifact(&path, &contract)
            .expect("matching canonical backtest artifact should load");
        assert_eq!(loaded.artifact_kind, CANONICAL_BACKTEST_ARTIFACT_KIND);
        assert_eq!(loaded.metrics.trade_count, 7);

        let changed_contract = temporal_contract("label-policy-v2");
        let err = read_canonical_backtest_artifact(&path, &changed_contract)
            .expect_err("temporal drift must reject persisted backtest artifact");
        assert!(err.to_string().contains("temporal_contract_hash"));

        let _ = std::fs::remove_file(path);
    }

    fn sample_forward_test_summary() -> ForwardTestSummary {
        ForwardTestSummary {
            bars: 5,
            metrics: BacktestMetrics::from_metric_array([
                25.0, 1.6, 100_025.0, 0.015, 0.62, 1.9, 5.0, 0.0, 5.0, 0.85, 0.008,
            ]),
            span_days: 0.0,
        }
    }

    #[test]
    fn forward_test_artifact_binds_temporal_scope_and_rejects_drift() {
        let contract = temporal_contract("label-policy-v1");
        let scope =
            ForwardTestValidationScope::new("dataset-tail", "eval-config", "strategy", &contract);
        let artifact = ForwardTestValidationArtifactFile::new(scope, sample_forward_test_summary());

        artifact
            .validate_for_temporal_contract(&contract)
            .expect("matching contract should accept the forward-test artifact");

        let drifted = temporal_contract("label-policy-v2");
        let err = artifact
            .validate_for_temporal_contract(&drifted)
            .expect_err("temporal drift must reject the forward-test artifact");
        assert!(err.to_string().contains("forward test"));
    }

    #[test]
    fn forward_test_artifact_rejects_wrong_kind_and_unsupported_schema() {
        let contract = temporal_contract("label-policy-v1");
        let scope =
            ForwardTestValidationScope::new("dataset-tail", "eval-config", "strategy", &contract);
        let mut artifact =
            ForwardTestValidationArtifactFile::new(scope, sample_forward_test_summary());
        artifact.artifact_kind = "canonical_strategy_backtest_artifact".to_string();
        let err = artifact
            .validate_for_temporal_contract(&contract)
            .expect_err("wrong artifact_kind must reject the forward-test load");
        assert!(err.to_string().contains("forward-test validation artifact"));

        artifact.artifact_kind = FORWARD_TEST_VALIDATION_ARTIFACT_KIND.to_string();
        artifact.artifact_schema_version = FORWARD_TEST_VALIDATION_SCHEMA_VERSION + 1;
        let err = artifact
            .validate_for_temporal_contract(&contract)
            .expect_err("unsupported schema must reject the forward-test load");
        assert!(err.to_string().contains("forward-test validation schema"));
    }

    #[test]
    fn forward_test_artifact_round_trips_through_atomic_io() {
        let contract = temporal_contract("label-policy-v1");
        let scope =
            ForwardTestValidationScope::new("dataset-tail", "eval-config", "strategy", &contract);
        let artifact = ForwardTestValidationArtifactFile::new(scope, sample_forward_test_summary());
        let path = temp_path("forward-test");

        write_forward_test_validation_artifact_atomic(&path, &artifact)
            .expect("atomic forward-test artifact write should succeed");
        let loaded = read_forward_test_validation_artifact(&path, &contract)
            .expect("matching forward-test artifact should load");
        assert_eq!(loaded.artifact_kind, FORWARD_TEST_VALIDATION_ARTIFACT_KIND);
        assert_eq!(loaded.summary.bars, 5);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compute_forward_test_summary_builds_metrics_and_span() {
        let close = [1.0, 1.01, 1.02, 1.015, 1.025];
        let high = close;
        let low = close;
        let signals = [1_i8, 0, 1, 0, 0];
        let months = [1_i64; 5];
        let days = [1_i64, 1, 1, 2, 2];
        let timestamps = [
            1_700_000_000_000_i64,
            1_700_000_060_000,
            1_700_000_120_000,
            1_700_086_400_000,
            1_700_086_460_000,
        ];
        let summary = compute_forward_test_summary(ForwardTestInput {
            close: &close,
            high: &high,
            low: &low,
            signals: &signals,
            months: &months,
            days: &days,
            timestamps: &timestamps,
            settings: &flat_settings(),
        })
        .expect("forward-test summary should build");
        assert_eq!(summary.bars, 5);
        // The window spans roughly one calendar day (last - first ≈ 86 460s).
        assert!(summary.span_days >= 1.0 && summary.span_days < 2.0);
    }

    #[test]
    fn compute_forward_test_summary_rejects_mismatched_inputs() {
        let close = [1.0, 1.0, 1.0];
        let bad_high = [1.0, 1.0]; // length mismatch
        let signals = [0_i8; 3];
        let months = [1_i64; 3];
        let days = [1_i64; 3];
        let err = compute_forward_test_summary(ForwardTestInput {
            close: &close,
            high: &bad_high,
            low: &close,
            signals: &signals,
            months: &months,
            days: &days,
            timestamps: &[],
            settings: &flat_settings(),
        })
        .expect_err("length mismatch must be rejected");
        assert!(err.to_string().contains("length mismatch"));

        let err = compute_forward_test_summary(ForwardTestInput {
            close: &[],
            high: &[],
            low: &[],
            signals: &[],
            months: &[],
            days: &[],
            timestamps: &[],
            settings: &flat_settings(),
        })
        .expect_err("empty tail must be rejected");
        assert!(err.to_string().contains("at least one bar"));
    }

    fn sample_live_runtime_model() -> LiveExecutionRuntimeModel {
        LiveExecutionRuntimeModel {
            avg_slippage_pips: 0.4,
            avg_latency_ms: 35.0,
            spread_pips: 1.5,
            commission_per_trade: 7.0,
            partial_fill_rate: 0.05,
            kill_zone_blocking: true,
            backend_kind: "ctrader_live".to_string(),
        }
    }

    fn sample_live_summary() -> LiveExecutionSimulationSummary {
        LiveExecutionSimulationSummary {
            bars_simulated: 1_000,
            trades_simulated: 42,
            trades_blocked_by_kill_zone: 3,
            trades_partially_filled: 1,
            metrics: BacktestMetrics::from_metric_array([
                30.0, 1.4, 100_030.0, 0.025, 0.58, 1.7, 0.7, 0.0, 42.0, 0.82, 0.012,
            ]),
            runtime_model: sample_live_runtime_model(),
        }
    }

    #[test]
    fn live_execution_simulation_artifact_binds_runtime_model_and_temporal_scope() {
        let contract = temporal_contract("label-policy-v1");
        let model = sample_live_runtime_model();
        let scope = LiveExecutionSimulationScope::new(
            "dataset",
            "eval-config",
            "strategy",
            &model,
            &contract,
        )
        .expect("live execution scope construction should succeed");
        let artifact = LiveExecutionSimulationArtifactFile::new(scope, sample_live_summary());

        artifact
            .validate_for_temporal_contract(&contract)
            .expect("matching contract should accept the live-sim artifact");

        let drifted = temporal_contract("label-policy-v2");
        let err = artifact
            .validate_for_temporal_contract(&drifted)
            .expect_err("temporal drift must reject the live-sim artifact");
        assert!(err.to_string().contains("live execution simulation"));
    }

    #[test]
    fn live_execution_simulation_artifact_round_trips_through_atomic_io() {
        let contract = temporal_contract("label-policy-v1");
        let scope = LiveExecutionSimulationScope::new(
            "dataset",
            "eval-config",
            "strategy",
            &sample_live_runtime_model(),
            &contract,
        )
        .expect("scope construction should succeed");
        let artifact = LiveExecutionSimulationArtifactFile::new(scope, sample_live_summary());
        let path = temp_path("live-execution-simulation");

        write_live_execution_simulation_artifact_atomic(&path, &artifact)
            .expect("atomic live-sim artifact write should succeed");
        let loaded = read_live_execution_simulation_artifact(&path, &contract)
            .expect("matching live-sim artifact should load");
        assert_eq!(
            loaded.artifact_kind,
            LIVE_EXECUTION_SIMULATION_ARTIFACT_KIND
        );
        assert_eq!(loaded.summary.trades_simulated, 42);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn live_execution_simulation_artifact_rejects_wrong_kind_and_unsupported_schema() {
        let contract = temporal_contract("label-policy-v1");
        let scope = LiveExecutionSimulationScope::new(
            "dataset",
            "eval-config",
            "strategy",
            &sample_live_runtime_model(),
            &contract,
        )
        .expect("scope construction should succeed");
        let mut artifact = LiveExecutionSimulationArtifactFile::new(scope, sample_live_summary());
        artifact.artifact_kind = "canonical_strategy_backtest_artifact".to_string();
        let err = artifact
            .validate_for_temporal_contract(&contract)
            .expect_err("wrong artifact_kind must reject the live-sim load");
        assert!(
            err.to_string()
                .contains("live execution simulation artifact")
        );

        artifact.artifact_kind = LIVE_EXECUTION_SIMULATION_ARTIFACT_KIND.to_string();
        artifact.artifact_schema_version = LIVE_EXECUTION_SIMULATION_SCHEMA_VERSION + 1;
        let err = artifact
            .validate_for_temporal_contract(&contract)
            .expect_err("unsupported schema must reject the live-sim load");
        assert!(err.to_string().contains("live execution simulation schema"));
    }

    fn sample_prop_firm_trades() -> Vec<crate::quality::Trade> {
        vec![
            crate::quality::Trade {
                entry_time: 1_700_000_000_000,
                exit_time: Some(1_700_000_300_000),
                pnl: 800.0,
                pnl_pct: None,
                duration_hours: None,
                ..Default::default()
            },
            crate::quality::Trade {
                entry_time: 1_700_086_400_000,
                exit_time: Some(1_700_086_700_000),
                pnl: -400.0,
                pnl_pct: None,
                duration_hours: None,
                ..Default::default()
            },
            crate::quality::Trade {
                entry_time: 1_700_172_800_000,
                exit_time: Some(1_700_173_100_000),
                pnl: 600.0,
                pnl_pct: None,
                duration_hours: None,
                ..Default::default()
            },
        ]
    }

    #[test]
    fn prop_firm_risk_summary_passes_when_thresholds_are_respected() {
        let trades = sample_prop_firm_trades();
        // Relax the consistency knob — the 3-trade fixture has only two
        // winning days, so the larger winner naturally takes more than
        // the FTMO-default 50% share. The other defaults (loss limit,
        // overall DD, profit target) all pass for this fixture.
        let rules = PropFirmRiskRules {
            min_trading_days: 0,
            max_profit_consistency_ratio: 0.0,
            ..PropFirmRiskRules::default()
        };
        let summary = compute_prop_firm_risk_summary(PropFirmRiskInput {
            trades: &trades,
            initial_balance: 100_000.0,
            rules,
        });
        assert_eq!(summary.trades_observed, 3);
        assert_eq!(summary.trading_days_observed, 3);
        assert!(summary.all_rules_passed);
        assert!(!summary.daily_loss_breach);
        assert!(!summary.consistency_violation);
    }

    #[test]
    fn prop_firm_risk_rules_use_shared_relaxed_minimum_window() {
        let rules = PropFirmRiskRules::default();
        let defaults = neoethos_core::domain::prop_firm::PropFirmChallengeDefaults::FTMO_STANDARD;

        assert_eq!(
            rules.min_trading_days,
            defaults.relaxed_min_trading_days as usize
        );
    }

    #[test]
    fn prop_firm_risk_summary_flags_daily_loss_breach() {
        let trades = vec![crate::quality::Trade {
            entry_time: 1_700_000_000_000,
            exit_time: Some(1_700_000_300_000),
            pnl: -7_000.0,
            pnl_pct: None,
            duration_hours: None,
            ..Default::default()
        }];
        let summary = compute_prop_firm_risk_summary(PropFirmRiskInput {
            trades: &trades,
            initial_balance: 100_000.0,
            rules: PropFirmRiskRules::default(),
        });
        assert!(summary.daily_loss_breach);
        assert!(!summary.all_rules_passed);
        assert!(summary.max_daily_loss_pct_observed >= 0.05);
    }

    #[test]
    fn prop_firm_risk_artifact_round_trips_and_rejects_drift() {
        let contract = temporal_contract("label-policy-v1");
        let rules = PropFirmRiskRules::default();
        let scope = PropFirmRiskValidationScope::new(
            "dataset",
            "eval-config",
            "strategy",
            &rules,
            &contract,
        )
        .expect("prop-firm scope construction should succeed");
        let summary = compute_prop_firm_risk_summary(PropFirmRiskInput {
            trades: &sample_prop_firm_trades(),
            initial_balance: 100_000.0,
            rules,
        });
        let artifact = PropFirmRiskValidationArtifactFile::new(scope, summary);
        let path = temp_path("prop-firm-risk-validation");

        write_prop_firm_risk_validation_artifact_atomic(&path, &artifact)
            .expect("atomic prop-firm artifact write should succeed");
        let loaded = read_prop_firm_risk_validation_artifact(&path, &contract)
            .expect("matching prop-firm artifact should load");
        assert_eq!(
            loaded.artifact_kind,
            PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND
        );

        let drifted = temporal_contract("label-policy-v2");
        let err = read_prop_firm_risk_validation_artifact(&path, &drifted)
            .expect_err("temporal drift must reject the prop-firm artifact load");
        assert!(err.to_string().contains("temporal_contract_hash"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn prop_firm_risk_artifact_rejects_wrong_kind_and_unsupported_schema() {
        let contract = temporal_contract("label-policy-v1");
        let rules = PropFirmRiskRules::default();
        let scope = PropFirmRiskValidationScope::new(
            "dataset",
            "eval-config",
            "strategy",
            &rules,
            &contract,
        )
        .expect("scope construction should succeed");
        let summary = compute_prop_firm_risk_summary(PropFirmRiskInput {
            trades: &sample_prop_firm_trades(),
            initial_balance: 100_000.0,
            rules,
        });
        let mut artifact = PropFirmRiskValidationArtifactFile::new(scope, summary);
        artifact.artifact_kind = "live_execution_simulation_artifact".to_string();
        let err = artifact
            .validate_for_temporal_contract(&contract)
            .expect_err("wrong artifact_kind must reject the prop-firm load");
        assert!(
            err.to_string()
                .contains("prop-firm risk validation artifact")
        );

        artifact.artifact_kind = PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND.to_string();
        artifact.artifact_schema_version = PROP_FIRM_RISK_VALIDATION_SCHEMA_VERSION + 1;
        let err = artifact
            .validate_for_temporal_contract(&contract)
            .expect_err("unsupported schema must reject the prop-firm load");
        assert!(err.to_string().contains("prop-firm risk validation schema"));
    }
}
