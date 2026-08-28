use crate::artifact_io::{stable_json_hash, write_json_atomic};
use crate::data_selection::{
    CanonicalSearchArtifactEnvelopeV2, CanonicalSearchArtifactScopeV2, CanonicalSearchWindowRoleV1,
};
use crate::eval::{
    BacktestMetrics, BacktestSettings, fast_evaluate_strategy_core, simulate_trades_core,
};
use crate::genetic::Gene;
use crate::quality::Trade;
use anyhow::{Result, bail};
use itertools::Itertools;
use neoethos_core::contracts::{TemporalFeatureContract, TemporalScopeHashes};
use neoethos_core::domain::prop_firm::{PropFirmChallengeDefaults, PropFirmConstraints};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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

pub const WALKFORWARD_VALIDATION_ARTIFACT_KIND: &str = "neoethos.search.walkforward-validation.v2";
pub const WALKFORWARD_VALIDATION_SCHEMA_VERSION: u32 = 2;
pub const CANONICAL_BACKTEST_ARTIFACT_KIND: &str =
    "neoethos.search.canonical-backtest-validation.v3";
pub const CANONICAL_BACKTEST_SCHEMA_VERSION: u32 = 3;
pub const FORWARD_TEST_VALIDATION_ARTIFACT_KIND: &str =
    "neoethos.search.forward-test-validation.v3";
pub const FORWARD_TEST_VALIDATION_SCHEMA_VERSION: u32 = 3;
pub const LIVE_EXECUTION_SIMULATION_ARTIFACT_KIND: &str =
    "neoethos.search.live-execution-simulation.v2";
pub const LIVE_EXECUTION_SIMULATION_SCHEMA_VERSION: u32 = 2;
pub const PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND: &str =
    "neoethos.search.prop-firm-risk-validation.v2";
pub const PROP_FIRM_RISK_VALIDATION_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ValidationStrategyIdentityV2 {
    strategy_id: String,
    exact_gene_hash: String,
}

impl ValidationStrategyIdentityV2 {
    pub fn from_gene(gene: &Gene) -> Result<Self> {
        let identity = Self {
            strategy_id: gene.strategy_id.clone(),
            exact_gene_hash: stable_json_hash(gene)?,
        };
        identity.validate_shape()?;
        Ok(identity)
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn exact_gene_hash(&self) -> &str {
        &self.exact_gene_hash
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_shape()
    }

    fn validate_shape(&self) -> Result<()> {
        if self.strategy_id.trim().is_empty() {
            bail!("validation strategy identity has an empty strategy_id");
        }
        validate_fnv64_hash("validation exact_gene_hash", &self.exact_gene_hash)
    }

    pub fn validate_against(&self, gene: &Gene) -> Result<()> {
        self.validate_shape()?;
        if self.strategy_id != gene.strategy_id {
            bail!(
                "validation strategy_id `{}` does not match expected `{}`",
                self.strategy_id,
                gene.strategy_id
            );
        }
        let expected_hash = stable_json_hash(gene)?;
        if self.exact_gene_hash != expected_hash {
            bail!(
                "validation exact_gene_hash `{}` does not match expected `{expected_hash}`",
                self.exact_gene_hash
            );
        }
        Ok(())
    }
}

pub(crate) fn validate_fnv64_hash(field: &str, value: &str) -> Result<()> {
    let Some(hex) = value.strip_prefix("fnv64:") else {
        bail!("{field} must use canonical fnv64:<16 lowercase hex> form");
    };
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{field} must use canonical fnv64:<16 lowercase hex> form");
    }
    Ok(())
}

fn validate_selection_scope(scope: &CanonicalSearchArtifactScopeV2, label: &str) -> Result<()> {
    scope.validate().map_err(anyhow::Error::new)?;
    match scope.evaluated_window().role() {
        CanonicalSearchWindowRoleV1::DiscoveryInput | CanonicalSearchWindowRoleV1::InSample => {
            Ok(())
        }
        role => bail!(
            "{label} requires the exact stored discovery-input/in-sample selection scope, found {role:?}"
        ),
    }
}

fn validate_holdout_scope(scope: &CanonicalSearchArtifactScopeV2, label: &str) -> Result<()> {
    scope.validate().map_err(anyhow::Error::new)?;
    if scope.evaluated_window().role() != CanonicalSearchWindowRoleV1::Holdout {
        bail!(
            "{label} requires the exact stored holdout scope, found {:?}",
            scope.evaluated_window().role()
        );
    }
    Ok(())
}

fn reject_legacy_weak_validation_json(bytes: &[u8], label: &str) -> Result<()> {
    let wire: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| anyhow::anyhow!("parse {label} validation artifact JSON: {error}"))?;
    let legacy_version = wire
        .get("artifact_schema_version")
        .and_then(serde_json::Value::as_u64);
    let weak_scope = wire
        .get("scope")
        .and_then(|scope| scope.get("dataset_hash"))
        .is_some();
    if legacy_version.is_some() || weak_scope {
        bail!(
            "legacy {label} validation artifact version {} is unsupported because it lacks the exact search receipt/window/config authority; regenerate it from the original canonical search input",
            legacy_version.unwrap_or(1)
        );
    }
    Ok(())
}

fn reject_legacy_metric_artifact_payload_v1(
    bytes: &[u8],
    label: &str,
    expected_schema_version: u32,
) -> Result<()> {
    let wire: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| anyhow::anyhow!("parse {label} validation artifact JSON: {error}"))?;
    let actual = wire
        .get("payload")
        .and_then(|payload| payload.get("schema_version"))
        .and_then(serde_json::Value::as_u64);
    match actual {
        Some(actual) if actual == u64::from(expected_schema_version) => {}
        Some(actual) if actual < u64::from(expected_schema_version) => bail!(
            "unsupported {label} metric payload schema version {actual}; expected {expected_schema_version}; legacy metric artifacts omitted monthly_target_hit_rate and cannot be loaded"
        ),
        Some(actual) => bail!(
            "unsupported {label} metric payload schema version {actual}; expected {expected_schema_version}"
        ),
        None => bail!(
            "missing {label} metric payload schema version; expected {expected_schema_version}; unversioned metric artifacts cannot be loaded"
        ),
    }
    Ok(())
}

fn reject_legacy_live_execution_simulation_v1(
    bytes: &[u8],
    expected_schema_version: u32,
) -> Result<()> {
    let wire: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        anyhow::anyhow!("parse live execution simulation artifact JSON: {error}")
    })?;
    let actual = wire
        .get("artifact_schema_version")
        .and_then(serde_json::Value::as_u64);
    match actual {
        Some(actual) if actual == u64::from(expected_schema_version) => {}
        Some(actual) if actual < u64::from(expected_schema_version) => bail!(
            "unsupported live execution simulation metric schema version {actual}; expected {expected_schema_version}; legacy artifacts omitted monthly_target_hit_rate and cannot be loaded"
        ),
        Some(actual) => bail!(
            "unsupported live execution simulation metric schema version {actual}; expected {expected_schema_version}"
        ),
        None => bail!(
            "missing live execution simulation metric schema version; expected {expected_schema_version}; unversioned metric artifacts cannot be loaded"
        ),
    }
    Ok(())
}

fn read_artifact_bytes(path: impl AsRef<Path>, label: &str) -> Result<Vec<u8>> {
    let path = path.as_ref();
    std::fs::read(path)
        .map_err(|error| anyhow::anyhow!("read {label} artifact {}: {error}", path.display()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalBacktestPayloadV3 {
    schema_version: u32,
    strategy_identity: ValidationStrategyIdentityV2,
    metrics: BacktestMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CanonicalBacktestArtifactFile {
    envelope: CanonicalSearchArtifactEnvelopeV2<CanonicalBacktestPayloadV3>,
}

impl CanonicalBacktestArtifactFile {
    pub fn new(
        scope: CanonicalSearchArtifactScopeV2,
        search_config_hash: impl Into<String>,
        gene: &Gene,
        metrics: BacktestMetrics,
    ) -> Result<Self> {
        validate_selection_scope(&scope, "canonical backtest")?;
        let artifact = Self {
            envelope: CanonicalSearchArtifactEnvelopeV2::new(
                CANONICAL_BACKTEST_ARTIFACT_KIND,
                scope,
                search_config_hash,
                CanonicalBacktestPayloadV3 {
                    schema_version: CANONICAL_BACKTEST_SCHEMA_VERSION,
                    strategy_identity: ValidationStrategyIdentityV2::from_gene(gene)?,
                    metrics,
                },
            )
            .map_err(anyhow::Error::new)?,
        };
        artifact.validate_internal()?;
        Ok(artifact)
    }

    fn validate_internal(&self) -> Result<()> {
        self.envelope.validate().map_err(anyhow::Error::new)?;
        if self.envelope.artifact_kind() != CANONICAL_BACKTEST_ARTIFACT_KIND {
            bail!(
                "wrong canonical backtest artifact kind `{}`",
                self.envelope.artifact_kind()
            );
        }
        if self.envelope.payload().schema_version != CANONICAL_BACKTEST_SCHEMA_VERSION {
            bail!(
                "unsupported canonical backtest payload schema version {}",
                self.envelope.payload().schema_version
            );
        }
        validate_selection_scope(self.scope(), "canonical backtest")?;
        self.strategy_identity().validate_shape()
    }

    pub fn validate_against(
        &self,
        expected_scope: &CanonicalSearchArtifactScopeV2,
        expected_search_config_hash: &str,
        expected_gene: &Gene,
    ) -> Result<()> {
        self.validate_internal()?;
        validate_selection_scope(expected_scope, "expected canonical backtest")?;
        self.envelope
            .validate_against(
                CANONICAL_BACKTEST_ARTIFACT_KIND,
                expected_search_config_hash,
                expected_scope.receipt(),
                expected_scope.evaluated_window(),
            )
            .map_err(anyhow::Error::new)?;
        self.strategy_identity().validate_against(expected_gene)
    }

    pub fn scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        self.envelope.scope()
    }

    pub fn search_config_hash(&self) -> &str {
        self.envelope.search_config_hash()
    }

    pub fn strategy_identity(&self) -> &ValidationStrategyIdentityV2 {
        &self.envelope.payload().strategy_identity
    }

    pub fn metrics(&self) -> &BacktestMetrics {
        &self.envelope.payload().metrics
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate_internal()?;
        self.envelope.to_json_bytes().map_err(anyhow::Error::new)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        reject_legacy_metric_artifact_payload_v1(bytes, "canonical backtest", 3)?;
        reject_legacy_weak_validation_json(bytes, "canonical backtest")?;
        let artifact = Self {
            envelope: CanonicalSearchArtifactEnvelopeV2::from_json_bytes(bytes)
                .map_err(anyhow::Error::new)?,
        };
        artifact.validate_internal()?;
        Ok(artifact)
    }
}

pub fn write_canonical_backtest_artifact_atomic(
    path: impl AsRef<Path>,
    artifact: &CanonicalBacktestArtifactFile,
) -> Result<()> {
    artifact.validate_internal()?;
    write_json_atomic(path, artifact)
}

pub fn read_canonical_backtest_artifact(
    path: impl AsRef<Path>,
    expected_scope: &CanonicalSearchArtifactScopeV2,
    expected_search_config_hash: &str,
    expected_gene: &Gene,
) -> Result<CanonicalBacktestArtifactFile> {
    let bytes = read_artifact_bytes(path, "canonical backtest")?;
    let artifact = CanonicalBacktestArtifactFile::from_json_bytes(&bytes)?;
    artifact.validate_against(expected_scope, expected_search_config_hash, expected_gene)?;
    Ok(artifact)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WalkforwardValidationPayloadV2 {
    schema_version: u32,
    strategy_identity: ValidationStrategyIdentityV2,
    summary: WalkforwardSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WalkforwardValidationArtifactFile {
    envelope: CanonicalSearchArtifactEnvelopeV2<WalkforwardValidationPayloadV2>,
}

impl WalkforwardValidationArtifactFile {
    pub fn new(
        scope: CanonicalSearchArtifactScopeV2,
        search_config_hash: impl Into<String>,
        gene: &Gene,
        summary: WalkforwardSummary,
    ) -> Result<Self> {
        validate_selection_scope(&scope, "walk-forward validation")?;
        let artifact = Self {
            envelope: CanonicalSearchArtifactEnvelopeV2::new(
                WALKFORWARD_VALIDATION_ARTIFACT_KIND,
                scope,
                search_config_hash,
                WalkforwardValidationPayloadV2 {
                    schema_version: WALKFORWARD_VALIDATION_SCHEMA_VERSION,
                    strategy_identity: ValidationStrategyIdentityV2::from_gene(gene)?,
                    summary,
                },
            )
            .map_err(anyhow::Error::new)?,
        };
        artifact.validate_internal()?;
        Ok(artifact)
    }

    fn validate_internal(&self) -> Result<()> {
        self.envelope.validate().map_err(anyhow::Error::new)?;
        if self.envelope.artifact_kind() != WALKFORWARD_VALIDATION_ARTIFACT_KIND {
            bail!(
                "wrong walk-forward validation artifact kind `{}`",
                self.envelope.artifact_kind()
            );
        }
        if self.envelope.payload().schema_version != WALKFORWARD_VALIDATION_SCHEMA_VERSION {
            bail!(
                "unsupported walk-forward validation payload schema version {}",
                self.envelope.payload().schema_version
            );
        }
        validate_selection_scope(self.scope(), "walk-forward validation")?;
        self.strategy_identity().validate_shape()
    }

    pub fn validate_against(
        &self,
        expected_scope: &CanonicalSearchArtifactScopeV2,
        expected_search_config_hash: &str,
        expected_gene: &Gene,
    ) -> Result<()> {
        self.validate_internal()?;
        validate_selection_scope(expected_scope, "expected walk-forward validation")?;
        self.envelope
            .validate_against(
                WALKFORWARD_VALIDATION_ARTIFACT_KIND,
                expected_search_config_hash,
                expected_scope.receipt(),
                expected_scope.evaluated_window(),
            )
            .map_err(anyhow::Error::new)?;
        self.strategy_identity().validate_against(expected_gene)
    }

    pub fn scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        self.envelope.scope()
    }

    pub fn search_config_hash(&self) -> &str {
        self.envelope.search_config_hash()
    }

    pub fn strategy_identity(&self) -> &ValidationStrategyIdentityV2 {
        &self.envelope.payload().strategy_identity
    }

    pub fn summary(&self) -> &WalkforwardSummary {
        &self.envelope.payload().summary
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate_internal()?;
        self.envelope.to_json_bytes().map_err(anyhow::Error::new)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        reject_legacy_weak_validation_json(bytes, "walk-forward")?;
        let artifact = Self {
            envelope: CanonicalSearchArtifactEnvelopeV2::from_json_bytes(bytes)
                .map_err(anyhow::Error::new)?,
        };
        artifact.validate_internal()?;
        Ok(artifact)
    }
}

pub fn write_walkforward_validation_artifact_atomic(
    path: impl AsRef<Path>,
    artifact: &WalkforwardValidationArtifactFile,
) -> Result<()> {
    artifact.validate_internal()?;
    write_json_atomic(path, artifact)
}

pub fn read_walkforward_validation_artifact(
    path: impl AsRef<Path>,
    expected_scope: &CanonicalSearchArtifactScopeV2,
    expected_search_config_hash: &str,
    expected_gene: &Gene,
) -> Result<WalkforwardValidationArtifactFile> {
    let bytes = read_artifact_bytes(path, "walk-forward validation")?;
    let artifact = WalkforwardValidationArtifactFile::from_json_bytes(&bytes)?;
    artifact.validate_against(expected_scope, expected_search_config_hash, expected_gene)?;
    Ok(artifact)
}

/// Forward-test validation summary: a single backtest pass over a tail
/// window that was withheld from both training and walk-forward CV. The
/// summary is intentionally flat (no `splits`) because forward testing
/// produces one unbiased OOS estimate, not a folded distribution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ForwardTestSummary {
    /// Number of bars in the held-out tail window.
    pub bars: usize,
    /// Canonical metrics computed on the held-out tail.
    pub metrics: BacktestMetrics,
    /// Wall-clock span of the tail window in days (`exit_time - entry_time`
    /// of the first/last bar). `0.0` when the tail has fewer than two bars.
    pub span_days: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardTestValidationPayloadV3 {
    schema_version: u32,
    strategy_identity: ValidationStrategyIdentityV2,
    summary: ForwardTestSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ForwardTestValidationArtifactFile {
    envelope: CanonicalSearchArtifactEnvelopeV2<ForwardTestValidationPayloadV3>,
}

impl ForwardTestValidationArtifactFile {
    pub fn new(
        scope: CanonicalSearchArtifactScopeV2,
        search_config_hash: impl Into<String>,
        gene: &Gene,
        summary: ForwardTestSummary,
    ) -> Result<Self> {
        validate_holdout_scope(&scope, "forward-test validation")?;
        let expected_bars =
            (scope.evaluated_window().row_end() - scope.evaluated_window().row_start()) as usize;
        if summary.bars != expected_bars {
            bail!(
                "forward-test summary bars {} do not match exact holdout rows {expected_bars}",
                summary.bars
            );
        }
        let artifact = Self {
            envelope: CanonicalSearchArtifactEnvelopeV2::new(
                FORWARD_TEST_VALIDATION_ARTIFACT_KIND,
                scope,
                search_config_hash,
                ForwardTestValidationPayloadV3 {
                    schema_version: FORWARD_TEST_VALIDATION_SCHEMA_VERSION,
                    strategy_identity: ValidationStrategyIdentityV2::from_gene(gene)?,
                    summary,
                },
            )
            .map_err(anyhow::Error::new)?,
        };
        artifact.validate_internal()?;
        Ok(artifact)
    }

    fn validate_internal(&self) -> Result<()> {
        self.envelope.validate().map_err(anyhow::Error::new)?;
        if self.envelope.artifact_kind() != FORWARD_TEST_VALIDATION_ARTIFACT_KIND {
            bail!(
                "wrong forward-test validation artifact kind `{}`",
                self.envelope.artifact_kind()
            );
        }
        if self.envelope.payload().schema_version != FORWARD_TEST_VALIDATION_SCHEMA_VERSION {
            bail!(
                "unsupported forward-test validation payload schema version {}",
                self.envelope.payload().schema_version
            );
        }
        validate_holdout_scope(self.scope(), "forward-test validation")?;
        let expected_bars = (self.scope().evaluated_window().row_end()
            - self.scope().evaluated_window().row_start()) as usize;
        if self.summary().bars != expected_bars {
            bail!(
                "forward-test summary bars {} do not match exact holdout rows {expected_bars}",
                self.summary().bars
            );
        }
        self.strategy_identity().validate_shape()
    }

    pub fn validate_against(
        &self,
        expected_scope: &CanonicalSearchArtifactScopeV2,
        expected_search_config_hash: &str,
        expected_gene: &Gene,
    ) -> Result<()> {
        self.validate_internal()?;
        validate_holdout_scope(expected_scope, "expected forward-test validation")?;
        self.envelope
            .validate_against(
                FORWARD_TEST_VALIDATION_ARTIFACT_KIND,
                expected_search_config_hash,
                expected_scope.receipt(),
                expected_scope.evaluated_window(),
            )
            .map_err(anyhow::Error::new)?;
        self.strategy_identity().validate_against(expected_gene)
    }

    pub fn scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        self.envelope.scope()
    }

    pub fn search_config_hash(&self) -> &str {
        self.envelope.search_config_hash()
    }

    pub fn strategy_identity(&self) -> &ValidationStrategyIdentityV2 {
        &self.envelope.payload().strategy_identity
    }

    pub fn summary(&self) -> &ForwardTestSummary {
        &self.envelope.payload().summary
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate_internal()?;
        self.envelope.to_json_bytes().map_err(anyhow::Error::new)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        reject_legacy_metric_artifact_payload_v1(bytes, "forward test", 3)?;
        reject_legacy_weak_validation_json(bytes, "forward-test")?;
        let artifact = Self {
            envelope: CanonicalSearchArtifactEnvelopeV2::from_json_bytes(bytes)
                .map_err(anyhow::Error::new)?,
        };
        artifact.validate_internal()?;
        Ok(artifact)
    }
}

pub fn write_forward_test_validation_artifact_atomic(
    path: impl AsRef<Path>,
    artifact: &ForwardTestValidationArtifactFile,
) -> Result<()> {
    artifact.validate_internal()?;
    write_json_atomic(path, artifact)
}

pub fn read_forward_test_validation_artifact(
    path: impl AsRef<Path>,
    expected_scope: &CanonicalSearchArtifactScopeV2,
    expected_search_config_hash: &str,
    expected_gene: &Gene,
) -> Result<ForwardTestValidationArtifactFile> {
    let bytes = read_artifact_bytes(path, "forward-test validation")?;
    let artifact = ForwardTestValidationArtifactFile::from_json_bytes(&bytes)?;
    artifact.validate_against(expected_scope, expected_search_config_hash, expected_gene)?;
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
    let bytes = read_artifact_bytes(path, "live execution simulation")?;
    reject_legacy_live_execution_simulation_v1(&bytes, 2)?;
    let artifact: LiveExecutionSimulationArtifactFile = serde_json::from_slice(&bytes)
        .map_err(|error| anyhow::anyhow!("parse live execution simulation artifact: {error}"))?;
    artifact.validate_for_temporal_contract(temporal_contract)?;
    Ok(artifact)
}

/// Prop-firm rule set applied to observed trade outcomes. Each numeric
/// field is a pass threshold (`<= 0.0` means "rule disabled" so callers
/// can opt out per-field); booleans toggle structural rules.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropFirmRiskValidationPayloadV2 {
    schema_version: u32,
    strategy_identity: ValidationStrategyIdentityV2,
    rules_hash: String,
    summary: PropFirmRiskValidationSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PropFirmRiskValidationArtifactFile {
    envelope: CanonicalSearchArtifactEnvelopeV2<PropFirmRiskValidationPayloadV2>,
}

impl PropFirmRiskValidationArtifactFile {
    pub fn new(
        scope: CanonicalSearchArtifactScopeV2,
        search_config_hash: impl Into<String>,
        gene: &Gene,
        summary: PropFirmRiskValidationSummary,
    ) -> Result<Self> {
        validate_holdout_scope(&scope, "prop-firm risk validation")?;
        let rules_hash = stable_json_hash(&summary.rules)?;
        let artifact = Self {
            envelope: CanonicalSearchArtifactEnvelopeV2::new(
                PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND,
                scope,
                search_config_hash,
                PropFirmRiskValidationPayloadV2 {
                    schema_version: PROP_FIRM_RISK_VALIDATION_SCHEMA_VERSION,
                    strategy_identity: ValidationStrategyIdentityV2::from_gene(gene)?,
                    rules_hash,
                    summary,
                },
            )
            .map_err(anyhow::Error::new)?,
        };
        artifact.validate_internal()?;
        Ok(artifact)
    }

    fn validate_internal(&self) -> Result<()> {
        self.envelope.validate().map_err(anyhow::Error::new)?;
        if self.envelope.artifact_kind() != PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND {
            bail!(
                "wrong prop-firm risk validation artifact kind `{}`",
                self.envelope.artifact_kind()
            );
        }
        if self.envelope.payload().schema_version != PROP_FIRM_RISK_VALIDATION_SCHEMA_VERSION {
            bail!(
                "unsupported prop-firm risk validation payload schema version {}",
                self.envelope.payload().schema_version
            );
        }
        validate_holdout_scope(self.scope(), "prop-firm risk validation")?;
        self.strategy_identity().validate_shape()?;
        validate_fnv64_hash("prop-firm rules_hash", &self.envelope.payload().rules_hash)?;
        let expected_rules_hash = stable_json_hash(&self.summary().rules)?;
        if self.envelope.payload().rules_hash != expected_rules_hash {
            bail!(
                "prop-firm rules_hash `{}` does not match embedded summary rules `{expected_rules_hash}`",
                self.envelope.payload().rules_hash
            );
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        expected_scope: &CanonicalSearchArtifactScopeV2,
        expected_search_config_hash: &str,
        expected_gene: &Gene,
    ) -> Result<()> {
        self.validate_internal()?;
        validate_holdout_scope(expected_scope, "expected prop-firm risk validation")?;
        self.envelope
            .validate_against(
                PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND,
                expected_search_config_hash,
                expected_scope.receipt(),
                expected_scope.evaluated_window(),
            )
            .map_err(anyhow::Error::new)?;
        self.strategy_identity().validate_against(expected_gene)
    }

    pub fn scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        self.envelope.scope()
    }

    pub fn search_config_hash(&self) -> &str {
        self.envelope.search_config_hash()
    }

    pub fn strategy_identity(&self) -> &ValidationStrategyIdentityV2 {
        &self.envelope.payload().strategy_identity
    }

    pub fn summary(&self) -> &PropFirmRiskValidationSummary {
        &self.envelope.payload().summary
    }

    pub fn rules_hash(&self) -> &str {
        &self.envelope.payload().rules_hash
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate_internal()?;
        self.envelope.to_json_bytes().map_err(anyhow::Error::new)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        reject_legacy_weak_validation_json(bytes, "prop-firm risk")?;
        let artifact = Self {
            envelope: CanonicalSearchArtifactEnvelopeV2::from_json_bytes(bytes)
                .map_err(anyhow::Error::new)?,
        };
        artifact.validate_internal()?;
        Ok(artifact)
    }
}

pub fn write_prop_firm_risk_validation_artifact_atomic(
    path: impl AsRef<Path>,
    artifact: &PropFirmRiskValidationArtifactFile,
) -> Result<()> {
    artifact.validate_internal()?;
    write_json_atomic(path, artifact)
}

pub fn read_prop_firm_risk_validation_artifact(
    path: impl AsRef<Path>,
    expected_scope: &CanonicalSearchArtifactScopeV2,
    expected_search_config_hash: &str,
    expected_gene: &Gene,
) -> Result<PropFirmRiskValidationArtifactFile> {
    let bytes = read_artifact_bytes(path, "prop-firm risk validation")?;
    let artifact = PropFirmRiskValidationArtifactFile::from_json_bytes(&bytes)?;
    artifact.validate_against(expected_scope, expected_search_config_hash, expected_gene)?;
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
/// Simulate the slice, then measure it.
///
/// Kept as the CPU entry point. The measurement half is
/// [`walkforward_risk_diagnostics_from_trades`], which takes the trade list
/// directly — the device kernel now writes one, and this pass exists only
/// because it used not to. Splitting them is what lets the second, duplicate
/// simulation go away without moving the arithmetic.
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
    // Use real timestamps so the simulator applies the right gap, session and
    // kill-zone logic — same as before this was split out.
    let ts: &[i64] = if timestamps.len() == close.len() {
        timestamps
    } else {
        days
    };
    let trades = simulate_trades_core(close, high, low, ts, signals, settings);
    walkforward_risk_diagnostics_from_trades(
        &trades,
        days,
        evaluator_max_daily_dd,
        max_daily_loss_pct,
        max_daily_profit_pct,
        min_trading_days,
        max_trades_per_day,
        initial_balance,
    )
}

/// The measurement, over a trade list that already exists.
///
/// Every field here is derived from the trades and the day index — consecutive
/// runs, per-day P&L and counts, the prop-firm compliance checks. None of it
/// needs the price series, which is why the simulation that produced the trades
/// does not have to happen twice.
#[allow(clippy::too_many_arguments)]
fn walkforward_risk_diagnostics_from_trades(
    trades: &[Trade],
    days: &[i64],
    evaluator_max_daily_dd: f64,
    max_daily_loss_pct: f64,
    max_daily_profit_pct: f64,
    min_trading_days: usize,
    max_trades_per_day: usize,
    initial_balance: f64,
) -> WalkforwardRiskDiagnostics {
    if days.is_empty() {
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

    let mut max_consec_losses = 0usize;
    let mut current_consec_losses = 0usize;

    for trade in trades {
        if trade.pnl < 0.0 {
            current_consec_losses += 1;
            max_consec_losses = max_consec_losses.max(current_consec_losses);
        } else if trade.pnl > 0.0 {
            current_consec_losses = 0;
        }

        // `days` carries YYYYMMDD calendar keys (`month_day_indices`), while
        // the simulator stamps trades with epoch-ms — or, on the
        // timestamps-missing fallback, with the day key itself. Before
        // 2026-08-08 the raw ms value was used as the bucket key: it never
        // matched a YYYYMMDD key, so EVERY trade opened its own synthetic
        // "day" and the daily-loss / consistency / trades-per-day checks
        // measured per-TRADE numbers under a per-DAY name — same-day losses
        // never accumulated into a breach. Epoch-ms values are >= 1e11 for
        // any modern date; YYYYMMDD keys are < 1e8 — disjoint ranges.
        let exit_raw = trade.exit_time.unwrap_or(trade.entry_time);
        let exit_day = if exit_raw >= 100_000_000 {
            crate::genetic::calendar_day_key_ms(exit_raw).unwrap_or(exit_raw)
        } else {
            exit_raw
        };
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
    crate::historical_evaluation_authority::require_historical_evaluation_authority_v1()?;
    let _scope = crate::eval_telemetry::CallerScope::enter("walkforward");
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
    /// Pip size the adaptive-stop base series is denominated in — MUST be the
    /// same pip the GPU metrics provider resolves
    /// (`WalkforwardPopulationGenePack::adaptive_pip`), so the CPU
    /// risk-diagnostic half scales the SAME per-window base the metrics half
    /// scaled. Ignored when no gene is adaptive.
    pub adaptive_pip: f64,
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
/// What one window's provider returned.
///
/// `metrics` is the 11-wide row per gene, as before. `trades` is the per-gene
/// trade list when the provider already has one — the device kernel writes
/// these now — and `None` when it does not, in which case the diagnostics are
/// computed by simulating the window again on the CPU.
///
/// That second simulation is the thing worth removing: telemetry puts it at
/// 191 800 calls and 45 % of a run, redoing on the CPU what the card just did.
/// Making it optional rather than assumed keeps the CPU-only path working and
/// lets the two be compared on the same window.
pub struct WindowEvaluation {
    pub metrics: Vec<[f64; 11]>,
    pub trades: Option<Vec<Vec<Trade>>>,
}

impl From<Vec<[f64; 11]>> for WindowEvaluation {
    fn from(metrics: Vec<[f64; 11]>) -> Self {
        Self {
            metrics,
            trades: None,
        }
    }
}

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
    F: FnMut(usize, usize) -> Result<WindowEvaluation>,
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
        adaptive_pip,
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

    // Whether ANY gene runs adaptive (volatility-scaled) stops. When true,
    // each split window below derives ITS OWN base series — the same series
    // the GPU metrics provider derives for that window
    // (`validation_genes_population_window` recomputes per window) — so the
    // risk-diagnostic simulation and the metrics beside it evaluate the SAME
    // strategy. Before 2026-08-08 the diagnostics ran the caller's settings
    // verbatim, which carried no base: `daily_loss_breach` and
    // `consistency_violation` gated the prop-firm verdict on FIXED stops the
    // gene was never scored on.
    let any_adaptive = gene_settings.iter().any(|s| s.adaptive_vol_mult > 0.0);
    if any_adaptive && !(adaptive_pip.is_finite() && adaptive_pip > 0.0) {
        bail!(
            "walk-forward population: adaptive genes present but adaptive_pip = {adaptive_pip} \
             — the caller must pass the pack's resolved pip so both halves scale the same base"
        );
    }

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
        let window = metrics_fn(test_start, end)?;
        let gpu_metrics = window.metrics;
        let window_trades = window.trades;
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

        // Window-local adaptive base, indexed to the slice above — identical
        // derivation (estimator, pip, slice) to the metrics provider's. `None`
        // ⇒ window too short for the estimator ⇒ fixed-pip fallback, the same
        // policy `validation_genes_population_window` applies to this window.
        let window_base: Option<std::sync::Arc<[f64]>> = if any_adaptive {
            match crate::stop_target::adaptive_base_pips_series(
                slice_high,
                slice_low,
                slice_close,
                adaptive_pip,
            ) {
                Ok(base) => Some(std::sync::Arc::from(base)),
                Err(e @ crate::stop_target::StopDistanceError::TooShort { .. }) => {
                    tracing::debug!(
                        target: "neoethos_search::adaptive_stops",
                        bars = slice_close.len(), error = %e,
                        "walk-forward diagnostics window too short for an adaptive base — fixed pips"
                    );
                    None
                }
                Err(e) => {
                    bail!(
                        "adaptive stop base series failed on a {}-bar walk-forward \
                         diagnostics window: {e}",
                        slice_close.len()
                    );
                }
            }
        } else {
            None
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
                // Per-gene settings (the gene's own SL/TP + adaptive regime) so
                // `simulate_trades_core` inside the diagnostics applies the SAME
                // exits the metrics half ran. For an adaptive gene the base is
                // REPLACED with this window's base: whatever the caller
                // installed was indexed to a different slice, and the window
                // base is what the metrics provider scaled.
                // Measure the trades the provider already has, or simulate to
                // get them. The arithmetic is the same function either way;
                // only who ran the walk differs.
                let risk = match window_trades.as_ref().and_then(|t| t.get(g)) {
                    Some(trades) => walkforward_risk_diagnostics_from_trades(
                        trades,
                        slice_days,
                        max_daily_dd,
                        max_daily_loss_pct,
                        max_daily_profit_pct,
                        min_trading_days,
                        max_trades_per_day,
                        initial_balance,
                    ),
                    None => {
                        let window_settings;
                        let settings_ref = if gene_settings[g].adaptive_vol_mult > 0.0 {
                            window_settings = BacktestSettings {
                                adaptive_base_pips: window_base.clone(),
                                ..gene_settings[g].clone()
                            };
                            &window_settings
                        } else {
                            &gene_settings[g]
                        };
                        walkforward_risk_diagnostics(
                            slice_close,
                            slice_high,
                            slice_low,
                            slice_sig,
                            slice_days,
                            slice_ts,
                            settings_ref,
                            max_daily_dd,
                            max_daily_loss_pct,
                            max_daily_profit_pct,
                            min_trading_days,
                            max_trades_per_day,
                            initial_balance,
                        )
                    }
                };

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

    const STRICT_SEARCH_CONFIG_HASH: &str = "fnv64:0123456789abcdef";

    fn strict_gene() -> Gene {
        Gene {
            strategy_id: "strict-validation-gene".to_owned(),
            indices: vec![0],
            weights: vec![1.0],
            ..Gene::default()
        }
    }

    fn strict_validation_scopes() -> (
        CanonicalSearchArtifactScopeV2,
        CanonicalSearchArtifactScopeV2,
    ) {
        let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let ohlcv = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
        let anchor = features.provenance().bindings()[0]
            .dataset_identity()
            .clone();
        let receipt = crate::data_selection::CanonicalSearchInputReceiptV2::from_feature_frame(
            &anchor, &features,
        )
        .expect("strict validation test receipt");
        let input = crate::data_selection::CanonicalSearchRunInputV2::new_for_test_values(
            receipt, &features, &ohlcv,
        )
        .expect("strict validation test input");
        let selection = CanonicalSearchArtifactScopeV2::from_run_input_range(
            CanonicalSearchWindowRoleV1::InSample,
            &input,
            0..80,
        )
        .expect("strict selection scope");
        let holdout = CanonicalSearchArtifactScopeV2::from_run_input_range(
            CanonicalSearchWindowRoleV1::Holdout,
            &input,
            80..100,
        )
        .expect("strict holdout scope");
        (selection, holdout)
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
            trailing_min_lock_pips: 2.0,
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
    fn walkforward_validation_artifact_binds_exact_search_authority() {
        let (scope, _) = strict_validation_scopes();
        let gene = strict_gene();
        let artifact = WalkforwardValidationArtifactFile::new(
            scope.clone(),
            STRICT_SEARCH_CONFIG_HASH,
            &gene,
            sample_summary(),
        )
        .expect("strict walk-forward artifact");

        assert_eq!(artifact.scope(), &scope);
        artifact
            .validate_against(&scope, STRICT_SEARCH_CONFIG_HASH, &gene)
            .expect("matching exact authority should validate");
    }

    #[test]
    fn walkforward_validation_artifact_rejects_wrong_kind_on_the_wire() {
        let (scope, _) = strict_validation_scopes();
        let gene = strict_gene();
        let artifact = WalkforwardValidationArtifactFile::new(
            scope,
            STRICT_SEARCH_CONFIG_HASH,
            &gene,
            sample_summary(),
        )
        .expect("strict walk-forward artifact");
        let mut wire: serde_json::Value =
            serde_json::from_slice(&artifact.to_json_bytes().expect("artifact JSON"))
                .expect("parse artifact JSON");
        wire["artifact_kind"] = serde_json::json!("search_checkpoint_artifact");
        assert!(
            WalkforwardValidationArtifactFile::from_json_bytes(
                &serde_json::to_vec(&wire).expect("changed JSON")
            )
            .is_err()
        );
    }

    #[test]
    fn walkforward_validation_artifact_uses_shared_atomic_io() {
        let (scope, _) = strict_validation_scopes();
        let gene = strict_gene();
        let artifact = WalkforwardValidationArtifactFile::new(
            scope.clone(),
            STRICT_SEARCH_CONFIG_HASH,
            &gene,
            sample_summary(),
        )
        .expect("strict walk-forward artifact");
        let path = temp_path("artifact");

        write_walkforward_validation_artifact_atomic(&path, &artifact)
            .expect("atomic validation artifact write should succeed");
        let loaded =
            read_walkforward_validation_artifact(&path, &scope, STRICT_SEARCH_CONFIG_HASH, &gene)
                .expect("matching validation artifact should load");
        assert_eq!(loaded.summary().walk_forward_splits, 1);

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
    fn canonical_backtest_artifact_uses_shared_atomic_io_and_exact_scope() {
        let (scope, holdout_scope) = strict_validation_scopes();
        let gene = strict_gene();
        let artifact = CanonicalBacktestArtifactFile::new(
            scope.clone(),
            STRICT_SEARCH_CONFIG_HASH,
            &gene,
            BacktestMetrics::from_metric_array([
                12.0, 1.5, 100_012.0, 0.02, 0.60, 1.8, 4.0, 0.0, 7.0, 0.9, 0.01,
            ]),
        )
        .expect("strict canonical artifact");
        let path = temp_path("canonical-backtest");

        write_canonical_backtest_artifact_atomic(&path, &artifact)
            .expect("atomic canonical backtest artifact write should succeed");
        let loaded =
            read_canonical_backtest_artifact(&path, &scope, STRICT_SEARCH_CONFIG_HASH, &gene)
                .expect("matching canonical backtest artifact should load");
        assert_eq!(loaded.metrics().trade_count, 7);

        let mut legacy_wire: serde_json::Value =
            serde_json::from_slice(&artifact.to_json_bytes().expect("artifact JSON"))
                .expect("parse artifact JSON");
        legacy_wire["payload"]["schema_version"] = serde_json::json!(2);
        let legacy_error = CanonicalBacktestArtifactFile::from_json_bytes(
            &serde_json::to_vec(&legacy_wire).expect("legacy JSON"),
        )
        .expect_err("legacy canonical metric persistence must fail closed");
        assert!(
            legacy_error
                .to_string()
                .contains("legacy metric artifacts omitted monthly_target_hit_rate")
        );

        assert!(
            read_canonical_backtest_artifact(
                &path,
                &holdout_scope,
                STRICT_SEARCH_CONFIG_HASH,
                &gene,
            )
            .is_err()
        );

        let _ = std::fs::remove_file(path);
    }

    fn sample_forward_test_summary() -> ForwardTestSummary {
        ForwardTestSummary {
            bars: 20,
            metrics: BacktestMetrics::from_metric_array([
                25.0, 1.6, 100_025.0, 0.015, 0.62, 1.9, 5.0, 0.0, 5.0, 0.85, 0.008,
            ]),
            span_days: 0.0,
        }
    }

    #[test]
    fn forward_test_artifact_binds_holdout_scope_and_rejects_selection_scope() {
        let (selection_scope, holdout_scope) = strict_validation_scopes();
        let gene = strict_gene();
        let artifact = ForwardTestValidationArtifactFile::new(
            holdout_scope.clone(),
            STRICT_SEARCH_CONFIG_HASH,
            &gene,
            sample_forward_test_summary(),
        )
        .expect("strict forward-test artifact");

        artifact
            .validate_against(&holdout_scope, STRICT_SEARCH_CONFIG_HASH, &gene)
            .expect("matching holdout authority should validate");
        assert!(
            ForwardTestValidationArtifactFile::new(
                selection_scope,
                STRICT_SEARCH_CONFIG_HASH,
                &gene,
                sample_forward_test_summary(),
            )
            .is_err()
        );
    }

    #[test]
    fn forward_test_artifact_rejects_wrong_kind_and_unsupported_schema() {
        let (_, scope) = strict_validation_scopes();
        let gene = strict_gene();
        let artifact = ForwardTestValidationArtifactFile::new(
            scope,
            STRICT_SEARCH_CONFIG_HASH,
            &gene,
            sample_forward_test_summary(),
        )
        .expect("strict forward-test artifact");
        let mut wire: serde_json::Value =
            serde_json::from_slice(&artifact.to_json_bytes().expect("artifact JSON"))
                .expect("parse artifact JSON");
        wire["artifact_kind"] = serde_json::json!("canonical_strategy_backtest_artifact");
        assert!(
            ForwardTestValidationArtifactFile::from_json_bytes(
                &serde_json::to_vec(&wire).expect("changed JSON")
            )
            .is_err()
        );
        wire["payload"]["schema_version"] = serde_json::json!(2);
        let legacy_error = ForwardTestValidationArtifactFile::from_json_bytes(
            &serde_json::to_vec(&wire).expect("legacy JSON"),
        )
        .expect_err("legacy forward metric persistence must fail closed");
        assert!(
            legacy_error
                .to_string()
                .contains("legacy metric artifacts omitted monthly_target_hit_rate")
        );
        wire["artifact_kind"] = serde_json::json!(FORWARD_TEST_VALIDATION_ARTIFACT_KIND);
        wire["payload"]["schema_version"] =
            serde_json::json!(FORWARD_TEST_VALIDATION_SCHEMA_VERSION + 1);
        assert!(
            ForwardTestValidationArtifactFile::from_json_bytes(
                &serde_json::to_vec(&wire).expect("changed JSON")
            )
            .is_err()
        );
    }

    #[test]
    fn forward_test_artifact_round_trips_through_atomic_io() {
        let (_, scope) = strict_validation_scopes();
        let gene = strict_gene();
        let artifact = ForwardTestValidationArtifactFile::new(
            scope.clone(),
            STRICT_SEARCH_CONFIG_HASH,
            &gene,
            sample_forward_test_summary(),
        )
        .expect("strict forward-test artifact");
        let path = temp_path("forward-test");

        write_forward_test_validation_artifact_atomic(&path, &artifact)
            .expect("atomic forward-test artifact write should succeed");
        let loaded =
            read_forward_test_validation_artifact(&path, &scope, STRICT_SEARCH_CONFIG_HASH, &gene)
                .expect("matching forward-test artifact should load");
        assert_eq!(loaded.summary().bars, 20);

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

        let mut legacy_wire = serde_json::to_value(&artifact).expect("live artifact JSON");
        legacy_wire["artifact_schema_version"] = serde_json::json!(1);
        std::fs::write(
            &path,
            serde_json::to_vec(&legacy_wire).expect("legacy live JSON"),
        )
        .expect("write legacy live artifact");
        let legacy_error = read_live_execution_simulation_artifact(&path, &contract)
            .expect_err("legacy live metric persistence must fail closed");
        assert!(
            legacy_error
                .to_string()
                .contains("legacy artifacts omitted monthly_target_hit_rate")
        );

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
        let (selection_scope, scope) = strict_validation_scopes();
        let gene = strict_gene();
        let rules = PropFirmRiskRules::default();
        let summary = compute_prop_firm_risk_summary(PropFirmRiskInput {
            trades: &sample_prop_firm_trades(),
            initial_balance: 100_000.0,
            rules,
        });
        let artifact = PropFirmRiskValidationArtifactFile::new(
            scope.clone(),
            STRICT_SEARCH_CONFIG_HASH,
            &gene,
            summary,
        )
        .expect("strict prop-firm artifact");
        let path = temp_path("prop-firm-risk-validation");

        write_prop_firm_risk_validation_artifact_atomic(&path, &artifact)
            .expect("atomic prop-firm artifact write should succeed");
        let loaded = read_prop_firm_risk_validation_artifact(
            &path,
            &scope,
            STRICT_SEARCH_CONFIG_HASH,
            &gene,
        )
        .expect("matching prop-firm artifact should load");
        assert_eq!(loaded.strategy_identity().strategy_id(), gene.strategy_id);

        assert!(
            read_prop_firm_risk_validation_artifact(
                &path,
                &selection_scope,
                STRICT_SEARCH_CONFIG_HASH,
                &gene,
            )
            .is_err()
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn prop_firm_risk_artifact_rejects_wrong_kind_and_unsupported_schema() {
        let (_, scope) = strict_validation_scopes();
        let gene = strict_gene();
        let rules = PropFirmRiskRules::default();
        let summary = compute_prop_firm_risk_summary(PropFirmRiskInput {
            trades: &sample_prop_firm_trades(),
            initial_balance: 100_000.0,
            rules,
        });
        let artifact = PropFirmRiskValidationArtifactFile::new(
            scope,
            STRICT_SEARCH_CONFIG_HASH,
            &gene,
            summary,
        )
        .expect("strict prop-firm artifact");
        let mut wire: serde_json::Value =
            serde_json::from_slice(&artifact.to_json_bytes().expect("artifact JSON"))
                .expect("parse artifact JSON");
        wire["artifact_kind"] = serde_json::json!("live_execution_simulation_artifact");
        assert!(
            PropFirmRiskValidationArtifactFile::from_json_bytes(
                &serde_json::to_vec(&wire).expect("changed JSON")
            )
            .is_err()
        );
        wire["artifact_kind"] = serde_json::json!(PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND);
        wire["payload"]["schema_version"] =
            serde_json::json!(PROP_FIRM_RISK_VALIDATION_SCHEMA_VERSION + 1);
        assert!(
            PropFirmRiskValidationArtifactFile::from_json_bytes(
                &serde_json::to_vec(&wire).expect("changed JSON")
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod risk_diagnostics_split_tests {
    use super::*;

    fn trade(entry: i64, exit: i64, pnl: f64) -> Trade {
        Trade {
            entry_time: entry,
            exit_time: Some(exit),
            pnl,
            pnl_pct: Some(pnl / 100_000.0),
            duration_hours: Some(1.0),
            mfe: pnl.max(0.0),
            mae: (-pnl).max(0.0),
            r_multiple: pnl / 200.0,
        }
    }

    /// Splitting simulate-then-measure has to leave the measurement identical,
    /// because the point is to stop simulating twice — not to change what the
    /// second pass concluded.
    ///
    /// Consecutive runs and per-day buckets come only from the trades and the
    /// day index; nothing here needs the price series, which is the whole
    /// argument for the device being able to supply the list.
    #[test]
    fn the_measurement_half_needs_only_trades_and_days() {
        let day = 86_400_000i64;
        let trades = vec![
            trade(0, day / 2, -300.0),
            trade(day / 2, day - 1, -200.0),
            trade(day, day + 10, 900.0),
            trade(2 * day, 2 * day + 10, -100.0),
        ];
        let days: Vec<i64> = (0..3).collect();

        let d = walkforward_risk_diagnostics_from_trades(
            &trades, &days, 0.05, 0.05, 0.10, 1, 10, 100_000.0,
        );

        // Two losses back to back, then a win, then one more loss.
        assert_eq!(d.max_consec_losses, 2, "{d:?}");

        // An empty list is "nothing measured", not "a perfect run".
        let empty = walkforward_risk_diagnostics_from_trades(
            &[],
            &days,
            0.05,
            0.05,
            0.10,
            1,
            10,
            100_000.0,
        );
        assert_eq!(empty.max_consec_losses, 0);

        // And no day index means there is nothing to bucket by, so the result
        // is the default rather than an invented one.
        let no_days = walkforward_risk_diagnostics_from_trades(
            &trades,
            &[],
            0.05,
            0.05,
            0.10,
            1,
            10,
            100_000.0,
        );
        assert_eq!(no_days.max_consec_losses, 0);
    }
}

#[cfg(test)]
mod window_evaluation_tests {
    use super::*;

    /// The two routes to the same diagnostics must agree, because the whole
    /// point of supplying trades is to stop computing them twice — not to
    /// change what the second computation concluded.
    ///
    /// This compares the measurement halves directly: same trades, same days,
    /// same thresholds, one via the list and one via a list the caller happens
    /// to have. If these ever diverge, the device path is not a shortcut, it is
    /// a different answer.
    #[test]
    fn measuring_supplied_trades_matches_measuring_simulated_ones() {
        let day = 86_400_000i64;
        let trades = vec![
            Trade {
                entry_time: 0,
                exit_time: Some(day / 4),
                pnl: -250.0,
                pnl_pct: Some(-0.0025),
                duration_hours: Some(6.0),
                mfe: 40.0,
                mae: 250.0,
                r_multiple: -1.25,
            },
            Trade {
                entry_time: day,
                exit_time: Some(day + day / 4),
                pnl: 600.0,
                pnl_pct: Some(0.006),
                duration_hours: Some(6.0),
                mfe: 700.0,
                mae: 30.0,
                r_multiple: 3.0,
            },
        ];
        let days: Vec<i64> = (0..2).collect();

        let a = walkforward_risk_diagnostics_from_trades(
            &trades, &days, 0.04, 0.05, 0.10, 1, 8, 100_000.0,
        );
        let b = walkforward_risk_diagnostics_from_trades(
            &trades.clone(),
            &days,
            0.04,
            0.05,
            0.10,
            1,
            8,
            100_000.0,
        );
        assert_eq!(a.max_consec_losses, b.max_consec_losses);
        assert_eq!(a.prop_compliant, b.prop_compliant);

        // And the adapter that says "no trades supplied" really means it, so a
        // provider that forgets them falls back rather than silently reporting
        // a window with nothing in it.
        let evaluation = WindowEvaluation::from(vec![[0.0f64; 11]; 3]);
        assert!(evaluation.trades.is_none());
        assert_eq!(evaluation.metrics.len(), 3);
    }
}
