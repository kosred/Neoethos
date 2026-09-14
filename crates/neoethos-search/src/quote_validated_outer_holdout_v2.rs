//! Pre-acquisition signal identity: policies and inputs, never future trade counts.
//!
//! V1's signal hash remains unchanged. V3 borrows the already prepared final
//! signal/confidence pairs and locks their bar/exit/account-risk inputs before any quote acquisition. Actual
//! decision witnesses come from the sealed kernel, not caller-supplied risk rows.

use std::collections::HashMap;
use std::io::{self, Write};

use neoethos_broker_truth::{
    CanonicalBarSignalResearchDecisionV1, CausalQuoteToAccountConversionV1,
    ClosedCanonicalBarTimeExitV1, ClosedCanonicalBarTrailingThresholdV1, ExecutionSymbolContractV1,
    QuoteValidatedResearchReplayBindingV1, QuoteValidatedResearchReplayPlanV1,
    QuoteValidatedResearchReplayPolicyV1, ResearchPositionDirectionV1,
    SealedHistoricalBidAskQuoteReplayEvidenceV1, preview_sealed_quote_validated_research_entry_v1,
};
use neoethos_core::domain::trailing::TrailingPolicy;
use neoethos_data::{CanonicalDatasetScope, CanonicalTimeframe, Ohlcv};

use super::*;
use crate::{CanonicalSearchWindowRoleV1, EvaluationConfig, Gene};

/// Exact run-scoped exit geometry, copied from the already resolved evaluation.
/// No settings reload, default policy, risk sizing or monetary authority here.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CanonicalSignalExitPolicyV2 {
    pub pip_size: f64,
    pub max_hold_bars: usize,
    pub trailing_enabled: bool,
    pub trailing_stop_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub trailing_min_lock_pips: f64,
}

impl CanonicalSignalExitPolicyV2 {
    pub fn from_evaluation(evaluation: &EvaluationConfig) -> Self {
        Self {
            pip_size: evaluation.pip_value,
            max_hold_bars: evaluation.max_hold_bars,
            trailing_enabled: evaluation.trailing_enabled,
            trailing_stop_multiplier: evaluation.trailing_atr_multiplier,
            trailing_be_trigger_r: evaluation.trailing_be_trigger_r,
            trailing_min_lock_pips: evaluation.trailing_min_lock_pips,
        }
    }

    fn validate(&self) -> Result<(), QuoteValidatedOuterHoldoutErrorV1> {
        if !self.pip_size.is_finite()
            || self.pip_size <= 0.0
            || !self.trailing_stop_multiplier.is_finite()
            || !self.trailing_be_trigger_r.is_finite()
            || !self.trailing_min_lock_pips.is_finite()
            || (self.trailing_enabled
                && (self.trailing_stop_multiplier <= 0.0
                    || self.trailing_be_trigger_r <= 0.0
                    || self.trailing_min_lock_pips < 0.0))
        {
            return Err(binding_error(
                "invalid or unbounded locked signal exit policy",
            ));
        }
        Ok(())
    }
}

/// Immutable shared-account sizing policy copied from the same Search run.
/// Entry risk uses current marked equity; carry and conversion fees settle at
/// closed-trade ledger time. The 100-lot ceiling is the existing Search ceiling,
/// not a broker volume rule. Quote replay additionally applies explicit captured
/// broker lot constraints and must never round a below-minimum order upward.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CanonicalSignalAccountRiskPolicyV3 {
    initial_balance: AccountMoneyV1,
    risk_per_trade_min: f64,
    risk_per_trade_max: f64,
    high_quality_confidence: f64,
}

impl CanonicalSignalAccountRiskPolicyV3 {
    pub fn from_evaluation(
        evaluation: &EvaluationConfig,
    ) -> Result<Self, QuoteValidatedOuterHoldoutErrorV1> {
        Self::new(
            AccountMoneyV1::new(&evaluation.account_currency, evaluation.initial_equity)
                .map_err(|error| binding_error(error.to_string()))?,
            evaluation.risk_per_trade_min,
            evaluation.risk_per_trade_max,
            evaluation.high_quality_confidence,
        )
    }

    pub fn new(
        initial_balance: AccountMoneyV1,
        risk_per_trade_min: f64,
        risk_per_trade_max: f64,
        high_quality_confidence: f64,
    ) -> Result<Self, QuoteValidatedOuterHoldoutErrorV1> {
        let policy = Self {
            initial_balance,
            risk_per_trade_min,
            risk_per_trade_max,
            high_quality_confidence,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<(), QuoteValidatedOuterHoldoutErrorV1> {
        if !self.initial_balance.amount().is_finite()
            || self.initial_balance.amount() <= 0.0
            || !self.risk_per_trade_min.is_finite()
            || self.risk_per_trade_min < 0.0
            || !self.risk_per_trade_max.is_finite()
            || self.risk_per_trade_max < self.risk_per_trade_min
            || self.risk_per_trade_max > 1.0
            || !self.high_quality_confidence.is_finite()
            || !(0.0..=1.0).contains(&self.high_quality_confidence)
            || self.high_quality_confidence == 0.0
        {
            return Err(binding_error(
                "invalid locked initial account balance or confidence-risk band",
            ));
        }
        AccountMoneyV1::new(
            self.initial_balance.currency(),
            self.initial_balance.amount(),
        )
        .map_err(|error| binding_error(error.to_string()))?;
        Ok(())
    }

    pub fn initial_balance(&self) -> &AccountMoneyV1 {
        &self.initial_balance
    }

    pub fn risk_fraction(&self, confidence: f64) -> Result<f64, QuoteValidatedOuterHoldoutErrorV1> {
        self.validate()?;
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(binding_error(
                "entry confidence must be finite and in [0, 1]",
            ));
        }
        Ok(self.risk_per_trade_min
            + (self.risk_per_trade_max - self.risk_per_trade_min)
                * (confidence / self.high_quality_confidence).min(1.0))
    }

    /// Canonical Search risk sizing before broker step/min/max quantization.
    /// Confidence zero is legitimate and uses minimum risk, not confidence 1.
    pub fn entry_lots(
        &self,
        confidence: f64,
        equity: &AccountMoneyV1,
        stop_pips: f64,
        pip_value_account_per_lot: f64,
    ) -> Result<f64, QuoteValidatedOuterHoldoutErrorV1> {
        let risk = self.risk_fraction(confidence)?;
        let denominator = stop_pips * pip_value_account_per_lot;
        if equity.currency() != self.initial_balance.currency()
            || !equity.amount().is_finite()
            || equity.amount() <= 0.0
            || !stop_pips.is_finite()
            || stop_pips <= 0.0
            || !pip_value_account_per_lot.is_finite()
            || pip_value_account_per_lot <= 0.0
            || !denominator.is_finite()
            || denominator <= 0.0
        {
            return Err(binding_error(
                "entry sizing requires positive finite equity, actual stop and account pip value",
            ));
        }
        let lots = equity.amount() * risk / denominator;
        if !lots.is_finite() || lots < 0.0 {
            return Err(binding_error(
                "entry lot arithmetic is nonfinite or negative",
            ));
        }
        Ok(lots.min(100.0))
    }
}

fn binding_error(detail: impl Into<String>) -> QuoteValidatedOuterHoldoutErrorV1 {
    outer_error(
        QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
        detail,
    )
}

// Stream the potentially large signal/bar arrays into the digest. Do not make
// a second genes × rows allocation just to encode their identity as JSON.
struct DigestWriter(Sha256);
impl Write for DigestWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn streamed_sha256<T: Serialize>(
    domain: &str,
    value: &T,
) -> Result<String, QuoteValidatedOuterHoldoutErrorV1> {
    let mut writer = DigestWriter(Sha256::new());
    writer.0.update(domain.as_bytes());
    writer.0.update([0]);
    serde_json::to_writer(&mut writer, value).map_err(|error| {
        outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::ArtifactEncodingFailed,
            format!("cannot encode {domain}: {error}"),
        )
    })?;
    Ok(format!("{:x}", writer.0.finalize()))
}

#[derive(Serialize)]
struct SignalPlanPayloadV3<'a> {
    schema_version: u16,
    canonical_search_input_receipt_sha256: &'a str,
    holdout_scope_identity_sha256: &'a str,
    portfolio_identity_sha256: &'a str,
    ordered_gene_identity_sha256: &'a str,
    search_config_hash: &'a str,
    ordered_signals: &'a [Vec<i8>],
    ordered_confidences: &'a [Vec<f64>],
    ordered_size_multipliers: &'a [Vec<f64>],
    bar_open_timestamps: &'a [i64],
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: Option<&'a [f64]>,
    exit_policy: CanonicalSignalExitPolicyV2,
    account_risk_policy: &'a CanonicalSignalAccountRiskPolicyV3,
    adaptive_stops_policy: &'a crate::stop_target::ResolvedAdaptiveStopsPolicyV1,
}

/// Non-deserializable, immutable borrow of the post-lock input. The identity
/// can be used by acquisition before any fills, non-entries or risk rows exist.
#[derive(Debug)]
pub struct LockedCanonicalSignalPlanV3<'a> {
    portfolio: &'a [Gene],
    ordered_signals: &'a [Vec<i8>],
    ordered_confidences: &'a [Vec<f64>],
    ordered_size_multipliers: &'a [Vec<f64>],
    bars: &'a Ohlcv,
    timestamps: &'a [i64],
    holdout_scope: &'a CanonicalSearchArtifactScopeV2,
    search_config_hash: &'a str,
    exit_policy: CanonicalSignalExitPolicyV2,
    account_risk_policy: CanonicalSignalAccountRiskPolicyV3,
    adaptive_stops_policy: crate::stop_target::ResolvedAdaptiveStopsPolicyV1,
    adaptive_base_pips: Option<Vec<f64>>,
    timeframe: CanonicalTimeframe,
    duration_ms: i64,
    account_id: i64,
    symbol_id: i64,
    symbol_name: String,
    window: EvidenceWindowV1,
    portfolio_identity_sha256: String,
    scope_identity_sha256: String,
    identity_sha256: String,
}

impl<'a> LockedCanonicalSignalPlanV3<'a> {
    pub fn new(
        portfolio: &'a [Gene],
        portfolio_identity_sha256: &str,
        ordered_signals: &'a [Vec<i8>],
        ordered_confidences: &'a [Vec<f64>],
        ordered_size_multipliers: &'a [Vec<f64>],
        bars: &'a Ohlcv,
        holdout_scope: &'a CanonicalSearchArtifactScopeV2,
        search_config_hash: &'a str,
        exit_policy: CanonicalSignalExitPolicyV2,
        account_risk_policy: CanonicalSignalAccountRiskPolicyV3,
        adaptive_stops_policy: &crate::stop_target::ResolvedAdaptiveStopsPolicyV1,
    ) -> Result<Self, QuoteValidatedOuterHoldoutErrorV1> {
        holdout_scope
            .validate()
            .map_err(|error| binding_error(error.to_string()))?;
        exit_policy.validate()?;
        account_risk_policy.validate()?;
        adaptive_stops_policy
            .validate()
            .map_err(|error| binding_error(error.to_string()))?;
        validate_sha256("full locked portfolio identity", portfolio_identity_sha256)?;
        let identity = holdout_scope
            .receipt()
            .validate()
            .map_err(|error| binding_error(error.to_string()))?;
        let CanonicalDatasetScope::CTrader {
            account_id,
            symbol_id,
            ..
        } = identity.scope()
        else {
            return Err(binding_error(
                "quote signal plan requires the exact cTrader dataset identity",
            ));
        };
        let duration_ms = identity.timeframe().fixed_duration_ms().ok_or_else(|| {
            binding_error("calendar quote replay needs an explicit broker bar-close calendar")
        })?;
        let timestamps = bars
            .timestamp
            .as_deref()
            .ok_or_else(|| binding_error("locked signal bars have no timestamps"))?;
        let rows = bars.close.len();
        let scope_window = holdout_scope.evaluated_window();
        if scope_window.role() != CanonicalSearchWindowRoleV1::Holdout
            || search_config_hash.trim().is_empty()
            || portfolio.is_empty()
            || portfolio.len() != ordered_signals.len()
            || portfolio.len() != ordered_confidences.len()
            || portfolio.len() != ordered_size_multipliers.len()
            || rows < 2
            || [
                timestamps.len(),
                bars.open.len(),
                bars.high.len(),
                bars.low.len(),
            ]
            .iter()
            .any(|length| *length != rows)
            || scope_window.row_end() - scope_window.row_start() != rows as u64
            || timestamps.first().copied() != Some(scope_window.timestamp_start_ms())
            || timestamps.last().copied() != Some(scope_window.timestamp_end_ms())
            || ordered_signals.iter().any(|signals| {
                signals.len() != rows || signals.iter().any(|signal| !(-1..=1).contains(signal))
            })
            || ordered_confidences.iter().any(|confidences| {
                confidences.len() != rows
                    || confidences.iter().any(|confidence| {
                        !confidence.is_finite() || !(0.0..=1.0).contains(confidence)
                    })
            })
            || ordered_size_multipliers.iter().any(|multipliers| {
                multipliers.len() != rows
                    || multipliers
                        .iter()
                        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            })
            || bars.volume.as_ref().is_some_and(|volume| {
                volume.len() != rows
                    || volume
                        .iter()
                        .any(|value| !value.is_finite() || *value < 0.0)
            })
        {
            return Err(binding_error(
                "locked genes/signals/bars do not match the exact final holdout",
            ));
        }
        for gene in portfolio {
            if !gene.stop_vol_mult.is_finite()
                || gene.stop_vol_mult < 0.0
                || !gene.sl_pips.is_finite()
                || gene.sl_pips <= 0.0
                || !gene.tp_pips.is_finite()
                || gene.tp_pips <= 0.0
            {
                return Err(binding_error(
                    "locked producer requires finite nonnegative adaptive multiplier and positive gene risk/target",
                ));
            }
        }
        for (index, timestamp) in timestamps.iter().copied().enumerate() {
            if timestamp < 0
                || timestamp % duration_ms != 0
                || (index > 0
                    && timestamps[index - 1]
                        .checked_add(duration_ms)
                        .is_none_or(|previous_end| previous_end > timestamp))
                || [
                    bars.open[index],
                    bars.high[index],
                    bars.low[index],
                    bars.close[index],
                ]
                .iter()
                .any(|price| !price.is_finite() || *price <= 0.0)
                || bars.low[index] > bars.open[index]
                || bars.low[index] > bars.close[index]
                || bars.high[index] < bars.open[index]
                || bars.high[index] < bars.close[index]
            {
                return Err(binding_error(format!("invalid locked bar at row {index}")));
            }
        }
        let end = timestamps[rows - 1]
            .checked_add(duration_ms)
            .ok_or_else(|| binding_error("locked final bar close overflow"))?;
        let window = EvidenceWindowV1::new(timestamps[0], end)
            .map_err(|error| binding_error(error.to_string()))?;
        let ordered_gene_identity_sha256 =
            canonical_locked_portfolio_identity_sha256_v1(&portfolio)?;
        // Match Search: an evolved positive multiplier activates the recipe.
        // The archived enabled flag governs gene generation, not reinterpretation
        // of an already evolved gene. A too-short whole slice retains the same
        // explicit fixed-pip fallback; unavailable warm-up CELLS never enter.
        let adaptive_base_pips = if portfolio.iter().any(|gene| gene.stop_vol_mult > 0.0) {
            match crate::stop_target::adaptive_base_pips_series_with_settings(
                &bars.high,
                &bars.low,
                &bars.close,
                exit_policy.pip_size,
                adaptive_stops_policy.settings(),
            ) {
                Ok(base) => Some(base),
                Err(crate::stop_target::StopDistanceError::TooShort { .. }) => None,
                Err(error) => {
                    return Err(binding_error(format!(
                        "locked adaptive stop recipe failed: {error}"
                    )));
                }
            }
        } else {
            None
        };
        let portfolio_identity_sha256 = portfolio_identity_sha256.to_owned();
        let scope_identity_sha256 = holdout_scope
            .identity_sha256()
            .map_err(|error| binding_error(error.to_string()))?;
        let identity_sha256 = streamed_sha256(
            "neoethos.canonical-bar-signal-plan.v3",
            &SignalPlanPayloadV3 {
                schema_version: 3,
                canonical_search_input_receipt_sha256: holdout_scope.receipt_sha256(),
                holdout_scope_identity_sha256: &scope_identity_sha256,
                portfolio_identity_sha256: &portfolio_identity_sha256,
                ordered_gene_identity_sha256: &ordered_gene_identity_sha256,
                search_config_hash,
                ordered_signals,
                ordered_confidences,
                ordered_size_multipliers,
                bar_open_timestamps: timestamps,
                open: &bars.open,
                high: &bars.high,
                low: &bars.low,
                close: &bars.close,
                volume: bars.volume.as_deref(),
                exit_policy,
                account_risk_policy: &account_risk_policy,
                adaptive_stops_policy,
            },
        )?;
        Ok(Self {
            portfolio,
            ordered_signals,
            ordered_confidences,
            ordered_size_multipliers,
            bars,
            timestamps,
            holdout_scope,
            search_config_hash,
            exit_policy,
            account_risk_policy,
            adaptive_stops_policy: adaptive_stops_policy.clone(),
            adaptive_base_pips,
            timeframe: identity.timeframe(),
            duration_ms,
            account_id: *account_id,
            symbol_id: *symbol_id,
            symbol_name: identity.symbol_name().to_owned(),
            window,
            portfolio_identity_sha256,
            scope_identity_sha256,
            identity_sha256,
        })
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }
    pub fn portfolio_identity_sha256(&self) -> &str {
        &self.portfolio_identity_sha256
    }
    pub fn scope_identity_sha256(&self) -> &str {
        &self.scope_identity_sha256
    }
    pub fn search_config_hash(&self) -> &str {
        self.search_config_hash
    }
    pub fn holdout_scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        self.holdout_scope
    }
    pub fn portfolio(&self) -> &[Gene] {
        self.portfolio
    }
    pub fn ordered_signals(&self) -> &[Vec<i8>] {
        self.ordered_signals
    }
    pub fn ordered_confidences(&self) -> &[Vec<f64>] {
        self.ordered_confidences
    }
    /// Zero risk is a non-entry, not permission to raise volume to a broker
    /// minimum. Both quote production and complete-lane validation consult this
    /// same immutable predicate before occupying a lane.
    pub fn entry_eligible(&self, lane: usize, row: usize) -> bool {
        self.ordered_signals[lane][row] != 0
            && self.entry_stop_target_pips(lane, row).is_some()
            && self.ordered_size_multipliers[lane][row] > 0.0
            && self
                .account_risk_policy
                .risk_fraction(self.ordered_confidences[lane][row])
                .is_ok_and(|risk| risk > 0.0)
    }
    /// The actual decision-bar risk/target used by both quote brackets and sizing.
    /// Warm-up cells in an enabled adaptive series return None, never fixed pips.
    pub fn entry_stop_target_pips(&self, lane: usize, row: usize) -> Option<(f64, f64)> {
        let gene = self.portfolio.get(lane)?;
        self.timestamps.get(row)?;
        crate::stop_target::resolve_entry_stop_target_pips(
            gene.sl_pips,
            gene.tp_pips,
            gene.stop_vol_mult,
            self.adaptive_base_pips
                .as_ref()
                .and_then(|base| base.get(row).copied()),
            self.adaptive_base_pips.is_some(),
            self.adaptive_stops_policy.reward_risk_fallback(),
        )
    }
    pub fn ordered_size_multipliers(&self) -> &[Vec<f64>] {
        self.ordered_size_multipliers
    }
    pub fn account_risk_policy(&self) -> &CanonicalSignalAccountRiskPolicyV3 {
        &self.account_risk_policy
    }
    pub fn bars(&self) -> &Ohlcv {
        self.bars
    }
    pub fn timestamps(&self) -> &[i64] {
        self.timestamps
    }
    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }
    pub const fn timeframe(&self) -> CanonicalTimeframe {
        self.timeframe
    }
    pub const fn exit_policy(&self) -> CanonicalSignalExitPolicyV2 {
        self.exit_policy
    }
    pub const fn locked_evaluation_window(&self) -> EvidenceWindowV1 {
        self.window
    }

    pub fn validate_replay_binding(
        &self,
        binding: &QuoteValidatedResearchReplayBindingV1,
        policy: &QuoteValidatedResearchReplayPolicyV1,
    ) -> Result<(), QuoteValidatedOuterHoldoutErrorV1> {
        binding
            .validate_replay_policy_v1(policy)
            .map_err(|error| binding_error(error.to_string()))?;
        if binding.canonical_signal_plan_sha256() != self.identity_sha256
            || binding.canonical_search_input_receipt_sha256()
                != self.holdout_scope.receipt_sha256()
            || binding.account_id() != self.account_id
            || binding.symbol_id() != self.symbol_id
            || binding.symbol_name() != self.symbol_name
            || binding.replay_scope().locked_evaluation_window() != self.window
            || policy.pip_size() != self.exit_policy.pip_size
        {
            return Err(binding_error(
                "captured quotes differ from the prelocked signal/risk plan",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedDecisionProvenanceV3 {
    pub portfolio_index: usize,
    pub decision_bar_index: usize,
    pub canonical_source_row: u64,
    pub signal_bar_open_unix_ms: i64,
    pub decision_at_unix_ms: i64,
    pub risk_pips: f64,
    pub confidence: f64,
    pub size_multiplier: f64,
    pub entry_sizing: Option<QuoteEntrySizingEvidenceV3>,
    pub executed_plan_sha256: String,
    pub quote_ledger_sha256: String,
}

#[derive(Debug)]
pub struct LockedPortfolioOuterHoldoutReplaySetV3 {
    replay_set: LockedPortfolioOuterHoldoutReplaySetV1,
    provenance: Vec<QuoteValidatedDecisionProvenanceV3>,
    acquisition_link_manifest_sha256: String,
    replay_policy_sha256: String,
    exit_policy: CanonicalSignalExitPolicyV2,
}

/// Explicit captured/operator-reviewed quantity grid; never infer FX lot sizes.
/// Evidence remains research-only, as do the separate execution-cost inputs.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteReplayLotConstraintsV3 {
    min_lots: f64,
    max_lots: f64,
    lot_step: f64,
    source_artifact_sha256: String,
}

impl QuoteReplayLotConstraintsV3 {
    pub fn new(
        min_lots: f64,
        max_lots: f64,
        lot_step: f64,
        source_artifact_sha256: String,
    ) -> Result<Self, QuoteValidatedOuterHoldoutErrorV1> {
        if [min_lots, max_lots, lot_step]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || max_lots < min_lots
            || lot_step > max_lots
        {
            return Err(binding_error("invalid explicit replay lot constraints"));
        }
        validate_sha256("lot constraints source", &source_artifact_sha256)?;
        Ok(Self {
            min_lots,
            max_lots,
            lot_step,
            source_artifact_sha256,
        })
    }

    fn normalize_down(&self, requested: f64) -> Result<f64, QuoteValidatedOuterHoldoutErrorV1> {
        let capped = requested.min(self.max_lots);
        let lots = (capped / self.lot_step).floor() * self.lot_step;
        if !lots.is_finite() || lots < self.min_lots || lots > capped {
            return Err(binding_error(
                "risk-sized entry cannot meet the explicit lot grid without increasing risk",
            ));
        }
        Ok(lots)
    }
}

/// Entry-time financial inputs. Exit-time conversion is a DIFFERENT input to
/// the existing economics builder, and cannot be reused here retroactively.
#[derive(Clone, Debug)]
pub struct QuoteEntryFinancialInputsV3 {
    pub symbol_contract: ExecutionSymbolContractV1,
    pub entry_conversion: CausalQuoteToAccountConversionV1,
    pub lot_constraints: QuoteReplayLotConstraintsV3,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteEntrySizingEvidenceV3 {
    pub balance_before_entry: AccountMoneyV1,
    pub unrealized_before_entry: AccountMoneyV1,
    pub equity_before_entry: AccountMoneyV1,
    pub pip_value_account_per_lot: f64,
    pub risk_fraction: f64,
    pub requested_lots: f64,
    pub filled_lots: f64,
    pub entry_conversion: CausalQuoteToAccountConversionV1,
    pub lot_constraints: QuoteReplayLotConstraintsV3,
}

/// Small immutable view lets the same completeness/origin check be exercised
/// without manufacturing a broker seal in unit tests. Only `new` below accepts
/// production outcomes, and it requires genuine sealed kernel witnesses.
struct OutcomeView<'a> {
    decision: &'a CanonicalBarSignalResearchDecisionV1,
    entry: Option<(i64, f64)>,
    completed_at: Option<i64>,
    time_exit: Option<&'a ClosedCanonicalBarTimeExitV1>,
    plan_sha256: &'a str,
    ledger_sha256: &'a str,
}

fn validate_lane<'a>(
    locked: &LockedCanonicalSignalPlanV3<'_>,
    lane: usize,
    mut outcomes: impl Iterator<Item = OutcomeView<'a>>,
) -> Result<Vec<QuoteValidatedDecisionProvenanceV3>, QuoteValidatedOuterHoldoutErrorV1> {
    let mut occupied_until = None;
    let mut provenance = Vec::new();
    for (index, &signal) in locked.ordered_signals[lane].iter().enumerate() {
        let at = locked.timestamps[index] + locked.duration_ms;
        if at >= locked.window.to_unix_ms_exclusive() {
            break;
        }
        if !locked.entry_eligible(lane, index) || occupied_until.is_some_and(|until| at <= until) {
            continue;
        }
        let outcome = outcomes.next().ok_or_else(|| {
            binding_error(format!(
                "missing eligible decision in lane {lane} at row {index}"
            ))
        })?;
        let direction = if signal == 1 {
            ResearchPositionDirectionV1::Long
        } else {
            ResearchPositionDirectionV1::Short
        };
        let decision = outcome.decision;
        if decision.signal_bar_open_unix_ms() != locked.timestamps[index]
            || decision.decision_at_unix_ms() != at
            || decision.direction() != direction
        {
            return Err(binding_error(format!(
                "wrong signal/decision origin in lane {lane} at row {index}"
            )));
        }
        let anchor_price = outcome
            .entry
            .map_or(locked.bars.close[index], |(_, price)| price);
        let sign = f64::from(signal);
        let (stop_pips, target_pips) = locked
            .entry_stop_target_pips(lane, index)
            .ok_or_else(|| binding_error("eligible decision has no resolved stop/target"))?;
        if decision.stop_price() != anchor_price - sign * stop_pips * locked.exit_policy.pip_size
            || decision.target_price()
                != anchor_price + sign * target_pips * locked.exit_policy.pip_size
        {
            return Err(binding_error(
                "executed brackets differ from the prelocked gene risk/target",
            ));
        }
        let mut expected_timer = None;
        if let Some((entry_at, _)) = outcome.entry {
            let entry_index = locked
                .timestamps
                .partition_point(|timestamp| *timestamp <= entry_at)
                .checked_sub(1)
                .ok_or_else(|| binding_error("entry precedes locked bars"))?;
            if entry_at < at || entry_at >= locked.timestamps[entry_index] + locked.duration_ms {
                return Err(binding_error("entry has no causal canonical holding clock"));
            }
            let last = if locked.exit_policy.max_hold_bars == 0 {
                None
            } else {
                Some(
                    entry_index
                        .checked_add(locked.exit_policy.max_hold_bars - 1)
                        .ok_or_else(|| binding_error("holding limit overflow"))?,
                )
            };
            if let Some(&timestamp) = last.and_then(|last| locked.timestamps.get(last)) {
                expected_timer = Some(
                    ClosedCanonicalBarTimeExitV1::new(timestamp, timestamp + locked.duration_ms)
                        .map_err(|error| binding_error(error.to_string()))?,
                );
            }
        } else if outcome.completed_at.is_none_or(|deadline| deadline < at) {
            return Err(binding_error(
                "non-entry has no valid pending-decision deadline",
            ));
        }
        if outcome.time_exit != expected_timer.as_ref() {
            return Err(binding_error(
                "executed holding limit differs from the prelocked policy",
            ));
        }
        validate_sha256("executed decision plan", outcome.plan_sha256)?;
        provenance.push(QuoteValidatedDecisionProvenanceV3 {
            portfolio_index: lane,
            decision_bar_index: index,
            canonical_source_row: locked.holdout_scope.evaluated_window().row_start()
                + index as u64,
            signal_bar_open_unix_ms: locked.timestamps[index],
            decision_at_unix_ms: at,
            risk_pips: stop_pips,
            confidence: locked.ordered_confidences[lane][index],
            size_multiplier: locked.ordered_size_multipliers[lane][index],
            entry_sizing: None,
            executed_plan_sha256: outcome.plan_sha256.to_owned(),
            quote_ledger_sha256: outcome.ledger_sha256.to_owned(),
        });
        occupied_until = outcome.completed_at;
        if outcome.entry.is_some() && occupied_until.is_none() {
            break;
        }
    }
    if outcomes.next().is_some() {
        return Err(binding_error(
            "extra replay outcome outside the complete locked signal lane",
        ));
    }
    Ok(provenance)
}

/// Reconstruct the full exit schedule from pinned bars/settings and the exact
/// reviewed entry preview. Equal fills are insufficient: an omitted, shifted
/// or substituted trailing plan must not be accepted as the locked strategy.
/// The price ratchet is shared with Trader's Position, not a second formula.
fn validate_executed_exit_policy(
    locked: &LockedCanonicalSignalPlanV3<'_>,
    binding: &QuoteValidatedResearchReplayBindingV1,
    policy: &QuoteValidatedResearchReplayPolicyV1,
    snapshot: &SealedHistoricalBidAskQuoteReplayEvidenceV1,
    ledger: &SealedHistoricalQuoteValidatedResearchLedgerV1,
) -> Result<(), QuoteValidatedOuterHoldoutErrorV1> {
    let decision = ledger.executed_decision();
    let mut expected = QuoteValidatedResearchReplayPlanV1::new(
        binding.clone(),
        policy.clone(),
        vec![decision.clone()],
        Vec::new(),
    )
    .map_err(|error| binding_error(error.to_string()))?;
    let entry = preview_sealed_quote_validated_research_entry_v1(&expected, snapshot)
        .map_err(|error| binding_error(error.to_string()))?;
    if let Some(entry) = entry {
        let position = ledger.positions().first().ok_or_else(|| {
            binding_error("sealed non-entry disagrees with the exact reviewed entry selection")
        })?;
        if position.entry_reference().timestamp_unix_ms() != entry.timestamp_unix_ms()
            || position.modeled_entry_price().to_bits() != entry.modeled_entry_price().to_bits()
        {
            return Err(binding_error(
                "sealed position disagrees with the reviewed entry selection",
            ));
        }
        let entry_index = locked
            .timestamps
            .partition_point(|timestamp| *timestamp <= entry.timestamp_unix_ms())
            .checked_sub(1)
            .ok_or_else(|| binding_error("entry precedes the locked exit bars"))?;
        let last_hold_index = if locked.exit_policy.max_hold_bars == 0 {
            locked.timestamps.len()
        } else {
            entry_index
                .checked_add(locked.exit_policy.max_hold_bars - 1)
                .ok_or_else(|| binding_error("holding limit overflow"))?
        };
        let exit = locked.exit_policy;
        let mut thresholds = Vec::new();
        if exit.trailing_enabled {
            let trailing = TrailingPolicy::new(
                exit.trailing_be_trigger_r,
                exit.trailing_stop_multiplier,
                exit.trailing_min_lock_pips,
                exit.pip_size,
            )
            .ok_or_else(|| binding_error("invalid locked trailing geometry"))?;
            let direction = match decision.direction() {
                ResearchPositionDirectionV1::Long => 1,
                ResearchPositionDirectionV1::Short => -1,
            };
            let mut current = None;
            for index in entry_index..last_hold_index.min(locked.timestamps.len()) {
                let open = locked.timestamps[index];
                let end = open + locked.duration_ms;
                let (high, low) = if index == entry_index {
                    // Never reuse a pre-entry candle extreme for the entry bar.
                    entry
                        .bid_extrema_before(end)
                        .map_err(|error| binding_error(error.to_string()))?
                } else {
                    (locked.bars.high[index], locked.bars.low[index])
                };
                if let Some(price) = trailing.next_stop_price(
                    entry.modeled_entry_price(),
                    decision.stop_price(),
                    direction,
                    high,
                    low,
                    current,
                ) {
                    current = Some(price);
                    thresholds.push(
                        ClosedCanonicalBarTrailingThresholdV1::new(
                            open,
                            end,
                            decision.direction(),
                            price,
                        )
                        .map_err(|error| binding_error(error.to_string()))?,
                    );
                }
            }
        }
        expected = QuoteValidatedResearchReplayPlanV1::new(
            binding.clone(),
            policy.clone(),
            vec![decision.clone()],
            thresholds,
        )
        .map_err(|error| binding_error(error.to_string()))?;
        if let Some(&open) = locked.timestamps.get(last_hold_index) {
            let timer = ClosedCanonicalBarTimeExitV1::new(open, open + locked.duration_ms)
                .map_err(|error| binding_error(error.to_string()))?;
            expected = expected
                .with_time_exit(timer)
                .map_err(|error| binding_error(error.to_string()))?;
        }
    } else if !ledger.positions().is_empty() {
        return Err(binding_error(
            "sealed position has no eligible reviewed entry",
        ));
    }
    if expected
        .identity_sha256()
        .map_err(|error| binding_error(error.to_string()))?
        != ledger.executed_plan_sha256()
    {
        return Err(binding_error(
            "executed trailing schedule or full exit plan differs from the pinned policy",
        ));
    }
    Ok(())
}

/// Size the complete same-symbol portfolio in entry-time order. Only exits
/// STRICTLY before an entry contribute settled cash. At equal timestamps the
/// still-open position is marked from the entry's sealed synchronized book;
/// its eventual net profit/exit fees are never pulled backwards into sizing.
/// Simultaneous entries use stable (portfolio,row) ordering for entry fees.
/// This deterministic research settlement rule is part of the V3 domain.
fn size_locked_account_entries<E, F>(
    locked: &LockedCanonicalSignalPlanV3<'_>,
    ordered: &mut [(
        QuoteValidatedDecisionProvenanceV3,
        SealedHistoricalQuoteValidatedResearchLedgerV1,
    )],
    mut entry_inputs: E,
    mut execution_economics: F,
) -> Result<Vec<QuoteValidatedExecutionEconomicsLedgerV1>, QuoteValidatedOuterHoldoutErrorV1>
where
    E: FnMut(
        &SealedHistoricalQuoteValidatedResearchLedgerV1,
        &QuoteValidatedDecisionProvenanceV3,
    ) -> anyhow::Result<QuoteEntryFinancialInputsV3>,
    F: FnMut(
        &SealedHistoricalQuoteValidatedResearchLedgerV1,
        &QuoteEntrySizingEvidenceV3,
    ) -> anyhow::Result<QuoteValidatedExecutionEconomicsLedgerV1>,
{
    let mut entries = Vec::new();
    for (index, (origin, ledger)) in ordered.iter().enumerate() {
        if let Some(position) = ledger.positions().first() {
            if position.exit_reference().is_none() {
                return Err(binding_error(
                    "locked replay retains an open final position; closed execution economics are incomplete",
                ));
            }
            entries.push((
                position.entry_reference().timestamp_unix_ms(),
                origin.portfolio_index,
                origin.decision_bar_index,
                index,
            ));
        }
    }
    entries.sort_unstable();
    let risk = locked.account_risk_policy();
    let currency = risk.initial_balance().currency();
    let mut balance = risk.initial_balance().amount();
    let mut economics: Vec<QuoteValidatedExecutionEconomicsLedgerV1> =
        Vec::with_capacity(entries.len());
    let mut open: Vec<usize> = Vec::new();
    let mut contract_identity: Option<String> = None;
    for (entry_at, _, _, index) in entries {
        let (origin, ledger) = &mut ordered[index];
        let position = &ledger.positions()[0];
        let financial = entry_inputs(ledger, origin)
            .map_err(|error| binding_error(format!("entry financial inputs: {error:#}")))?;
        let contract = &financial.symbol_contract;
        let recomputed = ExecutionSymbolContractV1::new(
            contract.symbol_name(),
            contract.base_currency(),
            contract.quote_currency(),
            contract.contract_units_per_lot(),
        )
        .map_err(|error| binding_error(error.to_string()))?;
        if &recomputed != contract
            || contract.symbol_name() != locked.symbol_name()
            || financial.entry_conversion.source_currency() != contract.quote_currency()
            || financial.entry_conversion.target_account_currency() != currency
            || contract_identity
                .as_ref()
                .is_some_and(|identity| identity != contract.identity_sha256())
        {
            return Err(binding_error(
                "entry contract/conversion differs from the locked same-symbol account",
            ));
        }
        contract_identity.get_or_insert_with(|| contract.identity_sha256().to_owned());
        financial
            .entry_conversion
            .validate_causal_for(entry_at)
            .map_err(|error| binding_error(error.to_string()))?;
        let conversion_rate = financial
            .entry_conversion
            .conversion_rate_account_per_quote();

        let mut still_open = Vec::with_capacity(open.len());
        for prior in open {
            let previous = &economics[prior];
            if previous.exit_fill_timestamp_unix_ms() < entry_at {
                // Its entry commission was already debited at its own entry.
                balance += previous.net_pnl_account_currency().amount()
                    + previous.entry_commission_account_currency().amount();
                if !balance.is_finite() {
                    return Err(binding_error("settled account balance overflow"));
                }
            } else {
                still_open.push(prior);
            }
        }
        open = still_open;
        let mut unrealized = 0.0;
        for &prior in &open {
            let previous = &economics[prior];
            let (sign, mark) = match previous.direction() {
                ResearchPositionDirectionV1::Long => (1.0, position.entry_book().bid_reference()),
                ResearchPositionDirectionV1::Short => (-1.0, position.entry_book().ask_reference()),
            };
            if mark.timestamp_unix_ms() > entry_at {
                return Err(binding_error("sealed entry book contains a future mark"));
            }
            let floating = sign
                * (mark.price() - previous.modeled_entry_price())
                * previous.base_units()
                * conversion_rate;
            unrealized += floating;
            if !floating.is_finite() || !unrealized.is_finite() {
                return Err(binding_error("marked account-money arithmetic overflow"));
            }
        }
        let money = |amount| {
            AccountMoneyV1::new(currency, amount).map_err(|error| binding_error(error.to_string()))
        };
        let equity = money(balance + unrealized)?;
        let pip_value =
            locked.exit_policy.pip_size * contract.contract_units_per_lot() * conversion_rate;
        let requested_lots =
            risk.entry_lots(origin.confidence, &equity, origin.risk_pips, pip_value)?
                * origin.size_multiplier;
        if !requested_lots.is_finite() || requested_lots <= 0.0 {
            return Err(binding_error(
                "vetoed/invalid size multiplier must not retain a directional entry",
            ));
        }
        let filled_lots = financial.lot_constraints.normalize_down(requested_lots)?;
        let sizing = QuoteEntrySizingEvidenceV3 {
            balance_before_entry: money(balance)?,
            unrealized_before_entry: money(unrealized)?,
            equity_before_entry: equity,
            pip_value_account_per_lot: pip_value,
            risk_fraction: risk.risk_fraction(origin.confidence)? * origin.size_multiplier,
            requested_lots,
            filled_lots,
            entry_conversion: financial.entry_conversion,
            lot_constraints: financial.lot_constraints,
        };
        let execution = execution_economics(ledger, &sizing)
            .map_err(|error| binding_error(format!("sized execution economics: {error:#}")))?;
        execution
            .validate_against_quote_ledger(ledger)
            .map_err(|error| binding_error(error.to_string()))?;
        if execution.filled_lots().to_bits() != filled_lots.to_bits()
            || execution.account_currency() != currency
            || execution.symbol_contract() != contract
        {
            return Err(binding_error(
                "execution economics substituted the computed lots, account or entry contract",
            ));
        }
        balance -= execution.entry_commission_account_currency().amount();
        if !balance.is_finite() {
            return Err(binding_error("entry commission overflows account balance"));
        }
        origin.entry_sizing = Some(sizing);
        open.push(economics.len());
        economics.push(execution);
    }
    Ok(economics)
}

impl LockedPortfolioOuterHoldoutReplaySetV3 {
    pub fn new<E, F>(
        locked: &LockedCanonicalSignalPlanV3<'_>,
        binding: &QuoteValidatedResearchReplayBindingV1,
        policy: &QuoteValidatedResearchReplayPolicyV1,
        snapshot: &SealedHistoricalBidAskQuoteReplayEvidenceV1,
        ordered_lanes: Vec<Vec<SealedHistoricalQuoteValidatedResearchLedgerV1>>,
        entry_inputs: E,
        execution_economics: F,
    ) -> Result<Self, QuoteValidatedOuterHoldoutErrorV1>
    where
        E: FnMut(
            &SealedHistoricalQuoteValidatedResearchLedgerV1,
            &QuoteValidatedDecisionProvenanceV3,
        ) -> anyhow::Result<QuoteEntryFinancialInputsV3>,
        F: FnMut(
            &SealedHistoricalQuoteValidatedResearchLedgerV1,
            &QuoteEntrySizingEvidenceV3,
        ) -> anyhow::Result<QuoteValidatedExecutionEconomicsLedgerV1>,
    {
        locked.validate_replay_binding(binding, policy)?;
        snapshot
            .validate_replay_context_v1(binding, policy)
            .map_err(|error| binding_error(error.to_string()))?;
        if ordered_lanes.len() != locked.portfolio.len() {
            return Err(binding_error(
                "replay must include every locked portfolio lane, even empty lanes",
            ));
        }
        let mut ordered = Vec::new();
        for (lane, ledgers) in ordered_lanes.into_iter().enumerate() {
            for ledger in &ledgers {
                let receipt = ledger.receipt();
                if receipt.replay_policy_sha256() != policy.identity_sha256()
                    || receipt.historical_acquisition_link_manifest_sha256()
                        != Some(snapshot.acquisition_link_manifest_sha256())
                    || ledger.positions().len() + ledger.entry_unavailable().len() != 1
                {
                    return Err(binding_error(
                        "lane ledger changes its reviewed snapshot/policy or decision count",
                    ));
                }
            }
            let provenance = validate_lane(
                locked,
                lane,
                ledgers.iter().map(|ledger| {
                    let position = ledger.positions().first();
                    OutcomeView {
                        decision: ledger.executed_decision(),
                        entry: position.map(|position| {
                            (
                                position.entry_reference().timestamp_unix_ms(),
                                position.modeled_entry_price(),
                            )
                        }),
                        completed_at: position
                            .and_then(|position| {
                                position
                                    .exit_reference()
                                    .map(|exit| exit.timestamp_unix_ms())
                            })
                            .or_else(|| {
                                ledger
                                    .entry_unavailable()
                                    .first()
                                    .map(|entry| entry.deadline_unix_ms())
                            }),
                        time_exit: ledger.executed_time_exit(),
                        plan_sha256: ledger.executed_plan_sha256(),
                        ledger_sha256: ledger.ledger_sha256(),
                    }
                }),
            )?;
            for ledger in &ledgers {
                validate_executed_exit_policy(locked, binding, policy, snapshot, ledger)?;
            }
            ordered.extend(provenance.into_iter().zip(ledgers));
        }
        let economics =
            size_locked_account_entries(locked, &mut ordered, entry_inputs, execution_economics)?;
        // Independent lanes may finish out of order. Order cash observations,
        // retaining the explicit gene/decision origin rather than zipping risk
        // distances against whichever parallel job happens to finish first.
        ordered.sort_by_key(|(origin, ledger)| {
            (
                ledger
                    .positions()
                    .first()
                    .and_then(|position| position.exit_reference())
                    .map(|exit| exit.timestamp_unix_ms())
                    .or_else(|| {
                        ledger
                            .entry_unavailable()
                            .first()
                            .map(|entry| entry.deadline_unix_ms())
                    })
                    .unwrap_or(i64::MAX),
                origin.portfolio_index,
                origin.decision_bar_index,
            )
        });
        let mut by_quote = HashMap::with_capacity(economics.len());
        for execution in economics {
            if by_quote
                .insert(execution.quote_ledger_sha256().to_owned(), execution)
                .is_some()
            {
                return Err(binding_error(
                    "duplicate execution economics for one quote ledger",
                ));
            }
        }
        let mut ordered_execution_economics_ledgers = Vec::new();
        for (_, ledger) in &ordered {
            if !ledger.positions().is_empty() {
                let execution = by_quote.remove(ledger.ledger_sha256()).ok_or_else(|| {
                    outer_error(
                        QuoteValidatedOuterHoldoutErrorCodeV1::MissingExecutionEconomics,
                        "quote position has no matching closed execution economics",
                    )
                })?;
                ordered_execution_economics_ledgers.push(execution);
            }
        }
        if !by_quote.is_empty() {
            return Err(binding_error(
                "execution economics contains outcomes outside the locked replay",
            ));
        }
        let (provenance, ordered_quote_ledgers): (Vec<_>, Vec<_>) = ordered.into_iter().unzip();
        let replay_set = LockedPortfolioOuterHoldoutReplaySetV1 {
            canonical_search_input_receipt_sha256: locked.holdout_scope.receipt_sha256().to_owned(),
            canonical_signal_plan_sha256: locked.identity_sha256.clone(),
            portfolio_identity_sha256: locked.portfolio_identity_sha256.clone(),
            search_config_hash: locked.search_config_hash.to_owned(),
            holdout_scope: locked.holdout_scope.clone(),
            account_id: locked.account_id,
            symbol_id: locked.symbol_id,
            locked_evaluation_window: locked.window,
            reviewed_replay_rule_identity_sha256: binding
                .reviewed_replay_rule_identity_sha256()
                .to_owned(),
            ordered_risk_pips: provenance.iter().map(|origin| origin.risk_pips).collect(),
            ordered_quote_ledgers,
            ordered_execution_economics_ledgers,
        };
        Ok(Self {
            replay_set,
            provenance,
            acquisition_link_manifest_sha256: snapshot
                .acquisition_link_manifest_sha256()
                .to_owned(),
            replay_policy_sha256: policy.identity_sha256().to_owned(),
            exit_policy: locked.exit_policy,
        })
    }
}

/// V2 adds prelocked-policy and per-decision provenance. The embedded V1
/// execution component is the unchanged sealed-fill/account-balance schema,
/// not an assertion that its signal identity used the old trade-count hash.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedOuterHoldoutResearchEvidenceV3 {
    schema_version: u16,
    acquisition_link_manifest_sha256: String,
    replay_policy_sha256: String,
    exit_policy: CanonicalSignalExitPolicyV2,
    account_risk_policy: CanonicalSignalAccountRiskPolicyV3,
    decision_provenance: Vec<QuoteValidatedDecisionProvenanceV3>,
    execution: QuoteValidatedOuterHoldoutResearchEvidenceV1,
    receipt_sha256: String,
}

impl QuoteValidatedOuterHoldoutResearchEvidenceV3 {
    pub fn metrics(&self) -> &QuoteValidatedOuterHoldoutMetricsV1 {
        self.execution.metrics()
    }
    pub fn execution(&self) -> &QuoteValidatedOuterHoldoutResearchEvidenceV1 {
        &self.execution
    }
    pub fn decision_provenance(&self) -> &[QuoteValidatedDecisionProvenanceV3] {
        &self.decision_provenance
    }
    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }
}

pub fn evaluate_locked_portfolio_outer_holdout_v3(
    locked: &LockedCanonicalSignalPlanV3<'_>,
    replay: LockedPortfolioOuterHoldoutReplaySetV3,
) -> Result<QuoteValidatedOuterHoldoutResearchEvidenceV3, QuoteValidatedOuterHoldoutErrorV1> {
    if replay.replay_set.canonical_signal_plan_sha256 != locked.identity_sha256 {
        return Err(binding_error(
            "replay result differs from the still-locked pre-acquisition plan",
        ));
    }
    let per_trade_pip_values: Vec<_> = replay
        .provenance
        .iter()
        .filter_map(|origin| {
            origin
                .entry_sizing
                .as_ref()
                .map(|sizing| sizing.pip_value_account_per_lot)
        })
        .collect();
    let execution = evaluate_bound_quote_ledgers_v1(
        locked.account_risk_policy.initial_balance.clone(),
        BoundPipValues::PerClosedTrade(&per_trade_pip_values),
        replay.replay_set,
        locked.holdout_scope.receipt_sha256().to_owned(),
        locked.identity_sha256.clone(),
        locked.portfolio_identity_sha256.clone(),
        locked.scope_identity_sha256.clone(),
    )?;
    let mut evidence = QuoteValidatedOuterHoldoutResearchEvidenceV3 {
        schema_version: 3,
        acquisition_link_manifest_sha256: replay.acquisition_link_manifest_sha256,
        replay_policy_sha256: replay.replay_policy_sha256,
        exit_policy: replay.exit_policy,
        account_risk_policy: locked.account_risk_policy.clone(),
        decision_provenance: replay.provenance,
        execution,
        receipt_sha256: String::new(),
    };
    evidence.receipt_sha256 =
        streamed_sha256("neoethos.prelocked-quote-outer-holdout.v3", &evidence)?;
    Ok(evidence)
}

#[cfg(test)]
#[path = "quote_validated_outer_holdout_v2_tests.rs"]
mod tests;
