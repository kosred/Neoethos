use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt;

use neoethos_broker_truth::{
    EvidenceWindowV1, ExecutionEconomicsArtifactClassV1, ExecutionEconomicsPromotionEligibilityV1,
    QuoteValidatedExecutionEconomicsLedgerV1, QuoteValidatedResearchAuthorityV1,
    QuoteValidatedResearchPromotionEligibilityV1, QuoteValidatedResearchReplayReceiptV1,
    SealedHistoricalQuoteValidatedResearchLedgerV1,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::CanonicalSearchArtifactScopeV2;

pub const QUOTE_VALIDATED_OUTER_HOLDOUT_SCHEMA_VERSION_V1: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteValidatedOuterHoldoutArtifactClassV1 {
    ResearchOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteValidatedOuterHoldoutPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuoteValidatedOuterHoldoutErrorCodeV1 {
    MissingSealedQuoteValidatedOuterHoldout,
    LegacyForwardTestV2Insufficient,
    LegacyPropFirmV2Insufficient,
    MissingReplayReceipt,
    UnexpectedReplayReceipt,
    DuplicateReplayReceipt,
    ReceiptOrderMismatch,
    BindingMismatch,
    MissingExecutionEconomics,
    InvalidMetricInput,
    ArtifactEncodingFailed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuoteValidatedOuterHoldoutErrorV1 {
    code: QuoteValidatedOuterHoldoutErrorCodeV1,
    detail: String,
}

impl QuoteValidatedOuterHoldoutErrorV1 {
    pub const fn code(&self) -> QuoteValidatedOuterHoldoutErrorCodeV1 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for QuoteValidatedOuterHoldoutErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.detail)
    }
}

impl Error for QuoteValidatedOuterHoldoutErrorV1 {}

fn outer_error(
    code: QuoteValidatedOuterHoldoutErrorCodeV1,
    detail: impl Into<String>,
) -> QuoteValidatedOuterHoldoutErrorV1 {
    QuoteValidatedOuterHoldoutErrorV1 {
        code,
        detail: detail.into(),
    }
}

fn validate_sha256(label: &str, digest: &str) -> Result<(), QuoteValidatedOuterHoldoutErrorV1> {
    if digest.len() != 64
        || !digest
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
            format!("{label} is not an exact SHA-256 digest"),
        ));
    }
    Ok(())
}

fn stable_sha256<T: Serialize>(
    domain: &str,
    value: &T,
) -> Result<String, QuoteValidatedOuterHoldoutErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::ArtifactEncodingFailed,
            format!("cannot encode {domain}: {error}"),
        )
    })?;
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Serialize)]
struct CanonicalSignalPlanHashPayloadV1<'a> {
    canonical_search_input_receipt_sha256: &'a str,
    portfolio_identity_sha256: &'a str,
    search_config_hash: &'a str,
    holdout_scope_identity_sha256: &'a str,
    ordered_signals: &'a [Vec<i8>],
    ordered_risk_pips: &'a [f64],
}

pub fn canonical_locked_portfolio_identity_sha256_v1<T: Serialize>(
    locked_portfolio: &T,
) -> Result<String, QuoteValidatedOuterHoldoutErrorV1> {
    stable_sha256("neoethos.locked-final-portfolio.v1", locked_portfolio)
}

fn canonical_signal_plan_sha256_v1(
    canonical_search_input_receipt_sha256: &str,
    portfolio_identity_sha256: &str,
    search_config_hash: &str,
    holdout_scope_identity_sha256: &str,
    ordered_signals: &[Vec<i8>],
    ordered_risk_pips: &[f64],
) -> Result<String, QuoteValidatedOuterHoldoutErrorV1> {
    stable_sha256(
        "neoethos.canonical-bar-signal-plan.v1",
        &CanonicalSignalPlanHashPayloadV1 {
            canonical_search_input_receipt_sha256,
            portfolio_identity_sha256,
            search_config_hash,
            holdout_scope_identity_sha256,
            ordered_signals,
            ordered_risk_pips,
        },
    )
}

#[derive(Debug)]
pub struct LockedPortfolioOuterHoldoutReplaySetV1 {
    canonical_search_input_receipt_sha256: String,
    canonical_signal_plan_sha256: String,
    portfolio_identity_sha256: String,
    search_config_hash: String,
    holdout_scope: CanonicalSearchArtifactScopeV2,
    account_id: i64,
    symbol_id: i64,
    locked_evaluation_window: EvidenceWindowV1,
    reviewed_replay_rule_identity_sha256: String,
    ordered_risk_pips: Vec<f64>,
    ordered_quote_ledgers: Vec<SealedHistoricalQuoteValidatedResearchLedgerV1>,
    ordered_execution_economics_ledgers: Vec<QuoteValidatedExecutionEconomicsLedgerV1>,
}

impl LockedPortfolioOuterHoldoutReplaySetV1 {
    pub fn new(
        canonical_search_input_receipt_sha256: impl Into<String>,
        canonical_signal_plan_sha256: impl Into<String>,
        portfolio_identity_sha256: impl Into<String>,
        search_config_hash: impl Into<String>,
        holdout_scope: CanonicalSearchArtifactScopeV2,
        account_id: i64,
        symbol_id: i64,
        locked_evaluation_window: EvidenceWindowV1,
        reviewed_replay_rule_identity_sha256: impl Into<String>,
        ordered_risk_pips: Vec<f64>,
        ordered_quote_ledgers: Vec<SealedHistoricalQuoteValidatedResearchLedgerV1>,
        ordered_execution_economics_ledgers: Vec<QuoteValidatedExecutionEconomicsLedgerV1>,
    ) -> Result<Self, QuoteValidatedOuterHoldoutErrorV1> {
        let replay_set = Self {
            canonical_search_input_receipt_sha256: canonical_search_input_receipt_sha256.into(),
            canonical_signal_plan_sha256: canonical_signal_plan_sha256.into(),
            portfolio_identity_sha256: portfolio_identity_sha256.into(),
            search_config_hash: search_config_hash.into(),
            holdout_scope,
            account_id,
            symbol_id,
            locked_evaluation_window,
            reviewed_replay_rule_identity_sha256: reviewed_replay_rule_identity_sha256.into(),
            ordered_risk_pips,
            ordered_quote_ledgers,
            ordered_execution_economics_ledgers,
        };
        replay_set.validate_shape()?;
        Ok(replay_set)
    }

    pub const fn holdout_scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        &self.holdout_scope
    }

    pub fn canonical_search_input_receipt_sha256(&self) -> &str {
        &self.canonical_search_input_receipt_sha256
    }

    pub fn portfolio_identity_sha256(&self) -> &str {
        &self.portfolio_identity_sha256
    }

    pub fn search_config_hash(&self) -> &str {
        &self.search_config_hash
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn symbol_id(&self) -> i64 {
        self.symbol_id
    }

    pub const fn locked_evaluation_window(&self) -> EvidenceWindowV1 {
        self.locked_evaluation_window
    }

    pub fn reviewed_replay_rule_identity_sha256(&self) -> &str {
        &self.reviewed_replay_rule_identity_sha256
    }

    pub fn canonical_signal_plan_sha256(&self) -> &str {
        &self.canonical_signal_plan_sha256
    }

    fn validate_shape(&self) -> Result<(), QuoteValidatedOuterHoldoutErrorV1> {
        self.holdout_scope.validate().map_err(|error| {
            outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
                format!("locked holdout scope is invalid: {error}"),
            )
        })?;
        for (label, digest) in [
            (
                "canonical search input receipt",
                self.canonical_search_input_receipt_sha256.as_str(),
            ),
            (
                "canonical signal plan",
                self.canonical_signal_plan_sha256.as_str(),
            ),
            ("locked portfolio", self.portfolio_identity_sha256.as_str()),
            (
                "reviewed replay rule",
                self.reviewed_replay_rule_identity_sha256.as_str(),
            ),
        ] {
            validate_sha256(label, digest)?;
        }
        if self.search_config_hash.trim().is_empty() || self.account_id <= 0 || self.symbol_id <= 0
        {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
                "locked replay set has an empty config identity or non-positive broker identity",
            ));
        }
        if self.ordered_quote_ledgers.is_empty() {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::MissingReplayReceipt,
                "locked portfolio has no sealed quote-replay ledger",
            ));
        }
        if self.ordered_risk_pips.len() != self.ordered_quote_ledgers.len()
            || self
                .ordered_risk_pips
                .iter()
                .any(|risk| !risk.is_finite() || *risk <= 0.0)
        {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
                "every ordered one-decision replay requires one positive exact risk distance",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedOuterHoldoutTradeOutcomeV1 {
    quote_ledger_sha256: String,
    execution_economics_ledger_sha256: String,
    exit_timestamp_unix_ms: i64,
    net_pnl_account_currency: f64,
    net_pips: f64,
    r_multiple: f64,
}

impl QuoteValidatedOuterHoldoutTradeOutcomeV1 {
    pub fn quote_ledger_sha256(&self) -> &str {
        &self.quote_ledger_sha256
    }

    pub fn execution_economics_ledger_sha256(&self) -> &str {
        &self.execution_economics_ledger_sha256
    }

    pub const fn exit_timestamp_unix_ms(&self) -> i64 {
        self.exit_timestamp_unix_ms
    }

    pub const fn net_pnl_account_currency(&self) -> f64 {
        self.net_pnl_account_currency
    }

    pub const fn net_pips(&self) -> f64 {
        self.net_pips
    }

    pub const fn r_multiple(&self) -> f64 {
        self.r_multiple
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedOuterHoldoutMetricsV1 {
    net_profit: f64,
    sharpe: Option<f64>,
    peak_equity: f64,
    max_drawdown: f64,
    win_rate: Option<f64>,
    profit_factor: Option<f64>,
    expectancy: Option<f64>,
    trade_count: usize,
    consistency: Option<f64>,
    max_daily_drawdown: f64,
    entry_unavailable: usize,
}

impl QuoteValidatedOuterHoldoutMetricsV1 {
    pub const fn net_profit(&self) -> f64 {
        self.net_profit
    }

    pub const fn sharpe(&self) -> Option<f64> {
        self.sharpe
    }

    pub const fn peak_equity(&self) -> f64 {
        self.peak_equity
    }

    pub const fn max_drawdown(&self) -> f64 {
        self.max_drawdown
    }

    pub const fn win_rate(&self) -> Option<f64> {
        self.win_rate
    }

    pub const fn profit_factor(&self) -> Option<f64> {
        self.profit_factor
    }

    pub const fn expectancy(&self) -> Option<f64> {
        self.expectancy
    }

    pub const fn trade_count(&self) -> usize {
        self.trade_count
    }

    pub const fn consistency(&self) -> Option<f64> {
        self.consistency
    }

    pub const fn max_daily_drawdown(&self) -> f64 {
        self.max_daily_drawdown
    }

    pub const fn entry_unavailable(&self) -> usize {
        self.entry_unavailable
    }
}

fn derive_complete_quote_validated_metrics_v1(
    trade_outcomes: &[QuoteValidatedOuterHoldoutTradeOutcomeV1],
    entry_unavailable: usize,
) -> Result<QuoteValidatedOuterHoldoutMetricsV1, QuoteValidatedOuterHoldoutErrorV1> {
    let pnl = trade_outcomes
        .iter()
        .map(QuoteValidatedOuterHoldoutTradeOutcomeV1::net_pnl_account_currency)
        .collect::<Vec<_>>();
    if pnl.iter().any(|value| !value.is_finite()) {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::InvalidMetricInput,
            "quote-validated net PnL contains a non-finite value",
        ));
    }

    let trade_count = pnl.len();
    let net_profit = pnl.iter().sum::<f64>();
    let expectancy = (trade_count > 0).then_some(net_profit / trade_count as f64);
    let wins = pnl.iter().filter(|value| **value > 0.0).count();
    let win_rate = (trade_count > 0).then_some(wins as f64 / trade_count as f64);
    let gross_profit = pnl.iter().filter(|value| **value > 0.0).sum::<f64>();
    let gross_loss = -pnl.iter().filter(|value| **value < 0.0).sum::<f64>();
    let profit_factor = (gross_loss > 0.0).then_some(gross_profit / gross_loss);

    let sharpe = if trade_count >= 2 {
        let mean = net_profit / trade_count as f64;
        let variance = pnl
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / (trade_count - 1) as f64;
        (variance > 0.0).then_some(mean / variance.sqrt())
    } else {
        None
    };

    let mut equity = 0.0_f64;
    let mut peak_equity = 0.0_f64;
    let mut max_drawdown = 0.0_f64;
    let mut daily = BTreeMap::<i64, Vec<f64>>::new();
    for outcome in trade_outcomes {
        equity += outcome.net_pnl_account_currency();
        peak_equity = peak_equity.max(equity);
        max_drawdown = max_drawdown.max(peak_equity - equity);
        daily
            .entry(outcome.exit_timestamp_unix_ms().div_euclid(86_400_000))
            .or_default()
            .push(outcome.net_pnl_account_currency());
    }
    let mut max_daily_drawdown = 0.0_f64;
    let mut positive_days = 0_usize;
    for day in daily.values() {
        let mut day_equity = 0.0_f64;
        let mut day_peak = 0.0_f64;
        for value in day {
            day_equity += value;
            day_peak = day_peak.max(day_equity);
            max_daily_drawdown = max_daily_drawdown.max(day_peak - day_equity);
        }
        if day_equity > 0.0 {
            positive_days += 1;
        }
    }
    let consistency = (!daily.is_empty()).then_some(positive_days as f64 / daily.len() as f64);

    Ok(QuoteValidatedOuterHoldoutMetricsV1 {
        net_profit,
        sharpe,
        peak_equity,
        max_drawdown,
        win_rate,
        profit_factor,
        expectancy,
        trade_count,
        consistency,
        max_daily_drawdown,
        entry_unavailable,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedOuterHoldoutReceiptV1 {
    schema_version: u16,
    canonical_search_input_receipt_sha256: String,
    canonical_signal_plan_sha256: String,
    portfolio_identity_sha256: String,
    search_config_hash: String,
    holdout_scope_identity_sha256: String,
    account_id: i64,
    symbol_id: i64,
    locked_evaluation_window: EvidenceWindowV1,
    reviewed_replay_rule_identity_sha256: String,
    quote_replay_receipts: Vec<QuoteValidatedResearchReplayReceiptV1>,
    ordered_historical_link_manifest_sha256s: Vec<String>,
    ordered_execution_economics_ledger_sha256s: Vec<String>,
    metrics: QuoteValidatedOuterHoldoutMetricsV1,
    artifact_class: QuoteValidatedOuterHoldoutArtifactClassV1,
    promotion_eligibility: QuoteValidatedOuterHoldoutPromotionEligibilityV1,
    receipt_sha256: String,
}

impl QuoteValidatedOuterHoldoutReceiptV1 {
    pub fn canonical_search_input_receipt_sha256(&self) -> &str {
        &self.canonical_search_input_receipt_sha256
    }

    pub fn canonical_signal_plan_sha256(&self) -> &str {
        &self.canonical_signal_plan_sha256
    }

    pub fn portfolio_identity_sha256(&self) -> &str {
        &self.portfolio_identity_sha256
    }

    pub fn search_config_hash(&self) -> &str {
        &self.search_config_hash
    }

    pub fn holdout_scope_identity_sha256(&self) -> &str {
        &self.holdout_scope_identity_sha256
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn symbol_id(&self) -> i64 {
        self.symbol_id
    }

    pub const fn locked_evaluation_window(&self) -> EvidenceWindowV1 {
        self.locked_evaluation_window
    }

    pub fn quote_replay_receipts(&self) -> &[QuoteValidatedResearchReplayReceiptV1] {
        &self.quote_replay_receipts
    }

    pub fn reviewed_replay_rule_identity_sha256(&self) -> &str {
        &self.reviewed_replay_rule_identity_sha256
    }

    pub fn ordered_historical_link_manifest_sha256s(&self) -> &[String] {
        &self.ordered_historical_link_manifest_sha256s
    }

    pub fn metrics(&self) -> &QuoteValidatedOuterHoldoutMetricsV1 {
        &self.metrics
    }

    pub const fn artifact_class(&self) -> QuoteValidatedOuterHoldoutArtifactClassV1 {
        self.artifact_class
    }

    pub const fn promotion_eligibility(&self) -> QuoteValidatedOuterHoldoutPromotionEligibilityV1 {
        self.promotion_eligibility
    }

    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }
}

#[derive(Serialize)]
struct OuterHoldoutReceiptHashPayloadV1<'a> {
    schema_version: u16,
    canonical_search_input_receipt_sha256: &'a str,
    canonical_signal_plan_sha256: &'a str,
    portfolio_identity_sha256: &'a str,
    search_config_hash: &'a str,
    holdout_scope_identity_sha256: &'a str,
    account_id: i64,
    symbol_id: i64,
    locked_evaluation_window: EvidenceWindowV1,
    reviewed_replay_rule_identity_sha256: &'a str,
    quote_replay_receipts: &'a [QuoteValidatedResearchReplayReceiptV1],
    ordered_historical_link_manifest_sha256s: &'a [String],
    ordered_execution_economics_ledger_sha256s: &'a [String],
    metrics: &'a QuoteValidatedOuterHoldoutMetricsV1,
    artifact_class: QuoteValidatedOuterHoldoutArtifactClassV1,
    promotion_eligibility: QuoteValidatedOuterHoldoutPromotionEligibilityV1,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedOuterHoldoutResearchEvidenceV1 {
    receipt: QuoteValidatedOuterHoldoutReceiptV1,
    metrics: QuoteValidatedOuterHoldoutMetricsV1,
    trade_outcomes: Vec<QuoteValidatedOuterHoldoutTradeOutcomeV1>,
}

impl QuoteValidatedOuterHoldoutResearchEvidenceV1 {
    pub fn receipt(&self) -> &QuoteValidatedOuterHoldoutReceiptV1 {
        &self.receipt
    }

    pub fn metrics(&self) -> &QuoteValidatedOuterHoldoutMetricsV1 {
        &self.metrics
    }

    pub fn trade_outcomes(&self) -> &[QuoteValidatedOuterHoldoutTradeOutcomeV1] {
        &self.trade_outcomes
    }
}

pub(crate) fn require_quote_validated_outer_holdout_v1(
    evidence: Option<&QuoteValidatedOuterHoldoutResearchEvidenceV1>,
) -> Result<&QuoteValidatedOuterHoldoutResearchEvidenceV1, QuoteValidatedOuterHoldoutErrorV1> {
    evidence.ok_or_else(|| {
        outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::MissingSealedQuoteValidatedOuterHoldout,
            "legacy ForwardTest V2 and PropFirm V2 artifacts are diagnostics only; complete promotion evidence requires an explicit sealed quote-validated outer holdout",
        )
    })
}

pub fn evaluate_locked_portfolio_outer_holdout_v1(
    locked_portfolio: &impl Serialize,
    ordered_signals: &[Vec<i8>],
    search_config_hash: &str,
    expected_holdout_scope: &CanonicalSearchArtifactScopeV2,
    initial_balance: f64,
    pip_value_per_lot: f64,
    replay_set: LockedPortfolioOuterHoldoutReplaySetV1,
) -> Result<QuoteValidatedOuterHoldoutResearchEvidenceV1, QuoteValidatedOuterHoldoutErrorV1> {
    replay_set.validate_shape()?;
    expected_holdout_scope.validate().map_err(|error| {
        outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
            format!("expected locked holdout scope is invalid: {error}"),
        )
    })?;
    if expected_holdout_scope != &replay_set.holdout_scope {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
            "sealed replay set holdout scope differs from the locked final outer holdout",
        ));
    }
    if !initial_balance.is_finite()
        || initial_balance <= 0.0
        || !pip_value_per_lot.is_finite()
        || pip_value_per_lot <= 0.0
    {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::InvalidMetricInput,
            "quote-validated metrics require positive finite balance and pip value per lot",
        ));
    }
    if search_config_hash != replay_set.search_config_hash {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
            "locked replay set search config differs from the final portfolio config",
        ));
    }

    let holdout_scope_identity_sha256 =
        replay_set
            .holdout_scope
            .identity_sha256()
            .map_err(|error| {
                outer_error(
                    QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
                    format!("cannot identify locked holdout scope: {error}"),
                )
            })?;
    let canonical_search_input_receipt_sha256 = replay_set
        .holdout_scope
        .receipt()
        .identity_sha256()
        .map_err(|error| {
            outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
                format!("cannot identify canonical search receipt: {error}"),
            )
        })?;
    if canonical_search_input_receipt_sha256 != replay_set.canonical_search_input_receipt_sha256 {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
            "locked holdout scope receipt differs from replay-set receipt identity",
        ));
    }
    let window = replay_set.holdout_scope.evaluated_window();
    if replay_set.locked_evaluation_window.from_unix_ms_inclusive() != window.timestamp_start_ms()
        || replay_set.locked_evaluation_window.to_unix_ms_exclusive() <= window.timestamp_end_ms()
    {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
            "quote replay locked window does not exactly begin at and extend beyond the canonical holdout",
        ));
    }

    let portfolio_identity_sha256 =
        canonical_locked_portfolio_identity_sha256_v1(locked_portfolio)?;
    if portfolio_identity_sha256 != replay_set.portfolio_identity_sha256 {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
            "sealed replay set belongs to a different locked final portfolio",
        ));
    }
    let canonical_signal_plan_sha256 = canonical_signal_plan_sha256_v1(
        &canonical_search_input_receipt_sha256,
        &portfolio_identity_sha256,
        search_config_hash,
        &holdout_scope_identity_sha256,
        ordered_signals,
        &replay_set.ordered_risk_pips,
    )?;
    if canonical_signal_plan_sha256 != replay_set.canonical_signal_plan_sha256 {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
            "sealed replay set canonical signal plan differs from recomputed final signals",
        ));
    }

    let mut quote_replay_receipts = Vec::with_capacity(replay_set.ordered_quote_ledgers.len());
    let mut ordered_historical_link_manifest_sha256s =
        Vec::with_capacity(replay_set.ordered_quote_ledgers.len());
    let mut observed_quote_ledgers = HashSet::with_capacity(replay_set.ordered_quote_ledgers.len());
    let mut entry_unavailable = 0_usize;
    let mut trade_outcomes = Vec::new();
    let mut economics = replay_set.ordered_execution_economics_ledgers.iter();
    let mut previous_exit_timestamp = None;

    for (ordinal, (quote_ledger, risk_pips)) in replay_set
        .ordered_quote_ledgers
        .iter()
        .zip(&replay_set.ordered_risk_pips)
        .enumerate()
    {
        if quote_ledger.authority() != QuoteValidatedResearchAuthorityV1::HistoricalBidAskQuotesOnly
            || quote_ledger.promotion_eligibility()
                != QuoteValidatedResearchPromotionEligibilityV1::NotPromotionEligible
        {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::UnexpectedReplayReceipt,
                format!(
                    "ordered quote ledger {ordinal} is not sealed historical research evidence"
                ),
            ));
        }
        let receipt = quote_ledger.receipt();
        let historical_acquisition_link_manifest_sha256 = receipt
            .historical_acquisition_link_manifest_sha256()
            .ok_or_else(|| {
                outer_error(
                    QuoteValidatedOuterHoldoutErrorCodeV1::MissingReplayReceipt,
                    format!("ordered quote ledger {ordinal} has no immutable acquisition link"),
                )
            })?;
        for (label, observed, expected) in [
            (
                "canonical receipt",
                receipt.canonical_search_input_receipt_sha256(),
                canonical_search_input_receipt_sha256.as_str(),
            ),
            (
                "canonical signal plan",
                receipt.canonical_signal_plan_sha256(),
                canonical_signal_plan_sha256.as_str(),
            ),
            (
                "reviewed replay rule",
                receipt.reviewed_replay_rule_identity_sha256(),
                replay_set.reviewed_replay_rule_identity_sha256.as_str(),
            ),
        ] {
            if observed != expected {
                return Err(outer_error(
                    QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
                    format!("ordered quote ledger {ordinal} {label} differs"),
                ));
            }
        }
        if receipt.account_id() != replay_set.account_id
            || receipt.symbol_id() != replay_set.symbol_id
            || receipt.locked_evaluation_window() != replay_set.locked_evaluation_window
        {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
                format!("ordered quote ledger {ordinal} account/symbol/window differs"),
            ));
        }
        if !observed_quote_ledgers.insert(quote_ledger.ledger_sha256().to_owned()) {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::DuplicateReplayReceipt,
                format!("ordered quote ledger {ordinal} repeats a ledger identity"),
            ));
        }
        ordered_historical_link_manifest_sha256s
            .push(historical_acquisition_link_manifest_sha256.to_owned());
        quote_replay_receipts.push(receipt.clone());

        if quote_ledger.positions().len() > 1
            || quote_ledger.entry_unavailable().len() > 1
            || quote_ledger.positions().len() + quote_ledger.entry_unavailable().len() != 1
        {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::UnexpectedReplayReceipt,
                "V1 outer-holdout integration requires exactly one decision outcome per sealed ledger",
            ));
        }
        entry_unavailable += quote_ledger.entry_unavailable().len();
        let Some(position) = quote_ledger.positions().first() else {
            continue;
        };
        let execution = economics.next().ok_or_else(|| {
            outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::MissingExecutionEconomics,
                format!("closed quote position {ordinal} has no ordered economics ledger"),
            )
        })?;
        execution.canonical_json_bytes().map_err(|error| {
            outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
                format!("execution economics ledger {ordinal} is invalid: {error}"),
            )
        })?;
        if execution.quote_ledger_sha256() != quote_ledger.ledger_sha256()
            || execution.artifact_class() != ExecutionEconomicsArtifactClassV1::ResearchOnly
            || execution.promotion_eligibility()
                != ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible
        {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::ReceiptOrderMismatch,
                format!("execution economics ledger {ordinal} is detached from quote-ledger order"),
            ));
        }
        let exit_timestamp_unix_ms = position
            .exit_reference()
            .ok_or_else(|| {
                outer_error(
                    QuoteValidatedOuterHoldoutErrorCodeV1::MissingExecutionEconomics,
                    format!("quote position {ordinal} has no closed exit reference"),
                )
            })?
            .timestamp_unix_ms();
        if previous_exit_timestamp.is_some_and(|previous| previous > exit_timestamp_unix_ms) {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::ReceiptOrderMismatch,
                "ordered quote-ledger exits are not chronological",
            ));
        }
        previous_exit_timestamp = Some(exit_timestamp_unix_ms);
        let net_pnl_account_currency = execution.net_pnl_account_currency().amount();
        let pip_money = pip_value_per_lot * execution.filled_lots();
        if !pip_money.is_finite() || pip_money <= 0.0 {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::InvalidMetricInput,
                "execution economics cannot be converted to net pips with the exact lot size",
            ));
        }
        let net_pips = net_pnl_account_currency / pip_money;
        trade_outcomes.push(QuoteValidatedOuterHoldoutTradeOutcomeV1 {
            quote_ledger_sha256: quote_ledger.ledger_sha256().to_owned(),
            execution_economics_ledger_sha256: execution.ledger_sha256().to_owned(),
            exit_timestamp_unix_ms,
            net_pnl_account_currency,
            net_pips,
            r_multiple: net_pips / risk_pips,
        });
    }
    if economics.next().is_some() {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::UnexpectedReplayReceipt,
            "execution economics contains an extra ledger outside ordered closed quote positions",
        ));
    }

    let metrics = derive_complete_quote_validated_metrics_v1(&trade_outcomes, entry_unavailable)?;
    let ordered_execution_economics_ledger_sha256s = replay_set
        .ordered_execution_economics_ledgers
        .iter()
        .map(|ledger| ledger.ledger_sha256().to_owned())
        .collect::<Vec<_>>();
    let mut receipt = QuoteValidatedOuterHoldoutReceiptV1 {
        schema_version: QUOTE_VALIDATED_OUTER_HOLDOUT_SCHEMA_VERSION_V1,
        canonical_search_input_receipt_sha256,
        canonical_signal_plan_sha256,
        portfolio_identity_sha256,
        search_config_hash: search_config_hash.to_owned(),
        holdout_scope_identity_sha256,
        account_id: replay_set.account_id,
        symbol_id: replay_set.symbol_id,
        locked_evaluation_window: replay_set.locked_evaluation_window,
        reviewed_replay_rule_identity_sha256: replay_set.reviewed_replay_rule_identity_sha256,
        quote_replay_receipts,
        ordered_historical_link_manifest_sha256s,
        ordered_execution_economics_ledger_sha256s,
        metrics: metrics.clone(),
        artifact_class: QuoteValidatedOuterHoldoutArtifactClassV1::ResearchOnly,
        promotion_eligibility:
            QuoteValidatedOuterHoldoutPromotionEligibilityV1::NotPromotionEligible,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = stable_sha256(
        "neoethos.quote-validated-outer-holdout-receipt.v1",
        &OuterHoldoutReceiptHashPayloadV1 {
            schema_version: receipt.schema_version,
            canonical_search_input_receipt_sha256: &receipt.canonical_search_input_receipt_sha256,
            canonical_signal_plan_sha256: &receipt.canonical_signal_plan_sha256,
            portfolio_identity_sha256: &receipt.portfolio_identity_sha256,
            search_config_hash: &receipt.search_config_hash,
            holdout_scope_identity_sha256: &receipt.holdout_scope_identity_sha256,
            account_id: receipt.account_id,
            symbol_id: receipt.symbol_id,
            locked_evaluation_window: receipt.locked_evaluation_window,
            reviewed_replay_rule_identity_sha256: &receipt.reviewed_replay_rule_identity_sha256,
            quote_replay_receipts: &receipt.quote_replay_receipts,
            ordered_historical_link_manifest_sha256s: &receipt
                .ordered_historical_link_manifest_sha256s,
            ordered_execution_economics_ledger_sha256s: &receipt
                .ordered_execution_economics_ledger_sha256s,
            metrics: &receipt.metrics,
            artifact_class: receipt.artifact_class,
            promotion_eligibility: receipt.promotion_eligibility,
        },
    )?;
    Ok(QuoteValidatedOuterHoldoutResearchEvidenceV1 {
        receipt,
        metrics,
        trade_outcomes,
    })
}
