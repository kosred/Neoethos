use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use neoethos_broker_truth::{
    AccountMoneyV1, EvidenceWindowV1, ExecutionEconomicsArtifactClassV1,
    ExecutionEconomicsPromotionEligibilityV1, QuoteValidatedExecutionEconomicsLedgerV1,
    QuoteValidatedResearchAuthorityV1, QuoteValidatedResearchPromotionEligibilityV1,
    QuoteValidatedResearchReplayReceiptV1, SealedHistoricalQuoteValidatedResearchLedgerV1,
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
    // Preserve the exact historical domain + compact JSON identity without
    // allocating a second expanded copy of every nested portfolio receipt.
    struct HashWriter(Sha256);
    impl std::io::Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut hash = HashWriter(Sha256::new());
    hash.0.update(domain.as_bytes());
    hash.0.update([0]);
    // JSON byte arrays emit a write for each number and separator. Batch those
    // tiny fragments before SHA-256 without materializing the expanded JSON or
    // changing a single byte of the historical identity stream.
    let mut writer = std::io::BufWriter::with_capacity(64 * 1024, hash);
    serde_json::to_writer(&mut writer, value).map_err(|error| {
        outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::ArtifactEncodingFailed,
            format!("cannot encode {domain}: {error}"),
        )
    })?;
    // into_inner flushes the final partial block and propagates its error;
    // dropping a buffered writer would silently discard a flush failure.
    let hash = writer.into_inner().map_err(|error| {
        outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::ArtifactEncodingFailed,
            format!("cannot finish encoding {domain}: {error}"),
        )
    })?;
    Ok(format!("{:x}", hash.0.finalize()))
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
    account_currency: String,
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

    pub fn account_currency(&self) -> &str {
        &self.account_currency
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
    metric_basis: &'static str,
    initial_balance: AccountMoneyV1,
    ending_balance: f64,
    net_profit: f64,
    net_return_fraction: f64,
    sharpe: Option<f64>,
    sharpe_unavailable_reason: &'static str,
    peak_equity: f64,
    max_drawdown: f64,
    max_drawdown_fraction: f64,
    win_rate: Option<f64>,
    profit_factor: Option<f64>,
    expectancy: Option<f64>,
    trade_count: usize,
    consistency: Option<f64>,
    max_daily_drawdown: f64,
    max_daily_drawdown_fraction: Option<f64>,
    entry_unavailable: usize,
}

impl QuoteValidatedOuterHoldoutMetricsV1 {
    /// Closed-trade balance observations, not mark-to-market portfolio equity.
    pub const fn metric_basis(&self) -> &'static str {
        self.metric_basis
    }

    pub fn initial_balance(&self) -> &AccountMoneyV1 {
        &self.initial_balance
    }

    pub const fn ending_balance(&self) -> f64 {
        self.ending_balance
    }

    pub const fn net_profit(&self) -> f64 {
        self.net_profit
    }

    /// Fraction of starting capital, not percentage points.
    pub const fn net_return_fraction(&self) -> f64 {
        self.net_return_fraction
    }

    /// Unavailable without a regular marked-equity and benchmark return series.
    /// Irregular closed-trade cash outcomes do not supply those observations.
    pub const fn sharpe(&self) -> Option<f64> {
        self.sharpe
    }

    pub const fn sharpe_unavailable_reason(&self) -> &'static str {
        self.sharpe_unavailable_reason
    }

    /// Compatibility name: this is the realized balance peak, not open equity.
    pub const fn peak_equity(&self) -> f64 {
        self.peak_equity
    }

    pub const fn max_drawdown(&self) -> f64 {
        self.max_drawdown
    }

    /// Maximum closed-balance decline divided by its contemporaneous peak.
    pub const fn max_drawdown_fraction(&self) -> f64 {
        self.max_drawdown_fraction
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

    /// UTC-day peak-to-trough closed-balance fraction, not a prop-firm rule.
    /// Unavailable if any observed day starts at a non-positive cash balance;
    /// closed cash alone cannot establish insolvency while other trades are open.
    pub const fn max_daily_drawdown_fraction(&self) -> Option<f64> {
        self.max_daily_drawdown_fraction
    }

    pub const fn entry_unavailable(&self) -> usize {
        self.entry_unavailable
    }
}

fn finite_metric_v1(label: &str, value: f64) -> Result<f64, QuoteValidatedOuterHoldoutErrorV1> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::InvalidMetricInput,
            format!("quote-validated {label} is not finite"),
        ))
    }
}

fn derive_complete_quote_validated_metrics_v1(
    initial_balance: &AccountMoneyV1,
    trade_outcomes: &[QuoteValidatedOuterHoldoutTradeOutcomeV1],
    entry_unavailable: usize,
) -> Result<QuoteValidatedOuterHoldoutMetricsV1, QuoteValidatedOuterHoldoutErrorV1> {
    // Revalidate because AccountMoneyV1 also has an untrusted Deserialize path.
    let initial_balance = AccountMoneyV1::new(initial_balance.currency(), initial_balance.amount())
        .map_err(|error| {
            outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::InvalidMetricInput,
                error.to_string(),
            )
        })?;
    let start = initial_balance.amount();
    if start <= 0.0 {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::InvalidMetricInput,
            "quote-validated starting balance must be positive",
        ));
    }
    for outcome in trade_outcomes {
        if outcome.account_currency() != initial_balance.currency() {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
                "closed execution currency differs from the run's starting balance currency",
            ));
        }
        finite_metric_v1("net PnL", outcome.net_pnl_account_currency())?;
        finite_metric_v1("net pips", outcome.net_pips())?;
        finite_metric_v1("R multiple", outcome.r_multiple())?;
    }
    if trade_outcomes
        .windows(2)
        .any(|pair| pair[0].exit_timestamp_unix_ms() > pair[1].exit_timestamp_unix_ms())
    {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::ReceiptOrderMismatch,
            "closed execution outcomes are not chronological",
        ));
    }

    let trade_count = trade_outcomes.len();
    let mut net_profit = 0.0;
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;
    let mut wins = 0_usize;
    let mut equity = start;
    let mut peak_equity = start;
    let mut max_drawdown = 0.0_f64;
    let mut max_drawdown_fraction = 0.0_f64;
    let mut max_daily_drawdown = 0.0_f64;
    let mut max_daily_drawdown_fraction = Some(0.0_f64);
    let mut previous_day = None;
    let mut day_peak = start;
    let mut day_pnl = 0.0;
    let mut active_days = 0_usize;
    let mut positive_days = 0_usize;

    // Only one balance observation is known per exit timestamp. An arbitrary
    // ordering of simultaneous portfolio closes must not invent interim peaks.
    // This ordered cash-ledger pass uses constant auxiliary memory.
    for closes in trade_outcomes
        .chunk_by(|left, right| left.exit_timestamp_unix_ms() == right.exit_timestamp_unix_ms())
    {
        let day = closes[0].exit_timestamp_unix_ms().div_euclid(86_400_000);
        if previous_day != Some(day) {
            if previous_day.is_some() && day_pnl > 0.0 {
                positive_days += 1;
            }
            active_days += 1;
            previous_day = Some(day);
            day_peak = equity;
            if day_peak <= 0.0 {
                max_daily_drawdown_fraction = None;
            }
            day_pnl = 0.0;
        }
        let mut close_pnl = 0.0;
        for outcome in closes {
            let pnl = outcome.net_pnl_account_currency();
            close_pnl = finite_metric_v1("simultaneous closed PnL", close_pnl + pnl)?;
            if pnl > 0.0 {
                wins += 1;
                gross_profit = finite_metric_v1("gross winning PnL", gross_profit + pnl)?;
            } else if pnl < 0.0 {
                gross_loss = finite_metric_v1("gross losing PnL", gross_loss - pnl)?;
            }
        }
        net_profit = finite_metric_v1("total net PnL", net_profit + close_pnl)?;
        equity = finite_metric_v1("closing balance", start + net_profit)?;
        day_pnl = finite_metric_v1("daily net PnL", day_pnl + close_pnl)?;
        peak_equity = peak_equity.max(equity);
        day_peak = day_peak.max(equity);
        let drawdown = finite_metric_v1("closed-balance drawdown", peak_equity - equity)?;
        let daily_drawdown = finite_metric_v1("daily closed-balance drawdown", day_peak - equity)?;
        max_drawdown = max_drawdown.max(drawdown);
        max_drawdown_fraction = max_drawdown_fraction.max(finite_metric_v1(
            "closed-balance drawdown fraction",
            drawdown / peak_equity,
        )?);
        max_daily_drawdown = max_daily_drawdown.max(daily_drawdown);
        if let Some(maximum) = max_daily_drawdown_fraction {
            max_daily_drawdown_fraction = Some(maximum.max(finite_metric_v1(
                "daily closed-balance drawdown fraction",
                daily_drawdown / day_peak,
            )?));
        }
    }
    if previous_day.is_some() && day_pnl > 0.0 {
        positive_days += 1;
    }
    let expectancy = (trade_count > 0).then(|| net_profit / trade_count as f64);
    let win_rate = (trade_count > 0).then(|| wins as f64 / trade_count as f64);
    let profit_factor = if gross_loss > 0.0 {
        Some(finite_metric_v1(
            "profit factor",
            gross_profit / gross_loss,
        )?)
    } else {
        None
    };
    let consistency = (active_days > 0).then(|| positive_days as f64 / active_days as f64);
    let net_return_fraction = finite_metric_v1("net return fraction", net_profit / start)?;

    Ok(QuoteValidatedOuterHoldoutMetricsV1 {
        metric_basis: "closed_trade_balance_at_exit_timestamps",
        initial_balance,
        ending_balance: equity,
        net_profit,
        net_return_fraction,
        // Trade cash PnL is neither a periodic return series nor marked equity.
        // Keep the existing optional field explicitly unavailable until those
        // observations and the benchmark policy are supplied by the producer.
        sharpe: None,
        sharpe_unavailable_reason: "regular_mark_to_market_and_benchmark_returns_not_supplied",
        peak_equity,
        max_drawdown,
        max_drawdown_fraction,
        win_rate,
        profit_factor,
        expectancy,
        trade_count,
        consistency,
        max_daily_drawdown,
        max_daily_drawdown_fraction,
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
    initial_balance: AccountMoneyV1,
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
    if !initial_balance.amount().is_finite()
        || initial_balance.amount() <= 0.0
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

    evaluate_bound_quote_ledgers_v1(
        initial_balance,
        BoundPipValues::Fixed(pip_value_per_lot),
        replay_set,
        canonical_search_input_receipt_sha256,
        canonical_signal_plan_sha256,
        portfolio_identity_sha256,
        holdout_scope_identity_sha256,
    )
}

// Both signal-plan versions share the same sealed-fill, economics and metric
// consumer. Only their input-identity/provenance validation differs.
enum BoundPipValues<'a> {
    Fixed(f64),
    PerClosedTrade(&'a [f64]),
}

fn evaluate_bound_quote_ledgers_v1(
    initial_balance: AccountMoneyV1,
    pip_values: BoundPipValues<'_>,
    replay_set: LockedPortfolioOuterHoldoutReplaySetV1,
    canonical_search_input_receipt_sha256: String,
    canonical_signal_plan_sha256: String,
    portfolio_identity_sha256: String,
    holdout_scope_identity_sha256: String,
) -> Result<QuoteValidatedOuterHoldoutResearchEvidenceV1, QuoteValidatedOuterHoldoutErrorV1> {
    let valid_pip_values = match &pip_values {
        BoundPipValues::Fixed(value) => value.is_finite() && *value > 0.0,
        BoundPipValues::PerClosedTrade(values) => {
            values.len() == replay_set.ordered_execution_economics_ledgers.len()
                && values.len()
                    == replay_set
                        .ordered_quote_ledgers
                        .iter()
                        .filter(|ledger| !ledger.positions().is_empty())
                        .count()
                && values.iter().all(|value| value.is_finite() && *value > 0.0)
        }
    };
    if !valid_pip_values {
        return Err(outer_error(
            QuoteValidatedOuterHoldoutErrorCodeV1::InvalidMetricInput,
            "quote-validated metrics require a positive finite pip value per lot for every closed trade",
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
        execution
            .validate_against_quote_ledger(quote_ledger)
            .map_err(|error| {
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
        if execution.account_currency() != initial_balance.currency() {
            return Err(outer_error(
                QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
                format!(
                    "execution economics ledger {ordinal} currency differs from the run's starting balance"
                ),
            ));
        }
        let net_pnl_account_currency = execution.net_pnl_account_currency().amount();
        let pip_value_per_lot = match &pip_values {
            BoundPipValues::Fixed(value) => *value,
            BoundPipValues::PerClosedTrade(values) => values[trade_outcomes.len()],
        };
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
            account_currency: execution.account_currency().to_owned(),
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

    let metrics = derive_complete_quote_validated_metrics_v1(
        &initial_balance,
        &trade_outcomes,
        entry_unavailable,
    )?;
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
        search_config_hash: replay_set.search_config_hash,
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

#[cfg(test)]
#[path = "quote_validated_outer_holdout_metrics_tests.rs"]
mod metrics_tests;

#[cfg(test)]
mod streaming_identity_tests {
    use super::*;
    use serde::ser::SerializeSeq;

    #[test]
    fn streaming_hash_preserves_historical_compact_json_bytes_and_errors() {
        #[derive(Serialize)]
        struct Fixture<'a> {
            text: &'a str,
            values: [f64; 4],
            nested: Option<[u64; 2]>,
        }
        let value = Fixture {
            text: "Δοκιμή\n\"quoted\"\\path",
            values: [-0.0, 1.0 / 3.0, f64::MIN_POSITIVE, f64::MAX],
            nested: Some([0, u64::MAX]),
        };
        let mut old = Sha256::new();
        old.update(b"neoethos.locked-final-portfolio.v1\0");
        old.update(serde_json::to_vec(&value).unwrap());
        assert_eq!(
            canonical_locked_portfolio_identity_sha256_v1(&value).unwrap(),
            format!("{:x}", old.finalize())
        );
        struct Refused;
        impl Serialize for Refused {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("intentional encoding refusal"))
            }
        }
        let error = canonical_locked_portfolio_identity_sha256_v1(&Refused).unwrap_err();
        assert_eq!(
            error.code(),
            QuoteValidatedOuterHoldoutErrorCodeV1::ArtifactEncodingFailed
        );
        assert!(error.detail().contains("intentional encoding refusal"));
    }

    #[test]
    fn streaming_hash_accepts_a_lazy_large_sequence_without_an_expanded_json_buffer() {
        struct Repeated;
        impl Serialize for Repeated {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut sequence = serializer.serialize_seq(Some(100_000))?;
                for _ in 0..100_000 {
                    sequence.serialize_element(&7_u8)?;
                }
                sequence.end()
            }
        }
        let mut expected = Sha256::new();
        expected.update(b"neoethos.locked-final-portfolio.v1\0[7");
        for _ in 1..100_000 {
            expected.update(b",7");
        }
        expected.update(b"]");
        assert_eq!(
            canonical_locked_portfolio_identity_sha256_v1(&Repeated).unwrap(),
            format!("{:x}", expected.finalize())
        );
    }

    #[test]
    fn buffered_hash_preserves_full_and_partial_blocks_and_late_errors() {
        for length in [0, 1, 65_535, 65_536, 65_537, 131_073] {
            // Both many small writes and a single string larger than the buffer.
            let bytes = vec![255_u8; length];
            let text = "x".repeat(length);
            let value = (&bytes, &text, "end\n\"\\α");
            let mut expected = Sha256::new();
            expected.update(b"neoethos.locked-final-portfolio.v1\0");
            expected.update(serde_json::to_vec(&value).unwrap());
            assert_eq!(
                canonical_locked_portfolio_identity_sha256_v1(&value).unwrap(),
                format!("{:x}", expected.finalize()),
                "payload length {length}"
            );
        }
        struct LateRefusal;
        impl Serialize for LateRefusal {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut sequence = serializer.serialize_seq(None)?;
                for _ in 0..100_000 {
                    sequence.serialize_element(&255_u8)?;
                }
                Err(serde::ser::Error::custom("refused after buffered output"))
            }
        }
        let error = canonical_locked_portfolio_identity_sha256_v1(&LateRefusal).unwrap_err();
        assert_eq!(
            error.code(),
            QuoteValidatedOuterHoldoutErrorCodeV1::ArtifactEncodingFailed
        );
        assert!(error.detail().contains("refused after buffered output"));
    }
}

#[path = "quote_validated_outer_holdout_v2.rs"]
mod prelocked;
pub use prelocked::{
    CanonicalSignalAccountRiskPolicyV3, CanonicalSignalExitPolicyV2, LockedCanonicalSignalPlanV3,
    LockedPortfolioOuterHoldoutReplaySetV3, QuoteEntryFinancialInputsV3,
    QuoteEntrySizingEvidenceV3, QuoteReplayLotConstraintsV3, QuoteValidatedDecisionProvenanceV3,
    QuoteValidatedOuterHoldoutResearchEvidenceV3, evaluate_locked_portfolio_outer_holdout_v3,
};
