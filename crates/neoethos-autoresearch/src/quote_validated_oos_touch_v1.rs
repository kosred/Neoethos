use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use chrono::{Datelike, TimeZone, Utc};
use neoethos_search::{
    QuoteValidatedOuterHoldoutArtifactClassV1, QuoteValidatedOuterHoldoutPromotionEligibilityV1,
    QuoteValidatedOuterHoldoutResearchEvidenceV1,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::journal::OosWindow;
use crate::session::{DatasetReceiptV1, SessionId, SweepId};

pub const QUOTE_VALIDATED_OOS_TOUCH_SCHEMA_VERSION_V1: u16 = 1;

/// The upstream search evidence is accepted only after
/// `SealedHistoricalQuoteValidatedResearchLedgerV1`,
/// `QuoteValidatedExecutionEconomicsLedgerV1`, and every
/// `QuoteValidatedResearchReplayReceiptV1` were verified by the search-owned
/// locked-portfolio boundary. This module never reconstructs those leaf types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteValidatedOosArtifactClassV1 {
    ResearchOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteValidatedOosPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuoteValidatedOosTouchErrorCodeV1 {
    MissingSealedQuoteReplay,
    LegacyOhlcOosEvidenceInsufficient,
    LegacyForwardTestV2Insufficient,
    ReceiptSetMismatch,
    PortfolioBindingMismatch,
    WindowBindingMismatch,
    InvalidMetricInput,
    ArtifactEncodingFailed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuoteValidatedOosTouchErrorV1 {
    code: QuoteValidatedOosTouchErrorCodeV1,
    detail: String,
}

impl QuoteValidatedOosTouchErrorV1 {
    pub const fn code(&self) -> QuoteValidatedOosTouchErrorCodeV1 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for QuoteValidatedOosTouchErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl Error for QuoteValidatedOosTouchErrorV1 {}

fn oos_error(
    code: QuoteValidatedOosTouchErrorCodeV1,
    detail: impl Into<String>,
) -> QuoteValidatedOosTouchErrorV1 {
    QuoteValidatedOosTouchErrorV1 {
        code,
        detail: detail.into(),
    }
}

fn validate_sha256(label: &str, digest: &str) -> Result<(), QuoteValidatedOosTouchErrorV1> {
    if digest.len() != 64
        || !digest
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(oos_error(
            QuoteValidatedOosTouchErrorCodeV1::ReceiptSetMismatch,
            format!("{label} is not an exact SHA-256 digest"),
        ));
    }
    Ok(())
}

fn stable_sha256<T: Serialize>(
    domain: &str,
    value: &T,
) -> Result<String, QuoteValidatedOosTouchErrorV1> {
    let encoded = serde_json::to_vec(value).map_err(|error| {
        oos_error(
            QuoteValidatedOosTouchErrorCodeV1::ArtifactEncodingFailed,
            format!("cannot encode {domain}: {error}"),
        )
    })?;
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(encoded);
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedOosTouchReceiptV1 {
    schema_version: u16,
    session_id: String,
    sweep: SweepId,
    slot: usize,
    config_hash: String,
    dataset_receipt: DatasetReceiptV1,
    dataset_receipt_identity: String,
    oos_window: OosWindow,
    promotion_portfolio_sha256: String,
    canonical_search_input_receipt_sha256: String,
    effective_search_config_hash: String,
    holdout_scope_identity_sha256: String,
    outer_holdout_receipt_sha256: String,
    ordered_quote_ledger_sha256s: Vec<String>,
    ordered_historical_link_manifest_sha256s: Vec<String>,
    entry_unavailable: usize,
    artifact_class: QuoteValidatedOosArtifactClassV1,
    promotion_eligibility: QuoteValidatedOosPromotionEligibilityV1,
    receipt_sha256: String,
}

impl QuoteValidatedOosTouchReceiptV1 {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub const fn sweep(&self) -> SweepId {
        self.sweep
    }

    pub const fn slot(&self) -> usize {
        self.slot
    }

    pub fn config_hash(&self) -> &str {
        &self.config_hash
    }

    pub fn dataset_receipt(&self) -> &DatasetReceiptV1 {
        &self.dataset_receipt
    }

    pub const fn oos_window(&self) -> OosWindow {
        self.oos_window
    }

    pub fn promotion_portfolio_sha256(&self) -> &str {
        &self.promotion_portfolio_sha256
    }

    pub fn ordered_quote_ledger_sha256s(&self) -> &[String] {
        &self.ordered_quote_ledger_sha256s
    }

    pub fn ordered_historical_link_manifest_sha256s(&self) -> &[String] {
        &self.ordered_historical_link_manifest_sha256s
    }

    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }
}

#[derive(Serialize)]
struct QuoteValidatedOosReceiptHashPayloadV1<'a> {
    schema_version: u16,
    session_id: &'a str,
    sweep: SweepId,
    slot: usize,
    config_hash: &'a str,
    dataset_receipt: &'a DatasetReceiptV1,
    dataset_receipt_identity: &'a str,
    oos_window: OosWindow,
    promotion_portfolio_sha256: &'a str,
    canonical_search_input_receipt_sha256: &'a str,
    effective_search_config_hash: &'a str,
    holdout_scope_identity_sha256: &'a str,
    outer_holdout_receipt_sha256: &'a str,
    ordered_quote_ledger_sha256s: &'a [String],
    ordered_historical_link_manifest_sha256s: &'a [String],
    entry_unavailable: usize,
    artifact_class: QuoteValidatedOosArtifactClassV1,
    promotion_eligibility: QuoteValidatedOosPromotionEligibilityV1,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedOosTouchEvidenceV1 {
    receipt: QuoteValidatedOosTouchReceiptV1,
    per_trade_net_pips: Vec<f64>,
    r_multiples: Vec<f64>,
    monthly_returns: Vec<f64>,
    period_keys: Vec<i64>,
    trades_per_day: f64,
    band_survives: bool,
}

impl QuoteValidatedOosTouchEvidenceV1 {
    pub fn receipt(&self) -> &QuoteValidatedOosTouchReceiptV1 {
        &self.receipt
    }

    pub const fn window(&self) -> OosWindow {
        self.receipt.oos_window
    }

    pub fn per_trade_net_pips(&self) -> &[f64] {
        &self.per_trade_net_pips
    }

    pub fn r_multiples(&self) -> &[f64] {
        &self.r_multiples
    }

    pub fn monthly_returns(&self) -> &[f64] {
        &self.monthly_returns
    }

    pub fn period_keys(&self) -> &[i64] {
        &self.period_keys
    }

    pub const fn trades_per_day(&self) -> f64 {
        self.trades_per_day
    }

    /// V1 contains one explicit execution-economics policy, not a versioned
    /// multi-policy cost-band proof. The judge must therefore refuse the band
    /// conjunct rather than infer survival from a point estimate.
    pub const fn band_survives(&self) -> bool {
        self.band_survives
    }
}

struct DerivedQuoteValidatedOosStatisticsV1 {
    per_trade_net_pips: Vec<f64>,
    r_multiples: Vec<f64>,
    monthly_returns: Vec<f64>,
    period_keys: Vec<i64>,
    trades_per_day: f64,
    entry_unavailable: usize,
}

fn derive_complete_oos_statistics_from_quote_ledgers_v1(
    evidence: &QuoteValidatedOuterHoldoutResearchEvidenceV1,
    oos_window: OosWindow,
    initial_balance: f64,
) -> Result<DerivedQuoteValidatedOosStatisticsV1, QuoteValidatedOosTouchErrorV1> {
    if !initial_balance.is_finite()
        || initial_balance <= 0.0
        || oos_window.start_ms > oos_window.end_ms
    {
        return Err(oos_error(
            QuoteValidatedOosTouchErrorCodeV1::InvalidMetricInput,
            "quote-validated OOS statistics require a valid window and positive finite balance",
        ));
    }
    let mut per_trade_net_pips = Vec::with_capacity(evidence.trade_outcomes().len());
    let mut r_multiples = Vec::with_capacity(evidence.trade_outcomes().len());
    let mut by_month = BTreeMap::<i64, f64>::new();
    for outcome in evidence.trade_outcomes() {
        let net_pnl_account_currency = outcome.net_pnl_account_currency();
        if !net_pnl_account_currency.is_finite()
            || !outcome.net_pips().is_finite()
            || !outcome.r_multiple().is_finite()
        {
            return Err(oos_error(
                QuoteValidatedOosTouchErrorCodeV1::InvalidMetricInput,
                "sealed quote/economics outcome contains a non-finite metric",
            ));
        }
        let timestamp = Utc
            .timestamp_millis_opt(outcome.exit_timestamp_unix_ms())
            .single()
            .ok_or_else(|| {
                oos_error(
                    QuoteValidatedOosTouchErrorCodeV1::InvalidMetricInput,
                    "sealed quote outcome has an invalid exit timestamp",
                )
            })?;
        let period = i64::from(timestamp.year()) * 100 + i64::from(timestamp.month());
        *by_month.entry(period).or_insert(0.0) += net_pnl_account_currency;
        per_trade_net_pips.push(outcome.net_pips());
        r_multiples.push(outcome.r_multiple());
    }
    let period_keys = by_month.keys().copied().collect::<Vec<_>>();
    let monthly_returns = by_month
        .values()
        .map(|net_pnl_account_currency| net_pnl_account_currency / initial_balance)
        .collect::<Vec<_>>();
    let days = ((oos_window.end_ms - oos_window.start_ms).max(1) as f64 / 86_400_000.0).max(1.0);
    Ok(DerivedQuoteValidatedOosStatisticsV1 {
        trades_per_day: per_trade_net_pips.len() as f64 / days,
        per_trade_net_pips,
        r_multiples,
        monthly_returns,
        period_keys,
        entry_unavailable: evidence.metrics().entry_unavailable(),
    })
}

pub fn evaluate_quote_validated_oos_touch_v1(
    session_id: &SessionId,
    sweep: SweepId,
    slot: usize,
    config_hash: &str,
    dataset_receipt: &DatasetReceiptV1,
    oos_window: OosWindow,
    promotion_portfolio_sha256: &str,
    expected_canonical_search_input_receipt_sha256: &str,
    expected_effective_search_config_hash: &str,
    expected_holdout_scope_identity_sha256: &str,
    evidence: QuoteValidatedOuterHoldoutResearchEvidenceV1,
    initial_balance: f64,
) -> Result<QuoteValidatedOosTouchEvidenceV1, QuoteValidatedOosTouchErrorV1> {
    if evidence.receipt().quote_replay_receipts().is_empty() {
        return Err(oos_error(
            QuoteValidatedOosTouchErrorCodeV1::MissingSealedQuoteReplay,
            "the final OOS touch has no sealed historical quote-replay receipt",
        ));
    }
    if dataset_receipt.oos_window != oos_window {
        return Err(oos_error(
            QuoteValidatedOosTouchErrorCodeV1::WindowBindingMismatch,
            "the requested OOS window differs from the immutable session dataset receipt",
        ));
    }
    for (label, digest) in [
        ("promotion portfolio", promotion_portfolio_sha256),
        (
            "canonical search input receipt",
            expected_canonical_search_input_receipt_sha256,
        ),
        ("holdout scope", expected_holdout_scope_identity_sha256),
        ("outer holdout receipt", evidence.receipt().receipt_sha256()),
    ] {
        validate_sha256(label, digest)?;
    }
    if config_hash.trim().is_empty() || expected_effective_search_config_hash.trim().is_empty() {
        return Err(oos_error(
            QuoteValidatedOosTouchErrorCodeV1::ReceiptSetMismatch,
            "the finalist proposal or effective search configuration has no identity",
        ));
    }
    if evidence.receipt().portfolio_identity_sha256() != promotion_portfolio_sha256 {
        return Err(oos_error(
            QuoteValidatedOosTouchErrorCodeV1::PortfolioBindingMismatch,
            "sealed quote replay belongs to a different immutable PromotionPortfolio",
        ));
    }
    if evidence.receipt().canonical_search_input_receipt_sha256()
        != expected_canonical_search_input_receipt_sha256
        || evidence.receipt().search_config_hash() != expected_effective_search_config_hash
        || evidence.receipt().holdout_scope_identity_sha256()
            != expected_holdout_scope_identity_sha256
    {
        return Err(oos_error(
            QuoteValidatedOosTouchErrorCodeV1::ReceiptSetMismatch,
            "sealed replay receipt differs from the exact OOS feature/scope/config binding",
        ));
    }
    if evidence.receipt().artifact_class()
        != QuoteValidatedOuterHoldoutArtifactClassV1::ResearchOnly
        || evidence.receipt().promotion_eligibility()
            != QuoteValidatedOuterHoldoutPromotionEligibilityV1::NotPromotionEligible
    {
        return Err(oos_error(
            QuoteValidatedOosTouchErrorCodeV1::ReceiptSetMismatch,
            "outer-holdout evidence is not research-only and fail-closed for promotion",
        ));
    }
    let locked_window = evidence.receipt().locked_evaluation_window();
    if locked_window.from_unix_ms_inclusive() < oos_window.start_ms
        || locked_window.from_unix_ms_inclusive() > oos_window.end_ms
        || locked_window.to_unix_ms_exclusive() <= oos_window.end_ms
    {
        return Err(oos_error(
            QuoteValidatedOosTouchErrorCodeV1::WindowBindingMismatch,
            "sealed quote replay does not cover the exact half-open finalist OOS boundary",
        ));
    }

    let ordered_quote_ledger_sha256s = evidence
        .receipt()
        .quote_replay_receipts()
        .iter()
        .map(|receipt| receipt.ledger_sha256().to_owned())
        .collect::<Vec<_>>();
    let ordered_historical_link_manifest_sha256s = evidence
        .receipt()
        .ordered_historical_link_manifest_sha256s()
        .to_vec();
    if ordered_quote_ledger_sha256s.len() != ordered_historical_link_manifest_sha256s.len() {
        return Err(oos_error(
            QuoteValidatedOosTouchErrorCodeV1::ReceiptSetMismatch,
            "quote-ledger and immutable acquisition-link receipt sets differ",
        ));
    }
    let statistics = derive_complete_oos_statistics_from_quote_ledgers_v1(
        &evidence,
        oos_window,
        initial_balance,
    )?;
    let mut receipt = QuoteValidatedOosTouchReceiptV1 {
        schema_version: QUOTE_VALIDATED_OOS_TOUCH_SCHEMA_VERSION_V1,
        session_id: session_id.to_string(),
        sweep,
        slot,
        config_hash: config_hash.to_owned(),
        dataset_receipt: dataset_receipt.clone(),
        dataset_receipt_identity: dataset_receipt.identity().to_string(),
        oos_window,
        promotion_portfolio_sha256: promotion_portfolio_sha256.to_owned(),
        canonical_search_input_receipt_sha256: expected_canonical_search_input_receipt_sha256
            .to_owned(),
        effective_search_config_hash: expected_effective_search_config_hash.to_owned(),
        holdout_scope_identity_sha256: expected_holdout_scope_identity_sha256.to_owned(),
        outer_holdout_receipt_sha256: evidence.receipt().receipt_sha256().to_owned(),
        ordered_quote_ledger_sha256s,
        ordered_historical_link_manifest_sha256s,
        entry_unavailable: statistics.entry_unavailable,
        artifact_class: QuoteValidatedOosArtifactClassV1::ResearchOnly,
        promotion_eligibility: QuoteValidatedOosPromotionEligibilityV1::NotPromotionEligible,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = stable_sha256(
        "neoethos.quote-validated-oos-touch-receipt.v1",
        &QuoteValidatedOosReceiptHashPayloadV1 {
            schema_version: receipt.schema_version,
            session_id: &receipt.session_id,
            sweep: receipt.sweep,
            slot: receipt.slot,
            config_hash: &receipt.config_hash,
            dataset_receipt: &receipt.dataset_receipt,
            dataset_receipt_identity: &receipt.dataset_receipt_identity,
            oos_window: receipt.oos_window,
            promotion_portfolio_sha256: &receipt.promotion_portfolio_sha256,
            canonical_search_input_receipt_sha256: &receipt.canonical_search_input_receipt_sha256,
            effective_search_config_hash: &receipt.effective_search_config_hash,
            holdout_scope_identity_sha256: &receipt.holdout_scope_identity_sha256,
            outer_holdout_receipt_sha256: &receipt.outer_holdout_receipt_sha256,
            ordered_quote_ledger_sha256s: &receipt.ordered_quote_ledger_sha256s,
            ordered_historical_link_manifest_sha256s: &receipt
                .ordered_historical_link_manifest_sha256s,
            entry_unavailable: receipt.entry_unavailable,
            artifact_class: receipt.artifact_class,
            promotion_eligibility: receipt.promotion_eligibility,
        },
    )?;
    Ok(QuoteValidatedOosTouchEvidenceV1 {
        receipt,
        per_trade_net_pips: statistics.per_trade_net_pips,
        r_multiples: statistics.r_multiples,
        monthly_returns: statistics.monthly_returns,
        period_keys: statistics.period_keys,
        trades_per_day: statistics.trades_per_day,
        band_survives: false,
    })
}
