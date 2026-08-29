use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use crate::contracts::{sha256_bytes, validate_sha256_hex};
use crate::execution_replay_v1::{
    QuoteValidatedResearchAuthorityV1, ResearchPositionDirectionV1,
    SealedHistoricalQuoteValidatedResearchLedgerV1,
};

pub const EXECUTION_ECONOMICS_SCHEMA_VERSION_V1: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionEconomicsErrorCodeV1 {
    InvalidContract,
    InvalidCurrency,
    InvalidSha256,
    InvalidEncoding,
    InvalidMoney,
    InvalidFilledLots,
    InvalidCommissionPolicy,
    InvalidConversionEvidence,
    InvalidSwapEvidence,
    InvalidConversionFeeEvidence,
    MissingPosition,
    MissingExitFill,
    MissingConversionEvidence,
    StaleConversionEvidence,
    CurrencyMismatch,
    SymbolMismatch,
    AuthorityMismatch,
    ArtifactDigestMismatch,
    UnsupportedSchemaVersion,
    LegacyWireRefused,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionEconomicsErrorV1 {
    code: ExecutionEconomicsErrorCodeV1,
    detail: String,
}

impl ExecutionEconomicsErrorV1 {
    fn new(code: ExecutionEconomicsErrorCodeV1, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> ExecutionEconomicsErrorCodeV1 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for ExecutionEconomicsErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "execution economics V1: {}", self.detail)
    }
}

impl Error for ExecutionEconomicsErrorV1 {}

pub type ExecutionEconomicsResultV1 =
    Result<QuoteValidatedExecutionEconomicsLedgerV1, ExecutionEconomicsErrorV1>;

fn economics_error(
    code: ExecutionEconomicsErrorCodeV1,
    detail: impl Into<String>,
) -> ExecutionEconomicsErrorV1 {
    ExecutionEconomicsErrorV1::new(code, detail)
}

fn validate_digest(label: &str, digest: &str) -> Result<(), ExecutionEconomicsErrorV1> {
    validate_sha256_hex(label, digest).map_err(|error| {
        economics_error(
            ExecutionEconomicsErrorCodeV1::InvalidSha256,
            error.to_string(),
        )
    })
}

fn hash_json<T: Serialize>(label: &str, value: &T) -> Result<String, ExecutionEconomicsErrorV1> {
    let encoded = serde_json::to_vec(value).map_err(|error| {
        economics_error(
            ExecutionEconomicsErrorCodeV1::InvalidEncoding,
            format!("cannot encode {label}: {error}"),
        )
    })?;
    let mut domain_separated = format!("neoethos-{label}-v1\n").into_bytes();
    domain_separated.extend_from_slice(&encoded);
    Ok(sha256_bytes(&domain_separated))
}

fn validate_currency(label: &str, currency: &str) -> Result<(), ExecutionEconomicsErrorV1> {
    if currency.len() != 3 || !currency.bytes().all(|byte| byte.is_ascii_uppercase()) {
        return Err(economics_error(
            ExecutionEconomicsErrorCodeV1::InvalidCurrency,
            format!("{label} must be exactly three uppercase ASCII letters"),
        ));
    }
    Ok(())
}

fn validate_positive_finite(
    label: &str,
    value: f64,
    code: ExecutionEconomicsErrorCodeV1,
) -> Result<(), ExecutionEconomicsErrorV1> {
    if !value.is_finite() || value <= 0.0 {
        return Err(economics_error(
            code,
            format!("{label} must be finite and strictly positive"),
        ));
    }
    Ok(())
}

fn validate_nonnegative_finite(
    label: &str,
    value: f64,
    code: ExecutionEconomicsErrorCodeV1,
) -> Result<(), ExecutionEconomicsErrorV1> {
    if !value.is_finite() || value < 0.0 {
        return Err(economics_error(
            code,
            format!("{label} must be finite and non-negative"),
        ));
    }
    Ok(())
}

fn require_same_bits(
    label: &str,
    actual: f64,
    expected: f64,
) -> Result<(), ExecutionEconomicsErrorV1> {
    if actual.to_bits() != expected.to_bits() {
        return Err(economics_error(
            ExecutionEconomicsErrorCodeV1::ArtifactDigestMismatch,
            format!("{label} differs from the deterministic V1 calculation"),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccountMoneyV1 {
    currency: String,
    amount: f64,
}

impl AccountMoneyV1 {
    pub fn new(
        currency: impl Into<String>,
        amount: f64,
    ) -> Result<Self, ExecutionEconomicsErrorV1> {
        let money = Self {
            currency: currency.into(),
            amount,
        };
        money.validate()?;
        Ok(money)
    }

    pub fn zero(currency: &str) -> Result<Self, ExecutionEconomicsErrorV1> {
        Self::new(currency, 0.0)
    }

    pub fn currency(&self) -> &str {
        &self.currency
    }

    pub const fn amount(&self) -> f64 {
        self.amount
    }

    fn validate(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        validate_currency("money currency", &self.currency)?;
        if !self.amount.is_finite() {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::InvalidMoney,
                "account-money amount must be finite",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionSymbolContractV1 {
    symbol_name: String,
    base_currency: String,
    quote_currency: String,
    contract_units_per_lot: f64,
    symbol_contract_identity_sha256: String,
}

#[derive(Serialize)]
struct SymbolContractHashPayloadV1<'a> {
    symbol_name: &'a str,
    base_currency: &'a str,
    quote_currency: &'a str,
    contract_units_per_lot: f64,
}

impl ExecutionSymbolContractV1 {
    pub fn new(
        symbol_name: impl Into<String>,
        base_currency: impl Into<String>,
        quote_currency: impl Into<String>,
        contract_units_per_lot: f64,
    ) -> Result<Self, ExecutionEconomicsErrorV1> {
        let mut contract = Self {
            symbol_name: symbol_name.into(),
            base_currency: base_currency.into(),
            quote_currency: quote_currency.into(),
            contract_units_per_lot,
            symbol_contract_identity_sha256: String::new(),
        };
        contract.validate_payload()?;
        contract.symbol_contract_identity_sha256 = contract.recomputed_identity_sha256()?;
        Ok(contract)
    }

    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }

    pub fn base_currency(&self) -> &str {
        &self.base_currency
    }

    pub fn quote_currency(&self) -> &str {
        &self.quote_currency
    }

    pub const fn contract_units_per_lot(&self) -> f64 {
        self.contract_units_per_lot
    }

    pub fn identity_sha256(&self) -> &str {
        &self.symbol_contract_identity_sha256
    }

    fn hash_payload(&self) -> SymbolContractHashPayloadV1<'_> {
        SymbolContractHashPayloadV1 {
            symbol_name: &self.symbol_name,
            base_currency: &self.base_currency,
            quote_currency: &self.quote_currency,
            contract_units_per_lot: self.contract_units_per_lot,
        }
    }

    fn recomputed_identity_sha256(&self) -> Result<String, ExecutionEconomicsErrorV1> {
        hash_json("execution-symbol-contract", &self.hash_payload())
    }

    fn validate_payload(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        if self.symbol_name.trim().is_empty() || self.symbol_name.trim() != self.symbol_name {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::InvalidContract,
                "symbol name must be non-empty and already trimmed",
            ));
        }
        validate_currency("base currency", &self.base_currency)?;
        validate_currency("quote currency", &self.quote_currency)?;
        validate_positive_finite(
            "contract units per lot",
            self.contract_units_per_lot,
            ExecutionEconomicsErrorCodeV1::InvalidContract,
        )
    }

    fn validate(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        self.validate_payload()?;
        validate_digest(
            "symbol contract identity",
            &self.symbol_contract_identity_sha256,
        )?;
        if self.symbol_contract_identity_sha256 != self.recomputed_identity_sha256()? {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::ArtifactDigestMismatch,
                "symbol contract identity does not match its exact payload",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalQuoteToAccountConversionV1 {
    source_currency: String,
    target_account_currency: String,
    conversion_rate_account_per_quote: f64,
    conversion_observed_at_unix_ms: i64,
    conversion_max_age_ms: i64,
    conversion_source_artifact_sha256: String,
    conversion_evidence_identity_sha256: String,
}

#[derive(Serialize)]
struct ConversionHashPayloadV1<'a> {
    source_currency: &'a str,
    target_account_currency: &'a str,
    conversion_rate_account_per_quote: f64,
    conversion_observed_at_unix_ms: i64,
    conversion_max_age_ms: i64,
    conversion_source_artifact_sha256: &'a str,
}

impl CausalQuoteToAccountConversionV1 {
    pub fn new(
        source_currency: impl Into<String>,
        target_account_currency: impl Into<String>,
        conversion_rate_account_per_quote: f64,
        conversion_observed_at_unix_ms: i64,
        conversion_max_age_ms: i64,
        conversion_source_artifact_sha256: impl Into<String>,
    ) -> Result<Self, ExecutionEconomicsErrorV1> {
        let mut conversion = Self {
            source_currency: source_currency.into(),
            target_account_currency: target_account_currency.into(),
            conversion_rate_account_per_quote,
            conversion_observed_at_unix_ms,
            conversion_max_age_ms,
            conversion_source_artifact_sha256: conversion_source_artifact_sha256.into(),
            conversion_evidence_identity_sha256: String::new(),
        };
        conversion.validate_payload()?;
        conversion.conversion_evidence_identity_sha256 = conversion.recomputed_identity_sha256()?;
        Ok(conversion)
    }

    pub fn source_currency(&self) -> &str {
        &self.source_currency
    }

    pub fn target_account_currency(&self) -> &str {
        &self.target_account_currency
    }

    pub const fn conversion_rate_account_per_quote(&self) -> f64 {
        self.conversion_rate_account_per_quote
    }

    pub const fn conversion_observed_at_unix_ms(&self) -> i64 {
        self.conversion_observed_at_unix_ms
    }

    pub fn identity_sha256(&self) -> &str {
        &self.conversion_evidence_identity_sha256
    }

    fn hash_payload(&self) -> ConversionHashPayloadV1<'_> {
        ConversionHashPayloadV1 {
            source_currency: &self.source_currency,
            target_account_currency: &self.target_account_currency,
            conversion_rate_account_per_quote: self.conversion_rate_account_per_quote,
            conversion_observed_at_unix_ms: self.conversion_observed_at_unix_ms,
            conversion_max_age_ms: self.conversion_max_age_ms,
            conversion_source_artifact_sha256: &self.conversion_source_artifact_sha256,
        }
    }

    fn recomputed_identity_sha256(&self) -> Result<String, ExecutionEconomicsErrorV1> {
        hash_json("quote-to-account-conversion", &self.hash_payload())
    }

    fn validate_payload(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        validate_currency("conversion source currency", &self.source_currency)?;
        validate_currency(
            "conversion target account currency",
            &self.target_account_currency,
        )?;
        validate_positive_finite(
            "conversion rate account per quote",
            self.conversion_rate_account_per_quote,
            ExecutionEconomicsErrorCodeV1::InvalidConversionEvidence,
        )?;
        if self.conversion_observed_at_unix_ms <= 0 || self.conversion_max_age_ms < 0 {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::InvalidConversionEvidence,
                "conversion timestamp must be positive and maximum age must be non-negative",
            ));
        }
        if self.source_currency == self.target_account_currency
            && self.conversion_rate_account_per_quote.to_bits() != 1.0_f64.to_bits()
        {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::CurrencyMismatch,
                "identity currency conversion must use an exact 1.0 rate",
            ));
        }
        validate_digest(
            "conversion source artifact",
            &self.conversion_source_artifact_sha256,
        )
    }

    fn validate(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        self.validate_payload()?;
        validate_digest(
            "conversion evidence identity",
            &self.conversion_evidence_identity_sha256,
        )?;
        if self.conversion_evidence_identity_sha256 != self.recomputed_identity_sha256()? {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::ArtifactDigestMismatch,
                "conversion evidence identity does not match its exact payload",
            ));
        }
        Ok(())
    }

    fn validate_causal_for(&self, fill_unix_ms: i64) -> Result<(), ExecutionEconomicsErrorV1> {
        let Some(age_ms) = fill_unix_ms.checked_sub(self.conversion_observed_at_unix_ms) else {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::StaleConversionEvidence,
                "conversion age overflows the signed millisecond range",
            ));
        };
        if age_ms < 0 || age_ms > self.conversion_max_age_ms {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::StaleConversionEvidence,
                "conversion evidence is future-dated or older than its explicit maximum age",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCommissionPolicyV1 {
    account_currency: String,
    commission_account_per_lot_per_fill: f64,
    commission_source_artifact_sha256: String,
    commission_policy_identity_sha256: String,
}

#[derive(Serialize)]
struct CommissionPolicyHashPayloadV1<'a> {
    account_currency: &'a str,
    commission_account_per_lot_per_fill: f64,
    commission_source_artifact_sha256: &'a str,
}

impl ExecutionCommissionPolicyV1 {
    pub fn new(
        account_currency: impl Into<String>,
        commission_account_per_lot_per_fill: f64,
        commission_source_artifact_sha256: impl Into<String>,
    ) -> Result<Self, ExecutionEconomicsErrorV1> {
        let mut policy = Self {
            account_currency: account_currency.into(),
            commission_account_per_lot_per_fill,
            commission_source_artifact_sha256: commission_source_artifact_sha256.into(),
            commission_policy_identity_sha256: String::new(),
        };
        policy.validate_payload()?;
        policy.commission_policy_identity_sha256 = policy.recomputed_identity_sha256()?;
        Ok(policy)
    }

    pub fn account_currency(&self) -> &str {
        &self.account_currency
    }

    pub const fn commission_account_per_lot_per_fill(&self) -> f64 {
        self.commission_account_per_lot_per_fill
    }

    pub fn identity_sha256(&self) -> &str {
        &self.commission_policy_identity_sha256
    }

    fn hash_payload(&self) -> CommissionPolicyHashPayloadV1<'_> {
        CommissionPolicyHashPayloadV1 {
            account_currency: &self.account_currency,
            commission_account_per_lot_per_fill: self.commission_account_per_lot_per_fill,
            commission_source_artifact_sha256: &self.commission_source_artifact_sha256,
        }
    }

    fn recomputed_identity_sha256(&self) -> Result<String, ExecutionEconomicsErrorV1> {
        hash_json("execution-commission-policy", &self.hash_payload())
    }

    fn validate_payload(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        validate_currency("commission account currency", &self.account_currency)?;
        validate_nonnegative_finite(
            "commission account per lot per fill",
            self.commission_account_per_lot_per_fill,
            ExecutionEconomicsErrorCodeV1::InvalidCommissionPolicy,
        )?;
        validate_digest(
            "commission source artifact",
            &self.commission_source_artifact_sha256,
        )
    }

    fn validate(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        self.validate_payload()?;
        validate_digest(
            "commission policy identity",
            &self.commission_policy_identity_sha256,
        )?;
        if self.commission_policy_identity_sha256 != self.recomputed_identity_sha256()? {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::ArtifactDigestMismatch,
                "commission policy identity does not match its exact payload",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedSwapCashflowV1 {
    account_currency: String,
    swap_account_currency_signed: f64,
    swap_observed_at_unix_ms: i64,
    swap_source_artifact_sha256: String,
    swap_evidence_identity_sha256: String,
}

#[derive(Serialize)]
struct SwapEvidenceHashPayloadV1<'a> {
    account_currency: &'a str,
    swap_account_currency_signed: f64,
    swap_observed_at_unix_ms: i64,
    swap_source_artifact_sha256: &'a str,
}

impl SignedSwapCashflowV1 {
    pub fn new(
        account_currency: impl Into<String>,
        swap_account_currency_signed: f64,
        swap_observed_at_unix_ms: i64,
        swap_source_artifact_sha256: impl Into<String>,
    ) -> Result<Self, ExecutionEconomicsErrorV1> {
        let mut cashflow = Self {
            account_currency: account_currency.into(),
            swap_account_currency_signed,
            swap_observed_at_unix_ms,
            swap_source_artifact_sha256: swap_source_artifact_sha256.into(),
            swap_evidence_identity_sha256: String::new(),
        };
        cashflow.validate_payload()?;
        cashflow.swap_evidence_identity_sha256 = cashflow.recomputed_identity_sha256()?;
        Ok(cashflow)
    }

    pub fn account_currency(&self) -> &str {
        &self.account_currency
    }

    pub const fn amount_signed(&self) -> f64 {
        self.swap_account_currency_signed
    }

    pub fn identity_sha256(&self) -> &str {
        &self.swap_evidence_identity_sha256
    }

    fn hash_payload(&self) -> SwapEvidenceHashPayloadV1<'_> {
        SwapEvidenceHashPayloadV1 {
            account_currency: &self.account_currency,
            swap_account_currency_signed: self.swap_account_currency_signed,
            swap_observed_at_unix_ms: self.swap_observed_at_unix_ms,
            swap_source_artifact_sha256: &self.swap_source_artifact_sha256,
        }
    }

    fn recomputed_identity_sha256(&self) -> Result<String, ExecutionEconomicsErrorV1> {
        hash_json("execution-swap-evidence", &self.hash_payload())
    }

    fn validate_payload(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        validate_currency("swap account currency", &self.account_currency)?;
        if !self.swap_account_currency_signed.is_finite() || self.swap_observed_at_unix_ms <= 0 {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::InvalidSwapEvidence,
                "swap cashflow must be finite and its evidence timestamp must be positive",
            ));
        }
        validate_digest("swap source artifact", &self.swap_source_artifact_sha256)
    }

    fn validate(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        self.validate_payload()?;
        validate_digest(
            "swap evidence identity",
            &self.swap_evidence_identity_sha256,
        )?;
        if self.swap_evidence_identity_sha256 != self.recomputed_identity_sha256()? {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::ArtifactDigestMismatch,
                "swap evidence identity does not match its exact payload",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PnlConversionFeeV1 {
    account_currency: String,
    pnl_conversion_fee_account_currency: f64,
    pnl_conversion_fee_source_artifact_sha256: String,
    pnl_conversion_fee_evidence_identity_sha256: String,
}

#[derive(Serialize)]
struct ConversionFeeHashPayloadV1<'a> {
    account_currency: &'a str,
    pnl_conversion_fee_account_currency: f64,
    pnl_conversion_fee_source_artifact_sha256: &'a str,
}

impl PnlConversionFeeV1 {
    pub fn new(
        account_currency: impl Into<String>,
        pnl_conversion_fee_account_currency: f64,
        pnl_conversion_fee_source_artifact_sha256: impl Into<String>,
    ) -> Result<Self, ExecutionEconomicsErrorV1> {
        let mut fee = Self {
            account_currency: account_currency.into(),
            pnl_conversion_fee_account_currency,
            pnl_conversion_fee_source_artifact_sha256: pnl_conversion_fee_source_artifact_sha256
                .into(),
            pnl_conversion_fee_evidence_identity_sha256: String::new(),
        };
        fee.validate_payload()?;
        fee.pnl_conversion_fee_evidence_identity_sha256 = fee.recomputed_identity_sha256()?;
        Ok(fee)
    }

    pub fn account_currency(&self) -> &str {
        &self.account_currency
    }

    pub const fn amount(&self) -> f64 {
        self.pnl_conversion_fee_account_currency
    }

    pub fn identity_sha256(&self) -> &str {
        &self.pnl_conversion_fee_evidence_identity_sha256
    }

    fn hash_payload(&self) -> ConversionFeeHashPayloadV1<'_> {
        ConversionFeeHashPayloadV1 {
            account_currency: &self.account_currency,
            pnl_conversion_fee_account_currency: self.pnl_conversion_fee_account_currency,
            pnl_conversion_fee_source_artifact_sha256: &self
                .pnl_conversion_fee_source_artifact_sha256,
        }
    }

    fn recomputed_identity_sha256(&self) -> Result<String, ExecutionEconomicsErrorV1> {
        hash_json("execution-pnl-conversion-fee", &self.hash_payload())
    }

    fn validate_payload(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        validate_currency(
            "PnL conversion-fee account currency",
            &self.account_currency,
        )?;
        validate_nonnegative_finite(
            "PnL conversion fee account currency",
            self.pnl_conversion_fee_account_currency,
            ExecutionEconomicsErrorCodeV1::InvalidConversionFeeEvidence,
        )?;
        validate_digest(
            "PnL conversion-fee source artifact",
            &self.pnl_conversion_fee_source_artifact_sha256,
        )
    }

    fn validate(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        self.validate_payload()?;
        validate_digest(
            "PnL conversion-fee evidence identity",
            &self.pnl_conversion_fee_evidence_identity_sha256,
        )?;
        if self.pnl_conversion_fee_evidence_identity_sha256 != self.recomputed_identity_sha256()? {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::ArtifactDigestMismatch,
                "PnL conversion-fee identity does not match its exact payload",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionEconomicsArtifactClassV1 {
    ResearchOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionEconomicsPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuoteValidatedExecutionEconomicsLedgerV1 {
    schema_version: u32,
    quote_ledger_sha256: String,
    quote_position_index: u64,
    symbol_contract: ExecutionSymbolContractV1,
    account_currency: String,
    filled_lots: f64,
    base_units: f64,
    direction: ResearchPositionDirectionV1,
    entry_fill_timestamp_unix_ms: i64,
    exit_fill_timestamp_unix_ms: i64,
    modeled_entry_price: f64,
    modeled_exit_price: f64,
    entry_notional_quote_currency: f64,
    gross_pnl_quote_currency: f64,
    conversion: CausalQuoteToAccountConversionV1,
    gross_pnl_account_currency: AccountMoneyV1,
    commission_policy: ExecutionCommissionPolicyV1,
    entry_commission_account_currency: AccountMoneyV1,
    exit_commission_account_currency: AccountMoneyV1,
    swap: SignedSwapCashflowV1,
    swap_account_currency_signed: AccountMoneyV1,
    pnl_conversion_fee: PnlConversionFeeV1,
    pnl_conversion_fee_account_currency: AccountMoneyV1,
    additional_spread_account_currency: AccountMoneyV1,
    net_pnl_account_currency: AccountMoneyV1,
    entry_fill_identity_sha256: String,
    exit_fill_identity_sha256: String,
    artifact_class: ExecutionEconomicsArtifactClassV1,
    promotion_eligibility: ExecutionEconomicsPromotionEligibilityV1,
    ledger_sha256: String,
}

#[derive(Serialize)]
struct DerivedFillIdentityHashPayloadV1<'a> {
    quote_ledger_sha256: &'a str,
    quote_position_index: u64,
    fill_role: &'a str,
    fill_timestamp_unix_ms: i64,
    modeled_fill_price: f64,
    direction: ResearchPositionDirectionV1,
}

#[derive(Serialize)]
struct ExecutionEconomicsHashPayloadV1<'a> {
    schema_version: u32,
    quote_ledger_sha256: &'a str,
    quote_position_index: u64,
    symbol_contract_identity_sha256: &'a str,
    account_currency: &'a str,
    conversion_evidence_identity_sha256: &'a str,
    entry_fill_identity_sha256: &'a str,
    exit_fill_identity_sha256: &'a str,
    commission_policy_identity_sha256: &'a str,
    swap_evidence_identity_sha256: &'a str,
    pnl_conversion_fee_evidence_identity_sha256: &'a str,
    filled_lots: f64,
    base_units: f64,
    direction: ResearchPositionDirectionV1,
    entry_fill_timestamp_unix_ms: i64,
    exit_fill_timestamp_unix_ms: i64,
    modeled_entry_price: f64,
    modeled_exit_price: f64,
    entry_notional_quote_currency: f64,
    gross_pnl_quote_currency: f64,
    conversion_rate_account_per_quote: f64,
    conversion_observed_at_unix_ms: i64,
    gross_pnl_account_currency: &'a AccountMoneyV1,
    entry_commission_account_currency: &'a AccountMoneyV1,
    exit_commission_account_currency: &'a AccountMoneyV1,
    swap_account_currency_signed: &'a AccountMoneyV1,
    pnl_conversion_fee_account_currency: &'a AccountMoneyV1,
    additional_spread_account_currency: &'a AccountMoneyV1,
    net_pnl_account_currency: &'a AccountMoneyV1,
    artifact_class: ExecutionEconomicsArtifactClassV1,
    promotion_eligibility: ExecutionEconomicsPromotionEligibilityV1,
}

impl QuoteValidatedExecutionEconomicsLedgerV1 {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ExecutionEconomicsErrorV1> {
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
            economics_error(
                ExecutionEconomicsErrorCodeV1::InvalidEncoding,
                format!("cannot decode execution economics V1 JSON: {error}"),
            )
        })?;
        let Some(object) = value.as_object() else {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::LegacyWireRefused,
                "execution economics artifact must be a versioned JSON object",
            ));
        };
        for required in [
            "schema_version",
            "quote_ledger_sha256",
            "symbol_contract",
            "conversion",
            "entry_fill_identity_sha256",
            "exit_fill_identity_sha256",
            "ledger_sha256",
        ] {
            if !object.contains_key(required) {
                return Err(economics_error(
                    ExecutionEconomicsErrorCodeV1::LegacyWireRefused,
                    format!("legacy execution wire is missing required V1 field {required}"),
                ));
            }
        }
        let schema_version = object
            .get("schema_version")
            .and_then(serde_json::Value::as_u64);
        if schema_version != Some(u64::from(EXECUTION_ECONOMICS_SCHEMA_VERSION_V1)) {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::UnsupportedSchemaVersion,
                "execution economics schema version is not V1",
            ));
        }
        let artifact: Self = serde_json::from_value(value).map_err(|error| {
            economics_error(
                ExecutionEconomicsErrorCodeV1::InvalidEncoding,
                format!("cannot decode exact execution economics V1 fields: {error}"),
            )
        })?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, ExecutionEconomicsErrorV1> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            economics_error(
                ExecutionEconomicsErrorCodeV1::InvalidEncoding,
                format!("cannot encode execution economics V1: {error}"),
            )
        })
    }

    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn quote_ledger_sha256(&self) -> &str {
        &self.quote_ledger_sha256
    }

    pub const fn quote_position_index(&self) -> u64 {
        self.quote_position_index
    }

    pub fn symbol_contract(&self) -> &ExecutionSymbolContractV1 {
        &self.symbol_contract
    }

    pub fn account_currency(&self) -> &str {
        &self.account_currency
    }

    pub const fn filled_lots(&self) -> f64 {
        self.filled_lots
    }

    pub const fn base_units(&self) -> f64 {
        self.base_units
    }

    pub const fn direction(&self) -> ResearchPositionDirectionV1 {
        self.direction
    }

    pub const fn entry_fill_timestamp_unix_ms(&self) -> i64 {
        self.entry_fill_timestamp_unix_ms
    }

    pub const fn exit_fill_timestamp_unix_ms(&self) -> i64 {
        self.exit_fill_timestamp_unix_ms
    }

    pub const fn modeled_entry_price(&self) -> f64 {
        self.modeled_entry_price
    }

    pub const fn modeled_exit_price(&self) -> f64 {
        self.modeled_exit_price
    }

    pub const fn gross_pnl_quote_currency(&self) -> f64 {
        self.gross_pnl_quote_currency
    }

    pub fn gross_pnl_account_currency(&self) -> &AccountMoneyV1 {
        &self.gross_pnl_account_currency
    }

    pub fn entry_commission_account_currency(&self) -> &AccountMoneyV1 {
        &self.entry_commission_account_currency
    }

    pub fn exit_commission_account_currency(&self) -> &AccountMoneyV1 {
        &self.exit_commission_account_currency
    }

    pub fn swap_account_currency_signed(&self) -> &AccountMoneyV1 {
        &self.swap_account_currency_signed
    }

    pub fn pnl_conversion_fee_account_currency(&self) -> &AccountMoneyV1 {
        &self.pnl_conversion_fee_account_currency
    }

    pub fn additional_spread_account_currency(&self) -> &AccountMoneyV1 {
        &self.additional_spread_account_currency
    }

    pub fn net_pnl_account_currency(&self) -> &AccountMoneyV1 {
        &self.net_pnl_account_currency
    }

    pub fn entry_fill_identity_sha256(&self) -> &str {
        &self.entry_fill_identity_sha256
    }

    pub fn exit_fill_identity_sha256(&self) -> &str {
        &self.exit_fill_identity_sha256
    }

    pub const fn artifact_class(&self) -> ExecutionEconomicsArtifactClassV1 {
        self.artifact_class
    }

    pub const fn promotion_eligibility(&self) -> ExecutionEconomicsPromotionEligibilityV1 {
        self.promotion_eligibility
    }

    pub fn ledger_sha256(&self) -> &str {
        &self.ledger_sha256
    }

    fn hash_payload(&self) -> ExecutionEconomicsHashPayloadV1<'_> {
        ExecutionEconomicsHashPayloadV1 {
            schema_version: self.schema_version,
            quote_ledger_sha256: &self.quote_ledger_sha256,
            quote_position_index: self.quote_position_index,
            symbol_contract_identity_sha256: self.symbol_contract.identity_sha256(),
            account_currency: &self.account_currency,
            conversion_evidence_identity_sha256: self.conversion.identity_sha256(),
            entry_fill_identity_sha256: &self.entry_fill_identity_sha256,
            exit_fill_identity_sha256: &self.exit_fill_identity_sha256,
            commission_policy_identity_sha256: self.commission_policy.identity_sha256(),
            swap_evidence_identity_sha256: self.swap.identity_sha256(),
            pnl_conversion_fee_evidence_identity_sha256: self.pnl_conversion_fee.identity_sha256(),
            filled_lots: self.filled_lots,
            base_units: self.base_units,
            direction: self.direction,
            entry_fill_timestamp_unix_ms: self.entry_fill_timestamp_unix_ms,
            exit_fill_timestamp_unix_ms: self.exit_fill_timestamp_unix_ms,
            modeled_entry_price: self.modeled_entry_price,
            modeled_exit_price: self.modeled_exit_price,
            entry_notional_quote_currency: self.entry_notional_quote_currency,
            gross_pnl_quote_currency: self.gross_pnl_quote_currency,
            conversion_rate_account_per_quote: self.conversion.conversion_rate_account_per_quote,
            conversion_observed_at_unix_ms: self.conversion.conversion_observed_at_unix_ms,
            gross_pnl_account_currency: &self.gross_pnl_account_currency,
            entry_commission_account_currency: &self.entry_commission_account_currency,
            exit_commission_account_currency: &self.exit_commission_account_currency,
            swap_account_currency_signed: &self.swap_account_currency_signed,
            pnl_conversion_fee_account_currency: &self.pnl_conversion_fee_account_currency,
            additional_spread_account_currency: &self.additional_spread_account_currency,
            net_pnl_account_currency: &self.net_pnl_account_currency,
            artifact_class: self.artifact_class,
            promotion_eligibility: self.promotion_eligibility,
        }
    }

    fn recomputed_ledger_sha256(&self) -> Result<String, ExecutionEconomicsErrorV1> {
        hash_json(
            "quote-validated-execution-economics-ledger",
            &self.hash_payload(),
        )
    }

    fn validate(&self) -> Result<(), ExecutionEconomicsErrorV1> {
        if self.schema_version != EXECUTION_ECONOMICS_SCHEMA_VERSION_V1 {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::UnsupportedSchemaVersion,
                "execution economics artifact is not schema V1",
            ));
        }
        validate_digest("quote ledger", &self.quote_ledger_sha256)?;
        validate_digest("entry fill identity", &self.entry_fill_identity_sha256)?;
        validate_digest("exit fill identity", &self.exit_fill_identity_sha256)?;
        validate_digest("execution economics ledger", &self.ledger_sha256)?;
        self.symbol_contract.validate()?;
        self.conversion.validate()?;
        self.commission_policy.validate()?;
        self.swap.validate()?;
        self.pnl_conversion_fee.validate()?;
        validate_currency("account currency", &self.account_currency)?;
        validate_positive_finite(
            "filled lots",
            self.filled_lots,
            ExecutionEconomicsErrorCodeV1::InvalidFilledLots,
        )?;
        for (label, value) in [
            ("base units", self.base_units),
            ("modeled entry price", self.modeled_entry_price),
            ("modeled exit price", self.modeled_exit_price),
            (
                "entry notional quote currency",
                self.entry_notional_quote_currency,
            ),
        ] {
            validate_positive_finite(label, value, ExecutionEconomicsErrorCodeV1::InvalidMoney)?;
        }
        if !self.gross_pnl_quote_currency.is_finite() {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::InvalidMoney,
                "gross PnL in quote currency must be finite",
            ));
        }
        for money in [
            &self.gross_pnl_account_currency,
            &self.entry_commission_account_currency,
            &self.exit_commission_account_currency,
            &self.swap_account_currency_signed,
            &self.pnl_conversion_fee_account_currency,
            &self.additional_spread_account_currency,
            &self.net_pnl_account_currency,
        ] {
            money.validate()?;
            if money.currency != self.account_currency {
                return Err(economics_error(
                    ExecutionEconomicsErrorCodeV1::CurrencyMismatch,
                    "ledger cashflow currency differs from account currency",
                ));
            }
        }
        if self.symbol_contract.quote_currency != self.conversion.source_currency
            || self.conversion.target_account_currency != self.account_currency
            || self.commission_policy.account_currency != self.account_currency
            || self.swap.account_currency != self.account_currency
            || self.pnl_conversion_fee.account_currency != self.account_currency
        {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::CurrencyMismatch,
                "contract, conversion, commission, swap or fee currency differs from the ledger",
            ));
        }
        require_same_bits(
            "base units",
            self.base_units,
            self.symbol_contract.contract_units_per_lot * self.filled_lots,
        )?;
        require_same_bits(
            "entry notional quote currency",
            self.entry_notional_quote_currency,
            self.modeled_entry_price * self.base_units,
        )?;
        if self.entry_fill_timestamp_unix_ms <= 0
            || self.exit_fill_timestamp_unix_ms < self.entry_fill_timestamp_unix_ms
        {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::ArtifactDigestMismatch,
                "fill timestamps must be positive and entry must not follow exit",
            ));
        }
        self.conversion
            .validate_causal_for(self.exit_fill_timestamp_unix_ms)?;
        let direction_sign = match self.direction {
            ResearchPositionDirectionV1::Long => 1.0,
            ResearchPositionDirectionV1::Short => -1.0,
        };
        require_same_bits(
            "gross PnL quote currency",
            self.gross_pnl_quote_currency,
            direction_sign * (self.modeled_exit_price - self.modeled_entry_price) * self.base_units,
        )?;
        require_same_bits(
            "gross PnL account currency",
            self.gross_pnl_account_currency.amount,
            self.gross_pnl_quote_currency * self.conversion.conversion_rate_account_per_quote,
        )?;
        let expected_commission =
            self.commission_policy.commission_account_per_lot_per_fill * self.filled_lots;
        require_same_bits(
            "entry commission account currency",
            self.entry_commission_account_currency.amount,
            expected_commission,
        )?;
        require_same_bits(
            "exit commission account currency",
            self.exit_commission_account_currency.amount,
            expected_commission,
        )?;
        require_same_bits(
            "signed swap account currency",
            self.swap_account_currency_signed.amount,
            self.swap.swap_account_currency_signed,
        )?;
        require_same_bits(
            "PnL conversion fee account currency",
            self.pnl_conversion_fee_account_currency.amount,
            self.pnl_conversion_fee.pnl_conversion_fee_account_currency,
        )?;
        require_same_bits(
            "additional spread account currency",
            self.additional_spread_account_currency.amount,
            0.0,
        )?;
        let expected_net = self.gross_pnl_account_currency.amount
            - self.entry_commission_account_currency.amount
            - self.exit_commission_account_currency.amount
            + self.swap_account_currency_signed.amount
            - self.pnl_conversion_fee_account_currency.amount;
        require_same_bits(
            "net PnL account currency",
            self.net_pnl_account_currency.amount,
            expected_net,
        )?;
        if self.artifact_class != ExecutionEconomicsArtifactClassV1::ResearchOnly
            || self.promotion_eligibility
                != ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible
        {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::AuthorityMismatch,
                "execution economics V1 must remain research-only and not promotion eligible",
            ));
        }
        if self.ledger_sha256 != self.recomputed_ledger_sha256()? {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::ArtifactDigestMismatch,
                "execution economics ledger identity does not match its exact payload",
            ));
        }
        let expected_entry_fill_identity_sha256 = derived_fill_identity(
            &self.quote_ledger_sha256,
            self.quote_position_index,
            "entry",
            self.entry_fill_timestamp_unix_ms,
            self.modeled_entry_price,
            self.direction,
        )?;
        let expected_exit_fill_identity_sha256 = derived_fill_identity(
            &self.quote_ledger_sha256,
            self.quote_position_index,
            "exit",
            self.exit_fill_timestamp_unix_ms,
            self.modeled_exit_price,
            self.direction,
        )?;
        if self.entry_fill_identity_sha256 != expected_entry_fill_identity_sha256
            || self.exit_fill_identity_sha256 != expected_exit_fill_identity_sha256
        {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::ArtifactDigestMismatch,
                "derived fill identities do not match the sealed-ledger-bound fill payloads",
            ));
        }
        Ok(())
    }
}

fn derived_fill_identity(
    quote_ledger_sha256: &str,
    quote_position_index: u64,
    fill_role: &str,
    fill_timestamp_unix_ms: i64,
    modeled_fill_price: f64,
    direction: ResearchPositionDirectionV1,
) -> Result<String, ExecutionEconomicsErrorV1> {
    hash_json(
        "quote-validated-derived-fill",
        &DerivedFillIdentityHashPayloadV1 {
            quote_ledger_sha256,
            quote_position_index,
            fill_role,
            fill_timestamp_unix_ms,
            modeled_fill_price,
            direction,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_quote_validated_execution_economics_v1(
    sealed_quote_ledger: &SealedHistoricalQuoteValidatedResearchLedgerV1,
    quote_position_index: usize,
    symbol_contract: &ExecutionSymbolContractV1,
    account_currency: &str,
    filled_lots: f64,
    conversion: Option<&CausalQuoteToAccountConversionV1>,
    commission_policy: &ExecutionCommissionPolicyV1,
    swap: &SignedSwapCashflowV1,
    pnl_conversion_fee: &PnlConversionFeeV1,
) -> ExecutionEconomicsResultV1 {
    if sealed_quote_ledger.authority()
        != QuoteValidatedResearchAuthorityV1::HistoricalBidAskQuotesOnly
    {
        return Err(economics_error(
            ExecutionEconomicsErrorCodeV1::AuthorityMismatch,
            "execution economics requires the sealed historical quote-replay authority",
        ));
    }
    validate_digest("sealed quote ledger", sealed_quote_ledger.ledger_sha256())?;
    symbol_contract.validate()?;
    commission_policy.validate()?;
    swap.validate()?;
    pnl_conversion_fee.validate()?;
    validate_currency("account currency", account_currency)?;
    validate_positive_finite(
        "filled lots",
        filled_lots,
        ExecutionEconomicsErrorCodeV1::InvalidFilledLots,
    )?;
    if symbol_contract.symbol_name() != sealed_quote_ledger.receipt().symbol_name() {
        return Err(economics_error(
            ExecutionEconomicsErrorCodeV1::SymbolMismatch,
            "symbol contract does not match the sealed quote ledger symbol",
        ));
    }
    if commission_policy.account_currency() != account_currency
        || swap.account_currency() != account_currency
        || pnl_conversion_fee.account_currency() != account_currency
    {
        return Err(economics_error(
            ExecutionEconomicsErrorCodeV1::CurrencyMismatch,
            "commission, swap or conversion-fee currency differs from account currency",
        ));
    }
    let conversion = conversion.ok_or_else(|| {
        economics_error(
            ExecutionEconomicsErrorCodeV1::MissingConversionEvidence,
            "quote-to-account conversion evidence is mandatory, including identity conversion",
        )
    })?;
    conversion.validate()?;
    if conversion.source_currency() != symbol_contract.quote_currency()
        || conversion.target_account_currency() != account_currency
    {
        return Err(economics_error(
            ExecutionEconomicsErrorCodeV1::CurrencyMismatch,
            "conversion source/target does not match quote/account currency",
        ));
    }

    let position = sealed_quote_ledger
        .positions()
        .get(quote_position_index)
        .ok_or_else(|| {
            economics_error(
                ExecutionEconomicsErrorCodeV1::MissingPosition,
                "quote position index is outside the sealed ledger",
            )
        })?;
    let modeled_entry_price = position.modeled_entry_price();
    let Some(modeled_exit_price) = position.modeled_exit_price() else {
        return Err(economics_error(
            ExecutionEconomicsErrorCodeV1::MissingExitFill,
            "open quote-replay position has no modeled exit fill",
        ));
    };
    let Some(exit_reference) = position.exit_reference() else {
        return Err(economics_error(
            ExecutionEconomicsErrorCodeV1::MissingExitFill,
            "open quote-replay position has no causal exit reference",
        ));
    };
    conversion.validate_causal_for(exit_reference.timestamp_unix_ms())?;

    let contract_units_per_lot = symbol_contract.contract_units_per_lot();
    let base_units = contract_units_per_lot * filled_lots;
    validate_positive_finite(
        "base units",
        base_units,
        ExecutionEconomicsErrorCodeV1::InvalidFilledLots,
    )?;
    let entry_notional_quote_currency = modeled_entry_price * base_units;
    let direction_sign = match position.direction() {
        ResearchPositionDirectionV1::Long => 1.0,
        ResearchPositionDirectionV1::Short => -1.0,
    };
    let gross_pnl_quote_currency =
        direction_sign * (modeled_exit_price - modeled_entry_price) * base_units;
    let conversion_rate_account_per_quote = conversion.conversion_rate_account_per_quote();
    let gross_pnl_account_currency = gross_pnl_quote_currency * conversion_rate_account_per_quote;
    let commission_account_per_lot_per_fill =
        commission_policy.commission_account_per_lot_per_fill();
    let entry_commission_account_currency = commission_account_per_lot_per_fill * filled_lots;
    let exit_commission_account_currency = commission_account_per_lot_per_fill * filled_lots;
    let swap_account_currency_signed = swap.amount_signed();
    let pnl_conversion_fee_account_currency = pnl_conversion_fee.amount();
    let additional_spread_account_currency = AccountMoneyV1::zero(account_currency)?;
    let net_pnl_account_currency = gross_pnl_account_currency
        - entry_commission_account_currency
        - exit_commission_account_currency
        + swap_account_currency_signed
        - pnl_conversion_fee_account_currency;
    for (label, value) in [
        (
            "entry notional quote currency",
            entry_notional_quote_currency,
        ),
        ("gross PnL quote currency", gross_pnl_quote_currency),
        ("gross PnL account currency", gross_pnl_account_currency),
        (
            "entry commission account currency",
            entry_commission_account_currency,
        ),
        (
            "exit commission account currency",
            exit_commission_account_currency,
        ),
        ("net PnL account currency", net_pnl_account_currency),
    ] {
        if !value.is_finite() {
            return Err(economics_error(
                ExecutionEconomicsErrorCodeV1::InvalidMoney,
                format!("{label} is not finite"),
            ));
        }
    }

    let quote_position_index = u64::try_from(quote_position_index).map_err(|_| {
        economics_error(
            ExecutionEconomicsErrorCodeV1::InvalidContract,
            "quote position index cannot be represented in the V1 wire",
        )
    })?;
    let quote_ledger_sha256 = sealed_quote_ledger.ledger_sha256().to_owned();
    let entry_fill_identity_sha256 = derived_fill_identity(
        &quote_ledger_sha256,
        quote_position_index,
        "entry",
        position.entry_reference().timestamp_unix_ms(),
        modeled_entry_price,
        position.direction(),
    )?;
    let exit_fill_identity_sha256 = derived_fill_identity(
        &quote_ledger_sha256,
        quote_position_index,
        "exit",
        exit_reference.timestamp_unix_ms(),
        modeled_exit_price,
        position.direction(),
    )?;
    let mut ledger = QuoteValidatedExecutionEconomicsLedgerV1 {
        schema_version: EXECUTION_ECONOMICS_SCHEMA_VERSION_V1,
        quote_ledger_sha256,
        quote_position_index,
        symbol_contract: symbol_contract.clone(),
        account_currency: account_currency.to_owned(),
        filled_lots,
        base_units,
        direction: position.direction(),
        entry_fill_timestamp_unix_ms: position.entry_reference().timestamp_unix_ms(),
        exit_fill_timestamp_unix_ms: exit_reference.timestamp_unix_ms(),
        modeled_entry_price,
        modeled_exit_price,
        entry_notional_quote_currency,
        gross_pnl_quote_currency,
        conversion: conversion.clone(),
        gross_pnl_account_currency: AccountMoneyV1::new(
            account_currency,
            gross_pnl_account_currency,
        )?,
        commission_policy: commission_policy.clone(),
        entry_commission_account_currency: AccountMoneyV1::new(
            account_currency,
            entry_commission_account_currency,
        )?,
        exit_commission_account_currency: AccountMoneyV1::new(
            account_currency,
            exit_commission_account_currency,
        )?,
        swap: swap.clone(),
        swap_account_currency_signed: AccountMoneyV1::new(
            account_currency,
            swap_account_currency_signed,
        )?,
        pnl_conversion_fee: pnl_conversion_fee.clone(),
        pnl_conversion_fee_account_currency: AccountMoneyV1::new(
            account_currency,
            pnl_conversion_fee_account_currency,
        )?,
        additional_spread_account_currency,
        net_pnl_account_currency: AccountMoneyV1::new(account_currency, net_pnl_account_currency)?,
        entry_fill_identity_sha256,
        exit_fill_identity_sha256,
        artifact_class: ExecutionEconomicsArtifactClassV1::ResearchOnly,
        promotion_eligibility: ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible,
        ledger_sha256: String::new(),
    };
    ledger.ledger_sha256 = ledger.recomputed_ledger_sha256()?;
    ledger.validate()?;
    Ok(ledger)
}
