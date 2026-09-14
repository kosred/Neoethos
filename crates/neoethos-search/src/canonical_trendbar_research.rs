//! Receipt-bound canonical-trendbar authority for historical screening research.
//!
//! This module authorizes numerical research only. It cannot authorize live
//! execution or promotion, and it does not construct any broker-financial
//! capability. One exclusive scope is visible to parallel search workers for
//! the duration of the receipt-bound discovery call.
//!
//! The V2 screening-cost envelope contains operator/research assumptions, not
//! historical Bid/Ask fills. A later quote replay must use executable-side
//! prices directly and must not charge this envelope's spread a second time.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, Result, bail, ensure};
use neoethos_core::research_conversion_fee::ResearchPnlConversionFeePolicyV1;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::data_selection::{CanonicalSearchInputReceiptV2, CanonicalSearchRunInputV2};
use crate::discovery::DiscoveryResult;
use crate::historical_research::{
    HistoricalResearchArtifactClassV1, HistoricalResearchPromotionEligibilityV1,
};

pub const CANONICAL_TRENDBAR_SCREENING_COST_SCHEMA_VERSION_V2: u16 = 2;
pub const CANONICAL_TRENDBAR_RESEARCH_EXECUTION_SCHEMA_VERSION_V3: u16 = 3;
pub const CANONICAL_TRENDBAR_RESEARCH_DISCOVERY_RESULT_SCHEMA_VERSION_V3: u16 = 3;

const CONTRACT_IDENTITY_DOMAIN_V3: &[u8] =
    b"neoethos.canonical-trendbar-research-execution-contract.v3\0";
const RESULT_IDENTITY_DOMAIN_V3: &[u8] =
    b"neoethos.canonical-trendbar-research-discovery-result.v3\0";

/// Explicit scalar assumptions used only by broad canonical-bar screening.
///
/// Spread is the full quoted width for one round trip through midpoint bars.
/// Slippage and commission are one-fill/one-side values, so both are charged
/// twice. Commission is account currency per standard lot per fill.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalTrendbarScreeningCostEnvelopeV2 {
    schema_version: u16,
    full_spread_pips_assumption: f64,
    slippage_pips_per_fill_assumption: f64,
    commission_account_per_lot_per_fill_assumption: f64,
}

impl CanonicalTrendbarScreeningCostEnvelopeV2 {
    pub fn new(
        full_spread_pips_assumption: f64,
        slippage_pips_per_fill_assumption: f64,
        commission_account_per_lot_per_fill_assumption: f64,
    ) -> Result<Self> {
        let envelope = Self {
            schema_version: CANONICAL_TRENDBAR_SCREENING_COST_SCHEMA_VERSION_V2,
            full_spread_pips_assumption,
            slippage_pips_per_fill_assumption,
            commission_account_per_lot_per_fill_assumption,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub const fn full_spread_pips_assumption(&self) -> f64 {
        self.full_spread_pips_assumption
    }

    pub const fn slippage_pips_per_fill_assumption(&self) -> f64 {
        self.slippage_pips_per_fill_assumption
    }

    pub const fn commission_account_per_lot_per_fill_assumption(&self) -> f64 {
        self.commission_account_per_lot_per_fill_assumption
    }

    pub fn screening_spread_and_slippage_round_trip_pips(&self) -> f64 {
        self.full_spread_pips_assumption + 2.0 * self.slippage_pips_per_fill_assumption
    }

    pub fn round_trip_commission_account_per_lot(&self) -> f64 {
        2.0 * self.commission_account_per_lot_per_fill_assumption
    }

    pub fn screening_round_trip_cost_pips(&self, pip_value_account_per_lot: f64) -> f64 {
        self.screening_spread_and_slippage_round_trip_pips()
            + self.round_trip_commission_account_per_lot() / pip_value_account_per_lot
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == CANONICAL_TRENDBAR_SCREENING_COST_SCHEMA_VERSION_V2,
            "unsupported canonical-trendbar screening-cost schema {}",
            self.schema_version
        );
        require_non_negative_finite(
            "full_spread_pips_assumption",
            self.full_spread_pips_assumption,
        )?;
        require_non_negative_finite(
            "slippage_pips_per_fill_assumption",
            self.slippage_pips_per_fill_assumption,
        )?;
        require_non_negative_finite(
            "commission_account_per_lot_per_fill_assumption",
            self.commission_account_per_lot_per_fill_assumption,
        )?;
        ensure!(
            self.screening_spread_and_slippage_round_trip_pips()
                .is_finite(),
            "screening spread/slippage round-trip cost must be finite"
        );
        ensure!(
            self.round_trip_commission_account_per_lot().is_finite(),
            "screening round-trip commission must be finite"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalTrendbarResearchExecutionContractV3 {
    schema_version: u16,
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    input_receipt: CanonicalSearchInputReceiptV2,
    input_receipt_sha256: String,
    symbol: String,
    account_currency: String,
    assumption_source_id: String,
    assumption_source_sha256: String,
    pip_size: f64,
    pip_value_per_lot: f64,
    screening_costs: CanonicalTrendbarScreeningCostEnvelopeV2,
    swap_long_pips_per_day: f64,
    swap_short_pips_per_day: f64,
    pnl_conversion_fee_rate: f64,
    /// Mandatory nested policy version; pre-policy wire must not be reinterpreted.
    pnl_conversion_fee_policy: ResearchPnlConversionFeePolicyV1,
}

/// Explicit compact wire body. Its receipt must come from the same enclosing
/// artifact; attaching it restores and validates the unchanged V3 contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalTrendbarResearchExecutionContractRefV1 {
    schema_version: u16,
    contract_schema_version: u16,
    contract_identity_sha256: String,
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    input_receipt_sha256: String,
    symbol: String,
    account_currency: String,
    assumption_source_id: String,
    assumption_source_sha256: String,
    pip_size: f64,
    pip_value_per_lot: f64,
    screening_costs: CanonicalTrendbarScreeningCostEnvelopeV2,
    swap_long_pips_per_day: f64,
    swap_short_pips_per_day: f64,
    pnl_conversion_fee_rate: f64,
    pnl_conversion_fee_policy: ResearchPnlConversionFeePolicyV1,
}

impl CanonicalTrendbarResearchExecutionContractRefV1 {
    pub fn from_contract(contract: &CanonicalTrendbarResearchExecutionContractV3) -> Result<Self> {
        let contract_identity_sha256 = contract.identity_sha256()?;
        Ok(Self {
            schema_version: 1,
            contract_schema_version: contract.schema_version,
            contract_identity_sha256,
            artifact_class: contract.artifact_class,
            promotion_eligibility: contract.promotion_eligibility,
            input_receipt_sha256: contract.input_receipt_sha256.clone(),
            symbol: contract.symbol.clone(),
            account_currency: contract.account_currency.clone(),
            assumption_source_id: contract.assumption_source_id.clone(),
            assumption_source_sha256: contract.assumption_source_sha256.clone(),
            pip_size: contract.pip_size,
            pip_value_per_lot: contract.pip_value_per_lot,
            screening_costs: contract.screening_costs.clone(),
            swap_long_pips_per_day: contract.swap_long_pips_per_day,
            swap_short_pips_per_day: contract.swap_short_pips_per_day,
            pnl_conversion_fee_rate: contract.pnl_conversion_fee_rate,
            pnl_conversion_fee_policy: contract.pnl_conversion_fee_policy,
        })
    }

    pub fn input_receipt_sha256(&self) -> &str {
        &self.input_receipt_sha256
    }

    pub fn attach(
        &self,
        receipt: &CanonicalSearchInputReceiptV2,
    ) -> Result<CanonicalTrendbarResearchExecutionContractV3> {
        ensure!(
            self.schema_version == 1,
            "unsupported shared screening-contract schema"
        );
        validate_sha256("shared screening contract", &self.contract_identity_sha256)?;
        ensure!(
            receipt.identity_sha256().map_err(anyhow::Error::new)? == self.input_receipt_sha256,
            "shared screening contract names a different receipt"
        );
        let contract = CanonicalTrendbarResearchExecutionContractV3 {
            schema_version: self.contract_schema_version,
            artifact_class: self.artifact_class,
            promotion_eligibility: self.promotion_eligibility,
            input_receipt: receipt.clone(),
            input_receipt_sha256: self.input_receipt_sha256.clone(),
            symbol: self.symbol.clone(),
            account_currency: self.account_currency.clone(),
            assumption_source_id: self.assumption_source_id.clone(),
            assumption_source_sha256: self.assumption_source_sha256.clone(),
            pip_size: self.pip_size,
            pip_value_per_lot: self.pip_value_per_lot,
            screening_costs: self.screening_costs.clone(),
            swap_long_pips_per_day: self.swap_long_pips_per_day,
            swap_short_pips_per_day: self.swap_short_pips_per_day,
            pnl_conversion_fee_rate: self.pnl_conversion_fee_rate,
            pnl_conversion_fee_policy: self.pnl_conversion_fee_policy,
        };
        contract.validate_against_receipt(receipt)?;
        ensure!(
            contract.identity_sha256()? == self.contract_identity_sha256,
            "shared screening contract changed its exact identity"
        );
        Ok(contract)
    }
}

#[derive(Debug, Clone)]
pub struct CanonicalTrendbarResearchCostAssumptionsV2<'a> {
    pub symbol: &'a str,
    pub account_currency: &'a str,
    pub assumption_source_id: &'a str,
    pub assumption_source_sha256: &'a str,
    pub pip_size: f64,
    pub pip_value_per_lot: f64,
    pub full_spread_pips_assumption: f64,
    pub slippage_pips_per_fill_assumption: f64,
    pub commission_account_per_lot_per_fill_assumption: f64,
    pub swap_long_pips_per_day: f64,
    pub swap_short_pips_per_day: f64,
    pub pnl_conversion_fee_rate: f64,
}

impl CanonicalTrendbarResearchExecutionContractV3 {
    pub fn new(
        input_receipt: CanonicalSearchInputReceiptV2,
        assumptions: CanonicalTrendbarResearchCostAssumptionsV2<'_>,
    ) -> Result<Self> {
        let input_receipt_sha256 = input_receipt
            .identity_sha256()
            .map_err(anyhow::Error::new)
            .context("hash canonical research input receipt")?;
        let contract = Self {
            schema_version: CANONICAL_TRENDBAR_RESEARCH_EXECUTION_SCHEMA_VERSION_V3,
            artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
            promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
            input_receipt,
            input_receipt_sha256,
            symbol: assumptions.symbol.to_owned(),
            account_currency: assumptions.account_currency.to_owned(),
            assumption_source_id: assumptions.assumption_source_id.to_owned(),
            assumption_source_sha256: assumptions.assumption_source_sha256.to_owned(),
            pip_size: assumptions.pip_size,
            pip_value_per_lot: assumptions.pip_value_per_lot,
            screening_costs: CanonicalTrendbarScreeningCostEnvelopeV2::new(
                assumptions.full_spread_pips_assumption,
                assumptions.slippage_pips_per_fill_assumption,
                assumptions.commission_account_per_lot_per_fill_assumption,
            )?,
            swap_long_pips_per_day: assumptions.swap_long_pips_per_day,
            swap_short_pips_per_day: assumptions.swap_short_pips_per_day,
            pnl_conversion_fee_rate: effective_contract_conversion_fee_rate(
                assumptions.symbol,
                assumptions.account_currency,
                assumptions.pnl_conversion_fee_rate,
            )?,
            pnl_conversion_fee_policy:
                ResearchPnlConversionFeePolicyV1::AbsoluteRealizedPriceGrossDebitV1,
        };
        contract.validate()?;
        Ok(contract)
    }

    pub const fn artifact_class(&self) -> HistoricalResearchArtifactClassV1 {
        self.artifact_class
    }

    pub const fn promotion_eligibility(&self) -> HistoricalResearchPromotionEligibilityV1 {
        self.promotion_eligibility
    }

    pub const fn input_receipt(&self) -> &CanonicalSearchInputReceiptV2 {
        &self.input_receipt
    }

    pub fn input_receipt_sha256(&self) -> &str {
        &self.input_receipt_sha256
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn account_currency(&self) -> &str {
        &self.account_currency
    }

    pub fn assumption_source_id(&self) -> &str {
        &self.assumption_source_id
    }

    pub fn assumption_source_sha256(&self) -> &str {
        &self.assumption_source_sha256
    }

    pub const fn pip_size(&self) -> f64 {
        self.pip_size
    }

    pub const fn pip_value_per_lot(&self) -> f64 {
        self.pip_value_per_lot
    }

    pub const fn screening_costs(&self) -> &CanonicalTrendbarScreeningCostEnvelopeV2 {
        &self.screening_costs
    }

    pub fn screening_spread_and_slippage_round_trip_pips(&self) -> f64 {
        self.screening_costs
            .screening_spread_and_slippage_round_trip_pips()
    }

    pub fn round_trip_commission_account_per_lot(&self) -> f64 {
        self.screening_costs.round_trip_commission_account_per_lot()
    }

    pub fn screening_round_trip_cost_pips(&self) -> f64 {
        self.screening_costs
            .screening_round_trip_cost_pips(self.pip_value_per_lot)
    }

    pub const fn swap_long_pips_per_day(&self) -> f64 {
        self.swap_long_pips_per_day
    }

    pub const fn swap_short_pips_per_day(&self) -> f64 {
        self.swap_short_pips_per_day
    }

    pub const fn pnl_conversion_fee_rate(&self) -> f64 {
        self.pnl_conversion_fee_rate
    }

    /// Compare the exact scalar account/cost inputs without rehashing the receipt.
    /// Callers must separately validate this contract and their input binding.
    /// This checks neither the remaining evaluation policy nor live authority.
    pub fn validate_evaluation_costs(&self, evaluation: &crate::EvaluationConfig) -> Result<()> {
        for (field, actual, expected) in [
            ("symbol", evaluation.symbol.as_str(), self.symbol()),
            (
                "account_currency",
                evaluation.account_currency.as_str(),
                self.account_currency(),
            ),
        ] {
            ensure!(
                actual == expected,
                "netted bar costs/account differ from the exact saved screening assumptions: {field} (evaluation {actual:?}, contract {expected:?})"
            );
        }
        for (field, actual, expected) in [
            ("pip_value", evaluation.pip_value, self.pip_size()),
            (
                "pip_value_per_lot",
                evaluation.pip_value_per_lot,
                self.pip_value_per_lot(),
            ),
            (
                "spread_pips",
                evaluation.spread_pips,
                self.screening_spread_and_slippage_round_trip_pips(),
            ),
            (
                "commission_per_trade",
                evaluation.commission_per_trade,
                self.round_trip_commission_account_per_lot(),
            ),
            (
                "swap_long_pips_per_day",
                evaluation.swap_long_pips_per_day,
                self.swap_long_pips_per_day(),
            ),
            (
                "swap_short_pips_per_day",
                evaluation.swap_short_pips_per_day,
                self.swap_short_pips_per_day(),
            ),
            (
                "pnl_conversion_fee_rate",
                evaluation.pnl_conversion_fee_rate,
                self.pnl_conversion_fee_rate(),
            ),
        ] {
            ensure!(
                actual == expected,
                "netted bar costs/account differ from the exact saved screening assumptions: {field} (evaluation {actual:?}, contract {expected:?})"
            );
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        #[cfg(test)]
        tests::CONTRACT_VALIDATIONS.with(|calls| calls.set(calls.get() + 1));
        ensure!(
            self.schema_version == CANONICAL_TRENDBAR_RESEARCH_EXECUTION_SCHEMA_VERSION_V3,
            "unsupported canonical-trendbar research contract schema {}",
            self.schema_version
        );
        ensure!(
            self.artifact_class == HistoricalResearchArtifactClassV1::ResearchOnly
                && self.promotion_eligibility
                    == HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
            "canonical-trendbar research contract must remain ResearchOnly and NotPromotionEligible"
        );
        self.input_receipt
            .validate()
            .map_err(anyhow::Error::new)
            .context("validate embedded canonical search receipt")?;
        let actual_receipt_sha256 = self
            .input_receipt
            .identity_sha256()
            .map_err(anyhow::Error::new)?;
        ensure!(
            self.input_receipt_sha256 == actual_receipt_sha256,
            "canonical-trendbar research receipt SHA-256 does not match its embedded receipt"
        );
        validate_identity_text("symbol", &self.symbol)?;
        validate_account_currency(&self.account_currency)?;
        validate_identity_text("assumption source id", &self.assumption_source_id)?;
        validate_sha256("assumption source", &self.assumption_source_sha256)?;
        require_positive_finite("pip_size", self.pip_size)?;
        require_positive_finite("pip_value_per_lot", self.pip_value_per_lot)?;
        self.screening_costs.validate()?;
        require_non_negative_finite(
            "screening_round_trip_cost_pips",
            self.screening_round_trip_cost_pips(),
        )?;
        require_finite("swap_long_pips_per_day", self.swap_long_pips_per_day)?;
        require_finite("swap_short_pips_per_day", self.swap_short_pips_per_day)?;
        require_finite("pnl_conversion_fee_rate", self.pnl_conversion_fee_rate)?;
        ensure!(
            (0.0..1.0).contains(&self.pnl_conversion_fee_rate),
            "pnl_conversion_fee_rate must be in [0, 1)"
        );
        ensure!(
            effective_contract_conversion_fee_rate(
                &self.symbol,
                &self.account_currency,
                self.pnl_conversion_fee_rate
            )?
            .to_bits()
                == self.pnl_conversion_fee_rate.to_bits(),
            "same-currency research contract must carry zero effective conversion fee"
        );
        Ok(())
    }

    pub fn validate_against_input(&self, input: &CanonicalSearchRunInputV2<'_>) -> Result<()> {
        self.validate_against_receipt(input.receipt())?;
        ensure!(
            input.receipt() == &self.input_receipt,
            "canonical-trendbar research contract receipt does not match the exact run input"
        );
        ensure!(
            input.anchor_identity().symbol_name() == self.symbol,
            "canonical-trendbar research symbol {} does not match input symbol {}",
            self.symbol,
            input.anchor_identity().symbol_name()
        );
        Ok(())
    }

    /// Validate the contract against an already sealed canonical-search input
    /// receipt without retaining or rebuilding the search feature frame.
    ///
    /// This is the training hand-off boundary used after an independently
    /// completed historical search. It deliberately proves the same exact
    /// receipt and anchor symbol as [`Self::validate_against_input`]; it is not
    /// a symbol-only or settings-only fallback.
    pub fn validate_against_receipt(&self, receipt: &CanonicalSearchInputReceiptV2) -> Result<()> {
        self.validate()?;
        let anchor = receipt
            .validate()
            .map_err(anyhow::Error::new)
            .context("validate canonical training input receipt")?;
        ensure!(
            receipt == &self.input_receipt,
            "canonical-trendbar research contract receipt does not match the exact training receipt"
        );
        ensure!(
            anchor.symbol_name() == self.symbol,
            "canonical-trendbar research symbol {} does not match receipt symbol {}",
            self.symbol,
            anchor.symbol_name()
        );
        Ok(())
    }

    pub fn identity_sha256(&self) -> Result<String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).context("serialize canonical research contract")?;
        Ok(domain_sha256(CONTRACT_IDENTITY_DOMAIN_V3, &bytes))
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalTrendbarResearchDiscoveryResultV3 {
    schema_version: u16,
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    execution_contract: CanonicalTrendbarResearchExecutionContractV3,
    discovery_result: DiscoveryResult,
    evidence_identity_sha256: String,
}

impl CanonicalTrendbarResearchDiscoveryResultV3 {
    pub(crate) fn new(
        execution_contract: CanonicalTrendbarResearchExecutionContractV3,
        discovery_result: DiscoveryResult,
    ) -> Result<Self> {
        let evidence_identity_sha256 =
            result_identity_sha256(&execution_contract, &discovery_result)?;
        let result = Self {
            schema_version: CANONICAL_TRENDBAR_RESEARCH_DISCOVERY_RESULT_SCHEMA_VERSION_V3,
            artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
            promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
            execution_contract,
            discovery_result,
            evidence_identity_sha256,
        };
        result.validate()?;
        Ok(result)
    }

    pub const fn artifact_class(&self) -> HistoricalResearchArtifactClassV1 {
        self.artifact_class
    }

    pub const fn promotion_eligibility(&self) -> HistoricalResearchPromotionEligibilityV1 {
        self.promotion_eligibility
    }

    pub const fn execution_contract(&self) -> &CanonicalTrendbarResearchExecutionContractV3 {
        &self.execution_contract
    }

    pub const fn discovery_result(&self) -> &DiscoveryResult {
        &self.discovery_result
    }

    pub fn evidence_identity_sha256(&self) -> &str {
        &self.evidence_identity_sha256
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == CANONICAL_TRENDBAR_RESEARCH_DISCOVERY_RESULT_SCHEMA_VERSION_V3,
            "unsupported canonical-trendbar research discovery-result schema {}",
            self.schema_version
        );
        ensure!(
            self.artifact_class == HistoricalResearchArtifactClassV1::ResearchOnly
                && self.promotion_eligibility
                    == HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
            "canonical-trendbar discovery result is not research-only"
        );
        self.execution_contract.validate()?;
        self.discovery_result.validate_evaluated_scopes()?;
        ensure!(
            self.discovery_result.search_input_receipt == *self.execution_contract.input_receipt(),
            "canonical-trendbar discovery result lost its exact research receipt"
        );
        validate_sha256(
            "research discovery evidence",
            &self.evidence_identity_sha256,
        )?;
        ensure!(
            self.evidence_identity_sha256
                == result_identity_sha256(&self.execution_contract, &self.discovery_result)?,
            "canonical-trendbar discovery evidence identity does not match its result"
        );
        Ok(())
    }
}

#[derive(Debug)]
struct ActiveCanonicalTrendbarResearchExecutionV3 {
    token: u64,
    contract: Arc<CanonicalTrendbarResearchExecutionContractV3>,
}

static ACTIVE_CANONICAL_TRENDBAR_RESEARCH_EXECUTION_V3: Mutex<
    Option<ActiveCanonicalTrendbarResearchExecutionV3>,
> = Mutex::new(None);
static NEXT_CANONICAL_TRENDBAR_RESEARCH_TOKEN_V3: AtomicU64 = AtomicU64::new(1);

pub(crate) struct CanonicalTrendbarResearchExecutionScopeV3 {
    token: u64,
}

impl Drop for CanonicalTrendbarResearchExecutionScopeV3 {
    fn drop(&mut self) {
        let mut active = lock_active();
        if active.as_ref().map(|value| value.token) == Some(self.token) {
            active.take();
        }
    }
}

pub(crate) fn install_canonical_trendbar_research_execution_v3(
    contract: &CanonicalTrendbarResearchExecutionContractV3,
) -> Result<CanonicalTrendbarResearchExecutionScopeV3> {
    // This is the trust boundary for every new scope, including decoded or
    // caller-modified contracts. The published deep clone has no mutable access
    // or interior mutability; changing the caller's copy cannot change it.
    contract.validate()?;
    let mut active = lock_active();
    if active.is_some() {
        bail!("active canonical-trendbar research execution already exists");
    }
    let token = NEXT_CANONICAL_TRENDBAR_RESEARCH_TOKEN_V3.fetch_add(1, Ordering::Relaxed);
    ensure!(
        token != 0,
        "canonical-trendbar research token space exhausted"
    );
    *active = Some(ActiveCanonicalTrendbarResearchExecutionV3 {
        token,
        contract: Arc::new(contract.clone()),
    });
    Ok(CanonicalTrendbarResearchExecutionScopeV3 { token })
}

/// Return only the snapshot validated by installation, while its scope is active.
/// The slot retains an Arc, so callers cannot obtain unique mutable access to it.
pub(crate) fn active_canonical_trendbar_research_execution_v3()
-> Option<Arc<CanonicalTrendbarResearchExecutionContractV3>> {
    lock_active()
        .as_ref()
        .map(|active| Arc::clone(&active.contract))
}

fn lock_active() -> MutexGuard<'static, Option<ActiveCanonicalTrendbarResearchExecutionV3>> {
    ACTIVE_CANONICAL_TRENDBAR_RESEARCH_EXECUTION_V3
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn result_identity_sha256(
    contract: &CanonicalTrendbarResearchExecutionContractV3,
    result: &DiscoveryResult,
) -> Result<String> {
    contract.validate()?;
    result.validate_evaluated_scopes()?;
    let mut bytes = Vec::new();
    push_string(&mut bytes, &contract.identity_sha256()?);
    let result_bytes =
        serde_json::to_vec(result).context("serialize complete research discovery result")?;
    push_bytes(&mut bytes, &result_bytes);
    Ok(domain_sha256(RESULT_IDENTITY_DOMAIN_V3, &bytes))
}

fn validate_identity_text(label: &str, value: &str) -> Result<()> {
    let trimmed = value.trim();
    ensure!(!trimmed.is_empty(), "{label} is empty");
    ensure!(
        trimmed == value,
        "{label} has leading or trailing whitespace"
    );
    ensure!(value.len() <= 128, "{label} exceeds 128 bytes");
    ensure!(
        value.bytes().all(|byte| byte.is_ascii_graphic()),
        "{label} contains non-ASCII or control bytes"
    );
    Ok(())
}

fn validate_account_currency(value: &str) -> Result<()> {
    ensure!(
        value.len() == 3 && value.bytes().all(|byte| byte.is_ascii_uppercase()),
        "account_currency must be an exact three-letter uppercase code"
    );
    Ok(())
}

fn effective_contract_conversion_fee_rate(symbol: &str, account: &str, rate: f64) -> Result<f64> {
    use neoethos_core::research_conversion_fee::{
        effective_conversion_fee_rate_v1, validate_conversion_fee_rate_v1,
    };
    validate_conversion_fee_rate_v1(rate).map_err(anyhow::Error::msg)?;
    if rate == 0.0 {
        return Ok(rate);
    }
    ensure!(
        symbol.len() == 6 && symbol.bytes().all(|byte| byte.is_ascii_uppercase()),
        "nonzero research conversion fee requires an exact six-letter FX symbol"
    );
    effective_conversion_fee_rate_v1(rate, &symbol[3..], account).map_err(anyhow::Error::msg)
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} SHA-256 must be 64 lowercase hexadecimal characters"
    );
    Ok(())
}

fn require_finite(label: &str, value: f64) -> Result<()> {
    ensure!(value.is_finite(), "{label} must be finite");
    Ok(())
}

fn require_positive_finite(label: &str, value: f64) -> Result<()> {
    require_finite(label, value)?;
    ensure!(value > 0.0, "{label} must be positive");
    Ok(())
}

fn require_non_negative_finite(label: &str, value: f64) -> Result<()> {
    require_finite(label, value)?;
    ensure!(value >= 0.0, "{label} must be non-negative");
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn push_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u64).to_be_bytes());
    target.extend_from_slice(value);
}

fn push_string(target: &mut Vec<u8>, value: &str) {
    push_bytes(target, value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    std::thread_local! {
        pub(super) static CONTRACT_VALIDATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[test]
    fn installed_research_scope_validates_once_and_shares_immutable_authority() -> Result<()> {
        use crate::historical_evaluation_authority::{
            HistoricalEvaluationAuthorityV1, require_historical_evaluation_authority_v1,
        };

        const CHILD: &str = "NEOETHOS_TEST_RESEARCH_SCOPE_ONCE_CHILD";
        const TEST: &str = "canonical_trendbar_research::tests::installed_research_scope_validates_once_and_shares_immutable_authority";
        const COMPLETED: &str = "validated-research-scope-once-pass";
        if std::env::var_os(CHILD).is_none() {
            // The active scope is process-wide; do not leak authority into other tests.
            let mut child = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()?;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while child.try_wait()?.is_none() {
                if std::time::Instant::now() >= deadline {
                    child.kill()?;
                    let output = child.wait_with_output()?;
                    bail!(
                        "research-scope child timed out:\n{}\n{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            let output = child.wait_with_output()?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            print!("{stdout}");
            eprint!("{stderr}");
            ensure!(
                output.status.success()
                    && stdout.contains("test result: ok. 1 passed; 0 failed;")
                    && stdout.lines().any(|line| line.ends_with(COMPLETED)),
                "isolated research-scope test did not complete exactly once"
            );
            return Ok(());
        }

        assert!(active_canonical_trendbar_research_execution_v3().is_none());
        assert!(require_historical_evaluation_authority_v1().is_err());
        let frame = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let anchor = frame.provenance().bindings()[0].dataset_identity();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(anchor, &frame)?;
        let mut original = CanonicalTrendbarResearchExecutionContractV3::new(
            receipt,
            CanonicalTrendbarResearchCostAssumptionsV2 {
                symbol: "EURUSD",
                account_currency: "USD",
                assumption_source_id: "validated-scope-test",
                assumption_source_sha256: &"a".repeat(64),
                pip_size: 0.0001,
                pip_value_per_lot: 10.0,
                full_spread_pips_assumption: 1.5,
                slippage_pips_per_fill_assumption: 0.5,
                commission_account_per_lot_per_fill_assumption: 7.0,
                swap_long_pips_per_day: -0.25,
                swap_short_pips_per_day: 0.1,
                pnl_conversion_fee_rate: 0.0,
            },
        )?;
        // Deserialization itself is not authority: installation must still reject
        // changed receipt bindings and invalid scalar costs before publication.
        for (field, value) in [
            ("input_receipt_sha256", serde_json::json!("0".repeat(64))),
            ("pip_value_per_lot", serde_json::json!(0.0)),
        ] {
            let mut wire = serde_json::to_value(&original)?;
            wire[field] = value;
            let invalid = serde_json::from_value(wire)?;
            CONTRACT_VALIDATIONS.with(|calls| calls.set(0));
            assert!(install_canonical_trendbar_research_execution_v3(&invalid).is_err());
            assert_eq!(CONTRACT_VALIDATIONS.with(|calls| calls.get()), 1);
            assert!(active_canonical_trendbar_research_execution_v3().is_none());
        }

        CONTRACT_VALIDATIONS.with(|calls| calls.set(0));
        let scope = install_canonical_trendbar_research_execution_v3(&original)?;
        assert_eq!(CONTRACT_VALIDATIONS.with(|calls| calls.get()), 1);
        let snapshot = active_canonical_trendbar_research_execution_v3().unwrap();
        assert_eq!(snapshot.as_ref(), &original);
        let mut shared = Arc::clone(&snapshot);
        assert!(Arc::get_mut(&mut shared).is_none());
        original.pip_value_per_lot = 0.0;
        assert_eq!(snapshot.pip_value_per_lot(), 10.0);

        std::thread::scope(|threads| {
            for _ in 0..10 {
                let expected = &snapshot;
                threads.spawn(move || {
                    CONTRACT_VALIDATIONS.with(|calls| calls.set(0));
                    for _ in 0..100 {
                        let authority = require_historical_evaluation_authority_v1().unwrap();
                        let HistoricalEvaluationAuthorityV1::CanonicalTrendbarResearch(actual) =
                            authority
                        else {
                            panic!("active research scope was not selected");
                        };
                        assert!(Arc::ptr_eq(&actual, expected));
                        assert_eq!(actual.pip_value_per_lot(), 10.0);
                    }
                    assert_eq!(CONTRACT_VALIDATIONS.with(|calls| calls.get()), 0);
                });
            }
        });
        assert_eq!(CONTRACT_VALIDATIONS.with(|calls| calls.get()), 1);
        assert!(install_canonical_trendbar_research_execution_v3(&snapshot).is_err());
        assert!(Arc::ptr_eq(
            &snapshot,
            &active_canonical_trendbar_research_execution_v3().unwrap()
        ));
        drop(scope);
        assert!(active_canonical_trendbar_research_execution_v3().is_none());
        assert!(require_historical_evaluation_authority_v1().is_err());
        // Retaining an old Arc does not reinstall authority; a mutated original
        // must be validated again and cannot establish the next scope.
        assert!(install_canonical_trendbar_research_execution_v3(&original).is_err());
        let next_scope = install_canonical_trendbar_research_execution_v3(&snapshot)?;
        drop(next_scope);
        assert!(active_canonical_trendbar_research_execution_v3().is_none());
        println!("{COMPLETED}");
        Ok(())
    }

    #[test]
    #[ignore = "explicit real-receipt lookup benchmark; run alone with an external timeout"]
    fn real_receipt_synthetic_cost_contract_lookup_benchmark() -> Result<()> {
        use crate::historical_evaluation_authority::{
            HistoricalEvaluationAuthorityV1, require_historical_evaluation_authority_v1,
        };

        #[derive(Deserialize)]
        struct TrialReturns {
            search_input_receipt: CanonicalSearchInputReceiptV2,
        }

        // Read the exact recorded receipt, not a regenerated feature fixture.
        // These USD costs are deliberately synthetic: this measures validation
        // overhead, not the original run's financial authority or GA throughput.
        let source = std::env::var_os("NEOETHOS_TEST_RESEARCH_SCOPE_RECEIPT_FILE")
            .context("set NEOETHOS_TEST_RESEARCH_SCOPE_RECEIPT_FILE to trial_returns.v3.json")?;
        let source = std::path::PathBuf::from(source);
        let setup_started = std::time::Instant::now();
        let file = std::fs::File::open(&source)?;
        let source_bytes = file.metadata()?.len();
        let TrialReturns {
            search_input_receipt,
        } = serde_json::from_reader(std::io::BufReader::new(file))?;
        let plan_width = search_input_receipt
            .recorded_feature_plan()?
            .context("benchmark requires the actual recorded feature plan")?
            .final_outputs()
            .len();
        let contract = CanonicalTrendbarResearchExecutionContractV3::new(
            search_input_receipt,
            CanonicalTrendbarResearchCostAssumptionsV2 {
                symbol: "EURUSD",
                account_currency: "USD",
                assumption_source_id: "real-receipt-synthetic-cost-benchmark",
                assumption_source_sha256: &"a".repeat(64),
                pip_size: 0.0001,
                pip_value_per_lot: 10.0,
                full_spread_pips_assumption: 1.5,
                slippage_pips_per_fill_assumption: 0.5,
                commission_account_per_lot_per_fill_assumption: 7.0,
                swap_long_pips_per_day: -0.25,
                swap_short_pips_per_day: 0.1,
                pnl_conversion_fee_rate: 0.0,
            },
        )?;
        assert!(active_canonical_trendbar_research_execution_v3().is_none());
        let scope = install_canonical_trendbar_research_execution_v3(&contract)?;
        let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;

        // Exactly one old-style validation; no additional warmup iterations.
        CONTRACT_VALIDATIONS.with(|calls| calls.set(0));
        let old_started = std::time::Instant::now();
        let old = require_historical_evaluation_authority_v1()?;
        let HistoricalEvaluationAuthorityV1::CanonicalTrendbarResearch(old) = old else {
            bail!("benchmark did not receive its installed research scope");
        };
        old.validate()?;
        std::hint::black_box(old);
        let old_per_call_us = old_started.elapsed().as_secs_f64() * 1_000_000.0;
        assert_eq!(CONTRACT_VALIDATIONS.with(|calls| calls.get()), 1);

        const LOOKUPS: usize = 1_000;
        CONTRACT_VALIDATIONS.with(|calls| calls.set(0));
        let new_started = std::time::Instant::now();
        for _ in 0..LOOKUPS {
            std::hint::black_box(require_historical_evaluation_authority_v1()?);
        }
        let new_per_call_us = new_started.elapsed().as_secs_f64() * 1_000_000.0 / LOOKUPS as f64;
        assert_eq!(CONTRACT_VALIDATIONS.with(|calls| calls.get()), 0);
        println!(
            "real_receipt_synthetic_cost_contract source={source:?} source_bytes={source_bytes} anchor={} receipt_sha256={} plan_width={plan_width} setup_ms={setup_ms:.6} old_lookups=1 old_validations=1 old_per_call_us={old_per_call_us:.6} new_lookups={LOOKUPS} new_validations=0 new_per_call_us={new_per_call_us:.6}",
            contract.input_receipt().anchor_dataset_identity(),
            contract.input_receipt_sha256(),
        );
        drop(scope);
        assert!(active_canonical_trendbar_research_execution_v3().is_none());
        Ok(())
    }

    #[test]
    fn shared_screening_contract_roundtrip_keeps_v3_hash_and_rejects_changed_costs() {
        let frame = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let anchor = frame.provenance().bindings()[0].dataset_identity();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(anchor, &frame).unwrap();
        let contract = CanonicalTrendbarResearchExecutionContractV3::new(
            receipt.clone(),
            CanonicalTrendbarResearchCostAssumptionsV2 {
                symbol: "EURUSD",
                account_currency: "USD",
                assumption_source_id: "shared-contract-test",
                assumption_source_sha256: &"a".repeat(64),
                pip_size: 0.0001,
                pip_value_per_lot: 10.0,
                full_spread_pips_assumption: 1.5,
                slippage_pips_per_fill_assumption: 0.5,
                commission_account_per_lot_per_fill_assumption: 7.0,
                swap_long_pips_per_day: -0.25,
                swap_short_pips_per_day: 0.1,
                pnl_conversion_fee_rate: 0.01,
            },
        )
        .unwrap();
        let compact =
            CanonicalTrendbarResearchExecutionContractRefV1::from_contract(&contract).unwrap();
        let bytes = serde_json::to_vec(&compact).unwrap();
        let mut missing_policy = serde_json::to_value(&compact).unwrap();
        missing_policy
            .as_object_mut()
            .unwrap()
            .remove("pnl_conversion_fee_policy");
        assert!(
            serde_json::from_value::<CanonicalTrendbarResearchExecutionContractRefV1>(
                missing_policy
            )
            .is_err()
        );
        let mut missing_full_policy = serde_json::to_value(&contract).unwrap();
        missing_full_policy
            .as_object_mut()
            .unwrap()
            .remove("pnl_conversion_fee_policy");
        assert!(
            serde_json::from_value::<CanonicalTrendbarResearchExecutionContractV3>(
                missing_full_policy
            )
            .is_err()
        );
        assert_eq!(
            contract.pnl_conversion_fee_rate(),
            0.0,
            "USD quote on USD account has no conversion fee"
        );
        let mut different_fee = contract.clone();
        different_fee.account_currency = "EUR".to_owned();
        different_fee.pnl_conversion_fee_rate = 0.005;
        different_fee.validate().unwrap();
        assert_ne!(
            different_fee.identity_sha256().unwrap(),
            contract.identity_sha256().unwrap()
        );
        assert!(!String::from_utf8_lossy(&bytes).contains("feature_plan_canonical_bytes"));
        let compact: CanonicalTrendbarResearchExecutionContractRefV1 =
            serde_json::from_slice(&bytes).unwrap();
        let restored = compact.attach(&receipt).unwrap();
        assert_eq!(restored, contract);
        assert_eq!(
            restored.identity_sha256().unwrap(),
            contract.identity_sha256().unwrap()
        );
        let mut changed = compact.clone();
        changed.screening_costs.full_spread_pips_assumption += 0.1;
        assert!(changed.attach(&receipt).is_err());
        let mut changed = compact.clone();
        changed.input_receipt_sha256 = "0".repeat(64);
        assert!(changed.attach(&receipt).is_err());
        let mut changed = compact;
        changed.contract_schema_version += 1;
        assert!(changed.attach(&receipt).is_err());
    }

    #[test]
    fn screening_cost_envelope_v2_counts_two_fill_sides() {
        let costs = CanonicalTrendbarScreeningCostEnvelopeV2::new(1.5, 0.5, 7.0)
            .expect("valid screening assumptions");

        assert_eq!(
            costs
                .screening_spread_and_slippage_round_trip_pips()
                .to_bits(),
            2.5_f64.to_bits()
        );
        assert_eq!(
            costs.round_trip_commission_account_per_lot().to_bits(),
            14.0_f64.to_bits()
        );
        assert_eq!(
            costs.screening_round_trip_cost_pips(10.0).to_bits(),
            3.9_f64.to_bits()
        );
    }

    #[test]
    fn screening_cost_envelope_v2_rejects_legacy_v1_semantics() {
        let legacy = serde_json::json!({
            "schema_version": 1,
            "spread_pips": 2.0,
            "round_trip_commission_per_trade": 14.0
        });
        assert!(
            serde_json::from_value::<CanonicalTrendbarScreeningCostEnvelopeV2>(legacy).is_err()
        );
    }

    #[test]
    fn effective_contract_fee_never_infers_currency_for_an_ambiguous_symbol() {
        assert_eq!(
            effective_contract_conversion_fee_rate("EURUSD", "USD", 0.01).unwrap(),
            0.0
        );
        assert_eq!(
            effective_contract_conversion_fee_rate("EURUSD", "EUR", 0.01).unwrap(),
            0.01
        );
        assert!(effective_contract_conversion_fee_rate("EURUSD.pro", "USD", 0.01).is_err());
        assert!(effective_contract_conversion_fee_rate("EURUSD", "USD", f64::NAN).is_err());
    }
}
