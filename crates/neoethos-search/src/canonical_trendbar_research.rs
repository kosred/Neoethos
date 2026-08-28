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
            pnl_conversion_fee_rate: assumptions.pnl_conversion_fee_rate,
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

    pub fn validate(&self) -> Result<()> {
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

#[derive(Debug, Clone)]
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
}
