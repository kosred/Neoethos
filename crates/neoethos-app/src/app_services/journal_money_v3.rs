//! Durable, money-first closed-position journal V3.
//!
//! This boundary persists every verified broker close deal as an immutable
//! file. A closed-position receipt exists only after those files cover the
//! exact raw entry volume and an account reconcile snapshot proves that the
//! position is absent. The bridge's bounded `recent_deals` window is not a
//! completeness proof and intentionally has no adapter into this module.
//!
//! Legacy journal V1/V2 rows remain useful for display. They are explicitly
//! classified as display-only, non-monetary, and non-promotable; no conversion
//! from their scalar `f64` fields into a V3 receipt exists here.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Datelike, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::app_services::broker_deal_economics::{
    BROKER_DEAL_MONEY_SCHEMA_VERSION_V1, BrokerDealMoneyEvidenceV1, BrokerPnlConversionFeeV1,
    BrokerSymbolVolumeScaleEvidenceV1,
};
use crate::app_services::ctrader_account::{
    CTraderAccountRuntimeSnapshot, CTraderPositionSnapshot,
};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;

pub const CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3: u32 = 3;

const JOURNAL_DIRECTORY: &str = "journal";
const MONEY_V3_DIRECTORY: &str = "money-v3";
const POSITIONS_DIRECTORY: &str = "positions";
const DEALS_DIRECTORY: &str = "deals";
const MANIFEST_FILE: &str = "manifest.v3.json";
const RECEIPT_FILE: &str = "receipt.v3.json";
const MAX_MONEY_DIGITS: u32 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JournalMoneyErrorCodeV3 {
    InvalidIdentity,
    InvalidEnvironment,
    InvalidCurrency,
    InvalidMoneyDigits,
    InvalidMoney,
    InvalidTimestamp,
    IdentityMismatch,
    DuplicateDealIdentityMismatch,
    FilledVolumeMismatch,
    PositionStillOpen,
    MissingDealEvidence,
    AlreadyFinalized,
    UnsupportedSchemaVersion,
    CorruptLedger,
    Io,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalMoneyErrorV3 {
    code: JournalMoneyErrorCodeV3,
    detail: String,
}

impl JournalMoneyErrorV3 {
    fn new(code: JournalMoneyErrorCodeV3, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> JournalMoneyErrorCodeV3 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for JournalMoneyErrorV3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "journal money V3: {}", self.detail)
    }
}

impl Error for JournalMoneyErrorV3 {}

fn journal_error(code: JournalMoneyErrorCodeV3, detail: impl Into<String>) -> JournalMoneyErrorV3 {
    JournalMoneyErrorV3::new(code, detail)
}

fn io_error(action: &str, path: &Path, error: std::io::Error) -> JournalMoneyErrorV3 {
    journal_error(
        JournalMoneyErrorCodeV3::Io,
        format!("{action} {}: {error}", path.display()),
    )
}

fn canonical_environment(value: &str) -> Result<&str, JournalMoneyErrorV3> {
    match value {
        "demo" => Ok("demo"),
        "live" => Ok("live"),
        _ => Err(journal_error(
            JournalMoneyErrorCodeV3::InvalidEnvironment,
            "environment must be the canonical lowercase value `demo` or `live`",
        )),
    }
}

fn runtime_environment(value: CTraderEnvironment) -> &'static str {
    match value {
        CTraderEnvironment::Demo => "demo",
        CTraderEnvironment::Live => "live",
    }
}

fn validate_currency(value: &str) -> Result<(), JournalMoneyErrorV3> {
    if value.len() != 3 || !value.bytes().all(|byte| byte.is_ascii_uppercase()) {
        return Err(journal_error(
            JournalMoneyErrorCodeV3::InvalidCurrency,
            "account currency must be exactly three uppercase ASCII letters",
        ));
    }
    Ok(())
}

fn validate_hash(label: &str, value: &str) -> Result<(), JournalMoneyErrorV3> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(journal_error(
            JournalMoneyErrorCodeV3::CorruptLedger,
            format!("{label} is not a lowercase SHA-256 identity"),
        ));
    }
    Ok(())
}

fn append_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn append_optional_i64(bytes: &mut Vec<u8>, value: Option<i64>) {
    match value {
        Some(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        None => bytes.push(0),
    }
}

fn sha256_hex(domain: &str, payload: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update(b"\n");
    hasher.update(payload);
    format!("{:x}", hasher.finalize())
}

fn checked_add(left: i64, right: i64, label: &str) -> Result<i64, JournalMoneyErrorV3> {
    left.checked_add(right).ok_or_else(|| {
        journal_error(
            JournalMoneyErrorCodeV3::InvalidMoney,
            format!("{label} overflows signed broker raw money"),
        )
    })
}

fn now_unix_ms() -> Result<i64, JournalMoneyErrorV3> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            journal_error(
                JournalMoneyErrorCodeV3::InvalidTimestamp,
                format!("system clock is before Unix epoch: {error}"),
            )
        })?
        .as_millis();
    i64::try_from(millis).map_err(|_| {
        journal_error(
            JournalMoneyErrorCodeV3::InvalidTimestamp,
            "current Unix timestamp does not fit i64 milliseconds",
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JournalMoneyArtifactClassV3 {
    VerifiedBrokerDealMoney,
    DisplayOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JournalMonetaryAuthorityV3 {
    VerifiedBrokerDealComponents,
    Refused,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JournalMoneyPromotionEligibilityV3 {
    EligibleForRiskAndPromotion,
    NotPromotionEligible,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyJournalDispositionV3 {
    artifact_class: JournalMoneyArtifactClassV3,
    monetary_authority: JournalMonetaryAuthorityV3,
    promotion_eligibility: JournalMoneyPromotionEligibilityV3,
}

impl LegacyJournalDispositionV3 {
    pub const fn artifact_class(&self) -> JournalMoneyArtifactClassV3 {
        self.artifact_class
    }

    pub const fn monetary_authority(&self) -> JournalMonetaryAuthorityV3 {
        self.monetary_authority
    }

    pub const fn promotion_eligibility(&self) -> JournalMoneyPromotionEligibilityV3 {
        self.promotion_eligibility
    }
}

pub fn classify_legacy_journal_v1_v2(
    schema_version: u32,
) -> Result<LegacyJournalDispositionV3, JournalMoneyErrorV3> {
    if !matches!(schema_version, 1 | 2) {
        return Err(journal_error(
            JournalMoneyErrorCodeV3::UnsupportedSchemaVersion,
            format!("schema version {schema_version} is not legacy journal V1/V2"),
        ));
    }
    Ok(LegacyJournalDispositionV3 {
        artifact_class: JournalMoneyArtifactClassV3::DisplayOnly,
        monetary_authority: JournalMonetaryAuthorityV3::Refused,
        promotion_eligibility: JournalMoneyPromotionEligibilityV3::NotPromotionEligible,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrokerPositionLifecycleWireV3 {
    pub environment: String,
    pub account_id: i64,
    pub position_id: i64,
    pub symbol_id: i64,
    pub symbol_name: String,
    pub position_side: String,
    pub account_currency: String,
    pub money_digits: u32,
    pub expected_entry_filled_volume_raw_centi_units: i64,
    pub entry_timestamp_ms: i64,
    pub entry_price: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerPositionLifecycleIdentityV3 {
    schema_version: u32,
    environment: String,
    account_id: i64,
    position_id: i64,
    symbol_id: i64,
    symbol_name: String,
    position_side: String,
    account_currency: String,
    money_digits: u32,
    expected_entry_filled_volume_raw_centi_units: i64,
    lot_size_raw_centi_units: i64,
    volume_scale_identity_sha256: String,
    entry_timestamp_ms: i64,
    entry_price: f64,
    lifecycle_identity_sha256: String,
}

impl BrokerPositionLifecycleIdentityV3 {
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn environment(&self) -> &str {
        &self.environment
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn position_id(&self) -> i64 {
        self.position_id
    }

    pub const fn symbol_id(&self) -> i64 {
        self.symbol_id
    }

    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }

    pub fn position_side(&self) -> &str {
        &self.position_side
    }

    pub fn account_currency(&self) -> &str {
        &self.account_currency
    }

    pub const fn money_digits(&self) -> u32 {
        self.money_digits
    }

    pub const fn expected_entry_filled_volume_raw_centi_units(&self) -> i64 {
        self.expected_entry_filled_volume_raw_centi_units
    }

    pub const fn lot_size_raw_centi_units(&self) -> i64 {
        self.lot_size_raw_centi_units
    }

    pub fn volume_scale_identity_sha256(&self) -> &str {
        &self.volume_scale_identity_sha256
    }

    pub const fn entry_timestamp_ms(&self) -> i64 {
        self.entry_timestamp_ms
    }

    pub const fn entry_price(&self) -> f64 {
        self.entry_price
    }

    pub fn lifecycle_identity_sha256(&self) -> &str {
        &self.lifecycle_identity_sha256
    }

    fn canonical_hash(&self) -> String {
        lifecycle_hash(
            &self.environment,
            self.account_id,
            self.position_id,
            self.symbol_id,
            &self.symbol_name,
            &self.position_side,
            &self.account_currency,
            self.money_digits,
            self.expected_entry_filled_volume_raw_centi_units,
            self.lot_size_raw_centi_units,
            &self.volume_scale_identity_sha256,
            self.entry_timestamp_ms,
            self.entry_price,
        )
    }

    fn validate(&self) -> Result<(), JournalMoneyErrorV3> {
        if self.schema_version != CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3 {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::UnsupportedSchemaVersion,
                format!("lifecycle schema {} is not V3", self.schema_version),
            ));
        }
        canonical_environment(&self.environment)?;
        validate_currency(&self.account_currency)?;
        if self.account_id <= 0
            || self.position_id <= 0
            || self.symbol_id <= 0
            || self.symbol_name.trim().is_empty()
            || self.symbol_name.trim() != self.symbol_name
            || !matches!(self.position_side.as_str(), "BUY" | "SELL")
            || self.money_digits > MAX_MONEY_DIGITS
            || self.expected_entry_filled_volume_raw_centi_units <= 0
            || self.lot_size_raw_centi_units <= 0
            || self.entry_timestamp_ms <= 0
            || !self.entry_price.is_finite()
            || self.entry_price <= 0.0
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::InvalidIdentity,
                "closed-position lifecycle identity is incomplete or invalid",
            ));
        }
        validate_hash("volume-scale identity", &self.volume_scale_identity_sha256)?;
        validate_hash("lifecycle identity", &self.lifecycle_identity_sha256)?;
        if self.canonical_hash() != self.lifecycle_identity_sha256 {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                "lifecycle identity hash does not match its exact fields",
            ));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn lifecycle_hash(
    environment: &str,
    account_id: i64,
    position_id: i64,
    symbol_id: i64,
    symbol_name: &str,
    position_side: &str,
    account_currency: &str,
    money_digits: u32,
    expected_entry_filled_volume_raw_centi_units: i64,
    lot_size_raw_centi_units: i64,
    volume_scale_identity_sha256: &str,
    entry_timestamp_ms: i64,
    entry_price: f64,
) -> String {
    let mut payload = Vec::new();
    payload.extend_from_slice(&CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3.to_be_bytes());
    append_string(&mut payload, environment);
    payload.extend_from_slice(&account_id.to_be_bytes());
    payload.extend_from_slice(&position_id.to_be_bytes());
    payload.extend_from_slice(&symbol_id.to_be_bytes());
    append_string(&mut payload, symbol_name);
    append_string(&mut payload, position_side);
    append_string(&mut payload, account_currency);
    payload.extend_from_slice(&money_digits.to_be_bytes());
    payload.extend_from_slice(&expected_entry_filled_volume_raw_centi_units.to_be_bytes());
    payload.extend_from_slice(&lot_size_raw_centi_units.to_be_bytes());
    append_string(&mut payload, volume_scale_identity_sha256);
    payload.extend_from_slice(&entry_timestamp_ms.to_be_bytes());
    payload.extend_from_slice(&entry_price.to_bits().to_be_bytes());
    sha256_hex("neoethos-broker-position-lifecycle-v3", &payload)
}

pub fn build_broker_position_lifecycle_identity_v3(
    wire: &BrokerPositionLifecycleWireV3,
    volume_scale: &BrokerSymbolVolumeScaleEvidenceV1,
) -> Result<BrokerPositionLifecycleIdentityV3, JournalMoneyErrorV3> {
    canonical_environment(&wire.environment)?;
    validate_currency(&wire.account_currency)?;
    if wire.environment != volume_scale.environment()
        || wire.account_id != volume_scale.account_id()
        || wire.symbol_id != volume_scale.symbol_id()
        || wire.symbol_name != volume_scale.symbol_name()
    {
        return Err(journal_error(
            JournalMoneyErrorCodeV3::IdentityMismatch,
            "position lifecycle differs from exact broker lot-size evidence",
        ));
    }
    if wire.account_id <= 0
        || wire.position_id <= 0
        || wire.symbol_id <= 0
        || wire.symbol_name.trim().is_empty()
        || wire.symbol_name.trim() != wire.symbol_name
        || !matches!(wire.position_side.as_str(), "BUY" | "SELL")
        || wire.money_digits > MAX_MONEY_DIGITS
        || wire.expected_entry_filled_volume_raw_centi_units <= 0
        || wire.entry_timestamp_ms <= 0
        || !wire.entry_price.is_finite()
        || wire.entry_price <= 0.0
    {
        return Err(journal_error(
            JournalMoneyErrorCodeV3::InvalidIdentity,
            "position lifecycle wire evidence is incomplete or invalid",
        ));
    }
    let lot_size_raw_centi_units = volume_scale.lot_size_raw_centi_units();
    let volume_scale_identity_sha256 = volume_scale.volume_scale_identity_sha256().to_string();
    validate_hash("volume-scale identity", &volume_scale_identity_sha256)?;
    let lifecycle_identity_sha256 = lifecycle_hash(
        &wire.environment,
        wire.account_id,
        wire.position_id,
        wire.symbol_id,
        &wire.symbol_name,
        &wire.position_side,
        &wire.account_currency,
        wire.money_digits,
        wire.expected_entry_filled_volume_raw_centi_units,
        lot_size_raw_centi_units,
        &volume_scale_identity_sha256,
        wire.entry_timestamp_ms,
        wire.entry_price,
    );
    let identity = BrokerPositionLifecycleIdentityV3 {
        schema_version: CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3,
        environment: wire.environment.clone(),
        account_id: wire.account_id,
        position_id: wire.position_id,
        symbol_id: wire.symbol_id,
        symbol_name: wire.symbol_name.clone(),
        position_side: wire.position_side.clone(),
        account_currency: wire.account_currency.clone(),
        money_digits: wire.money_digits,
        expected_entry_filled_volume_raw_centi_units: wire
            .expected_entry_filled_volume_raw_centi_units,
        lot_size_raw_centi_units,
        volume_scale_identity_sha256,
        entry_timestamp_ms: wire.entry_timestamp_ms,
        entry_price: wire.entry_price,
        lifecycle_identity_sha256,
    };
    identity.validate()?;
    Ok(identity)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JournalPnlConversionFeeV3 {
    Charged { raw_scaled_signed: i64 },
    NotApplied,
}

impl JournalPnlConversionFeeV3 {
    const fn raw_scaled_signed(self) -> i64 {
        match self {
            Self::Charged { raw_scaled_signed } => raw_scaled_signed,
            Self::NotApplied => 0,
        }
    }

    const fn hash_tag(self) -> u8 {
        match self {
            Self::Charged { .. } => 1,
            Self::NotApplied => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableBrokerDealMoneyV3 {
    schema_version: u32,
    environment: String,
    account_id: i64,
    deal_id: i64,
    order_id: i64,
    position_id: i64,
    symbol_id: i64,
    symbol_name: String,
    trade_side: String,
    filled_volume_raw_centi_units: i64,
    execution_timestamp_ms: i64,
    execution_price: f64,
    entry_price: f64,
    money_digits: u32,
    account_currency: String,
    gross_profit_raw_scaled: i64,
    commission_raw_scaled_signed: i64,
    swap_raw_scaled_signed: i64,
    pnl_conversion_fee: JournalPnlConversionFeeV3,
    component_sum_raw_scaled: i64,
    lot_size_raw_centi_units: i64,
    volume_scale_identity_sha256: String,
    deal_identity_sha256: String,
    durable_fill_identity_sha256: String,
}

impl DurableBrokerDealMoneyV3 {
    pub const fn deal_id(&self) -> i64 {
        self.deal_id
    }

    pub const fn order_id(&self) -> i64 {
        self.order_id
    }

    pub const fn filled_volume_raw_centi_units(&self) -> i64 {
        self.filled_volume_raw_centi_units
    }

    pub const fn execution_timestamp_ms(&self) -> i64 {
        self.execution_timestamp_ms
    }

    pub const fn execution_price(&self) -> f64 {
        self.execution_price
    }

    pub const fn entry_price(&self) -> f64 {
        self.entry_price
    }

    pub const fn money_digits(&self) -> u32 {
        self.money_digits
    }

    pub fn account_currency(&self) -> &str {
        &self.account_currency
    }

    pub const fn gross_profit_raw_scaled(&self) -> i64 {
        self.gross_profit_raw_scaled
    }

    pub const fn commission_raw_scaled_signed(&self) -> i64 {
        self.commission_raw_scaled_signed
    }

    pub const fn swap_raw_scaled_signed(&self) -> i64 {
        self.swap_raw_scaled_signed
    }

    pub const fn pnl_conversion_fee(&self) -> JournalPnlConversionFeeV3 {
        self.pnl_conversion_fee
    }

    pub const fn component_sum_raw_scaled(&self) -> i64 {
        self.component_sum_raw_scaled
    }

    pub fn volume_scale_identity_sha256(&self) -> &str {
        &self.volume_scale_identity_sha256
    }

    pub fn deal_identity_sha256(&self) -> &str {
        &self.deal_identity_sha256
    }

    pub fn durable_fill_identity_sha256(&self) -> &str {
        &self.durable_fill_identity_sha256
    }

    fn from_broker_evidence(
        lifecycle: &BrokerPositionLifecycleIdentityV3,
        fill: &BrokerDealMoneyEvidenceV1,
    ) -> Result<Self, JournalMoneyErrorV3> {
        lifecycle.validate()?;
        if fill.schema_version() != BROKER_DEAL_MONEY_SCHEMA_VERSION_V1
            || fill.environment() != lifecycle.environment
            || fill.account_id() != lifecycle.account_id
            || fill.position_id() != lifecycle.position_id
            || fill.symbol_id() != lifecycle.symbol_id
            || fill.symbol_name() != lifecycle.symbol_name
            || fill.account_currency() != lifecycle.account_currency
            || fill.money_digits() != lifecycle.money_digits
            || fill.volume_scale_identity_sha256() != lifecycle.volume_scale_identity_sha256
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::IdentityMismatch,
                "broker close deal differs from the position lifecycle identity",
            ));
        }
        let expected_close_side = match lifecycle.position_side.as_str() {
            "BUY" => "SELL",
            "SELL" => "BUY",
            _ => {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::InvalidIdentity,
                    "position side is not canonical",
                ));
            }
        };
        let expected_contract_units = lifecycle.lot_size_raw_centi_units as f64 / 100.0;
        let expected_lots =
            fill.filled_volume_raw_centi_units() as f64 / lifecycle.lot_size_raw_centi_units as f64;
        if fill.trade_side() != expected_close_side
            || fill.filled_volume_raw_centi_units() <= 0
            || fill.execution_timestamp_ms() < lifecycle.entry_timestamp_ms
            || fill.entry_price().to_bits() != lifecycle.entry_price.to_bits()
            || fill.contract_units_per_lot().to_bits() != expected_contract_units.to_bits()
            || fill.actual_filled_lots().to_bits() != expected_lots.to_bits()
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::IdentityMismatch,
                "broker close deal side, volume, price, or time differs from the lifecycle",
            ));
        }
        for (label, currency) in [
            (
                "gross profit",
                fill.gross_profit_account_currency().currency(),
            ),
            (
                "commission",
                fill.commission_account_currency_signed().currency(),
            ),
            ("swap", fill.swap_account_currency_signed().currency()),
            (
                "conversion fee",
                fill.pnl_conversion_fee_account_currency_signed().currency(),
            ),
            (
                "component sum",
                fill.component_sum_account_currency().currency(),
            ),
        ] {
            if currency != lifecycle.account_currency {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::IdentityMismatch,
                    format!("{label} account currency differs from the lifecycle"),
                ));
            }
        }
        validate_hash("broker deal identity", fill.deal_identity_sha256())?;
        validate_hash(
            "broker volume-scale identity",
            fill.volume_scale_identity_sha256(),
        )?;
        let pnl_conversion_fee = match fill.pnl_conversion_fee_state() {
            BrokerPnlConversionFeeV1::Charged { raw_scaled_signed } => {
                JournalPnlConversionFeeV3::Charged { raw_scaled_signed }
            }
            BrokerPnlConversionFeeV1::NotApplied => JournalPnlConversionFeeV3::NotApplied,
        };
        let component_sum_raw_scaled = checked_add(
            checked_add(
                checked_add(
                    fill.gross_profit_raw_scaled(),
                    fill.commission_raw_scaled_signed(),
                    "deal gross plus commission",
                )?,
                fill.swap_raw_scaled_signed(),
                "deal gross, commission, and swap",
            )?,
            pnl_conversion_fee.raw_scaled_signed(),
            "deal component sum",
        )?;
        if component_sum_raw_scaled != fill.component_sum_raw_scaled() {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::InvalidMoney,
                "broker deal component sum differs from its exact signed parts",
            ));
        }
        let mut durable = Self {
            schema_version: CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3,
            environment: fill.environment().to_string(),
            account_id: fill.account_id(),
            deal_id: fill.deal_id(),
            order_id: fill.order_id(),
            position_id: fill.position_id(),
            symbol_id: fill.symbol_id(),
            symbol_name: fill.symbol_name().to_string(),
            trade_side: fill.trade_side().to_string(),
            filled_volume_raw_centi_units: fill.filled_volume_raw_centi_units(),
            execution_timestamp_ms: fill.execution_timestamp_ms(),
            execution_price: fill.execution_price(),
            entry_price: fill.entry_price(),
            money_digits: fill.money_digits(),
            account_currency: fill.account_currency().to_string(),
            gross_profit_raw_scaled: fill.gross_profit_raw_scaled(),
            commission_raw_scaled_signed: fill.commission_raw_scaled_signed(),
            swap_raw_scaled_signed: fill.swap_raw_scaled_signed(),
            pnl_conversion_fee,
            component_sum_raw_scaled,
            lot_size_raw_centi_units: lifecycle.lot_size_raw_centi_units,
            volume_scale_identity_sha256: fill.volume_scale_identity_sha256().to_string(),
            deal_identity_sha256: fill.deal_identity_sha256().to_string(),
            durable_fill_identity_sha256: String::new(),
        };
        durable.durable_fill_identity_sha256 = durable.canonical_hash();
        durable.validate_against(lifecycle)?;
        Ok(durable)
    }

    fn canonical_hash(&self) -> String {
        let mut payload = Vec::new();
        payload.extend_from_slice(&self.schema_version.to_be_bytes());
        append_string(&mut payload, &self.environment);
        payload.extend_from_slice(&self.account_id.to_be_bytes());
        payload.extend_from_slice(&self.deal_id.to_be_bytes());
        payload.extend_from_slice(&self.order_id.to_be_bytes());
        payload.extend_from_slice(&self.position_id.to_be_bytes());
        payload.extend_from_slice(&self.symbol_id.to_be_bytes());
        append_string(&mut payload, &self.symbol_name);
        append_string(&mut payload, &self.trade_side);
        payload.extend_from_slice(&self.filled_volume_raw_centi_units.to_be_bytes());
        payload.extend_from_slice(&self.execution_timestamp_ms.to_be_bytes());
        payload.extend_from_slice(&self.execution_price.to_bits().to_be_bytes());
        payload.extend_from_slice(&self.entry_price.to_bits().to_be_bytes());
        payload.extend_from_slice(&self.money_digits.to_be_bytes());
        append_string(&mut payload, &self.account_currency);
        payload.extend_from_slice(&self.gross_profit_raw_scaled.to_be_bytes());
        payload.extend_from_slice(&self.commission_raw_scaled_signed.to_be_bytes());
        payload.extend_from_slice(&self.swap_raw_scaled_signed.to_be_bytes());
        payload.push(self.pnl_conversion_fee.hash_tag());
        payload.extend_from_slice(&self.pnl_conversion_fee.raw_scaled_signed().to_be_bytes());
        payload.extend_from_slice(&self.component_sum_raw_scaled.to_be_bytes());
        payload.extend_from_slice(&self.lot_size_raw_centi_units.to_be_bytes());
        append_string(&mut payload, &self.volume_scale_identity_sha256);
        append_string(&mut payload, &self.deal_identity_sha256);
        sha256_hex("neoethos-durable-broker-deal-money-v3", &payload)
    }

    fn validate_against(
        &self,
        lifecycle: &BrokerPositionLifecycleIdentityV3,
    ) -> Result<(), JournalMoneyErrorV3> {
        if self.schema_version != CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3 {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::UnsupportedSchemaVersion,
                format!("durable deal schema {} is not V3", self.schema_version),
            ));
        }
        if self.environment != lifecycle.environment
            || self.account_id != lifecycle.account_id
            || self.position_id != lifecycle.position_id
            || self.symbol_id != lifecycle.symbol_id
            || self.symbol_name != lifecycle.symbol_name
            || self.account_currency != lifecycle.account_currency
            || self.money_digits != lifecycle.money_digits
            || self.lot_size_raw_centi_units != lifecycle.lot_size_raw_centi_units
            || self.volume_scale_identity_sha256 != lifecycle.volume_scale_identity_sha256
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::IdentityMismatch,
                "persisted deal scope differs from its lifecycle manifest",
            ));
        }
        let expected_close_side = if lifecycle.position_side == "BUY" {
            "SELL"
        } else {
            "BUY"
        };
        if self.deal_id <= 0
            || self.order_id <= 0
            || self.trade_side != expected_close_side
            || self.filled_volume_raw_centi_units <= 0
            || self.execution_timestamp_ms < lifecycle.entry_timestamp_ms
            || !self.execution_price.is_finite()
            || self.execution_price <= 0.0
            || self.entry_price.to_bits() != lifecycle.entry_price.to_bits()
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                "persisted deal identity, side, price, volume, or timestamp is invalid",
            ));
        }
        let expected_sum = checked_add(
            checked_add(
                checked_add(
                    self.gross_profit_raw_scaled,
                    self.commission_raw_scaled_signed,
                    "persisted deal gross plus commission",
                )?,
                self.swap_raw_scaled_signed,
                "persisted deal gross, commission, and swap",
            )?,
            self.pnl_conversion_fee.raw_scaled_signed(),
            "persisted deal component sum",
        )?;
        if expected_sum != self.component_sum_raw_scaled {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                "persisted deal component sum differs from its signed parts",
            ));
        }
        validate_hash("broker deal identity", &self.deal_identity_sha256)?;
        validate_hash("durable fill identity", &self.durable_fill_identity_sha256)?;
        if self.canonical_hash() != self.durable_fill_identity_sha256 {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                "durable fill identity hash does not match its exact fields",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BrokerOpenPositionIdentityV3 {
    position_id: i64,
    symbol_id: i64,
    trade_side: String,
    volume_bits: u64,
    open_timestamp_ms: Option<i64>,
    price_bits: Option<u64>,
}

impl BrokerOpenPositionIdentityV3 {
    fn from_runtime(position: &CTraderPositionSnapshot) -> Result<Self, JournalMoneyErrorV3> {
        if position.position_id <= 0
            || position.symbol_id <= 0
            || !matches!(position.trade_side.as_str(), "BUY" | "SELL")
            || !position.volume.is_finite()
            || position.volume <= 0.0
            || position
                .open_timestamp_ms
                .is_some_and(|timestamp| timestamp <= 0)
            || position
                .price
                .is_some_and(|price| !price.is_finite() || price <= 0.0)
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::InvalidIdentity,
                "account reconcile contains an invalid open-position identity",
            ));
        }
        Ok(Self {
            position_id: position.position_id,
            symbol_id: position.symbol_id,
            trade_side: position.trade_side.clone(),
            volume_bits: position.volume.to_bits(),
            open_timestamp_ms: position.open_timestamp_ms,
            price_bits: position.price.map(f64::to_bits),
        })
    }

    fn append_hash_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.position_id.to_be_bytes());
        bytes.extend_from_slice(&self.symbol_id.to_be_bytes());
        append_string(bytes, &self.trade_side);
        bytes.extend_from_slice(&self.volume_bits.to_be_bytes());
        append_optional_i64(bytes, self.open_timestamp_ms);
        match self.price_bits {
            Some(bits) => {
                bytes.push(1);
                bytes.extend_from_slice(&bits.to_be_bytes());
            }
            None => bytes.push(0),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerFlatReconcileEvidenceV3 {
    schema_version: u32,
    environment: String,
    account_id: i64,
    account_currency: String,
    money_digits: u32,
    target_position_id: i64,
    observed_at_unix_ms: i64,
    open_positions: Vec<BrokerOpenPositionIdentityV3>,
    runtime_snapshot_identity_sha256: String,
}

impl BrokerFlatReconcileEvidenceV3 {
    pub fn from_account_runtime(
        lifecycle: &BrokerPositionLifecycleIdentityV3,
        runtime: &CTraderAccountRuntimeSnapshot,
    ) -> Result<Self, JournalMoneyErrorV3> {
        lifecycle.validate()?;
        let environment = runtime_environment(runtime.environment);
        if environment != lifecycle.environment
            || runtime.trader.account_id != lifecycle.account_id
            || runtime.reconcile.account_id != lifecycle.account_id
            || runtime.deposit_asset_name != lifecycle.account_currency
            || runtime.trader.money_digits != lifecycle.money_digits
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::IdentityMismatch,
                "account reconcile environment, account, currency, or money scale differs from the lifecycle",
            ));
        }
        // `runtime.reconcile.positions` is the account-wide broker state used
        // to prove target absence; bounded recent deal history is never used.
        let mut open_positions = runtime
            .reconcile
            .positions
            .iter()
            .map(BrokerOpenPositionIdentityV3::from_runtime)
            .collect::<Result<Vec<_>, _>>()?;
        open_positions.sort_by_key(|position| position.position_id);
        let mut unique_positions = BTreeSet::new();
        for position in &open_positions {
            if !unique_positions.insert(position.position_id) {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::CorruptLedger,
                    "account reconcile repeats an open position id",
                ));
            }
            if position.position_id == lifecycle.position_id {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::PositionStillOpen,
                    "target position is still present in the broker reconcile snapshot",
                ));
            }
        }
        let observed_at_unix_ms = now_unix_ms()?;
        let runtime_snapshot_identity_sha256 = flat_snapshot_hash(
            environment,
            lifecycle.account_id,
            &lifecycle.account_currency,
            lifecycle.money_digits,
            lifecycle.position_id,
            observed_at_unix_ms,
            &open_positions,
        );
        let evidence = Self {
            schema_version: CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3,
            environment: environment.to_string(),
            account_id: lifecycle.account_id,
            account_currency: lifecycle.account_currency.clone(),
            money_digits: lifecycle.money_digits,
            target_position_id: lifecycle.position_id,
            observed_at_unix_ms,
            open_positions,
            runtime_snapshot_identity_sha256,
        };
        evidence.validate_against(lifecycle)?;
        Ok(evidence)
    }

    pub const fn observed_at_unix_ms(&self) -> i64 {
        self.observed_at_unix_ms
    }

    pub fn runtime_snapshot_identity_sha256(&self) -> &str {
        &self.runtime_snapshot_identity_sha256
    }

    fn canonical_hash(&self) -> String {
        flat_snapshot_hash(
            &self.environment,
            self.account_id,
            &self.account_currency,
            self.money_digits,
            self.target_position_id,
            self.observed_at_unix_ms,
            &self.open_positions,
        )
    }

    fn validate_against(
        &self,
        lifecycle: &BrokerPositionLifecycleIdentityV3,
    ) -> Result<(), JournalMoneyErrorV3> {
        if self.schema_version != CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3 {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::UnsupportedSchemaVersion,
                format!("flat reconcile schema {} is not V3", self.schema_version),
            ));
        }
        if self.environment != lifecycle.environment
            || self.account_id != lifecycle.account_id
            || self.account_currency != lifecycle.account_currency
            || self.money_digits != lifecycle.money_digits
            || self.target_position_id != lifecycle.position_id
            || self.observed_at_unix_ms <= 0
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::IdentityMismatch,
                "persisted flat reconcile evidence differs from the lifecycle",
            ));
        }
        let mut previous = None;
        for position in &self.open_positions {
            if position.position_id <= 0
                || position.symbol_id <= 0
                || !matches!(position.trade_side.as_str(), "BUY" | "SELL")
                || f64::from_bits(position.volume_bits) <= 0.0
                || !f64::from_bits(position.volume_bits).is_finite()
                || position.position_id == self.target_position_id
                || previous.is_some_and(|prior| position.position_id <= prior)
            {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::CorruptLedger,
                    "persisted open-position set is invalid, unsorted, duplicated, or contains the target",
                ));
            }
            previous = Some(position.position_id);
        }
        validate_hash(
            "runtime snapshot identity",
            &self.runtime_snapshot_identity_sha256,
        )?;
        if self.canonical_hash() != self.runtime_snapshot_identity_sha256 {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                "flat reconcile snapshot hash does not match its exact fields",
            ));
        }
        Ok(())
    }
}

fn flat_snapshot_hash(
    environment: &str,
    account_id: i64,
    account_currency: &str,
    money_digits: u32,
    target_position_id: i64,
    observed_at_unix_ms: i64,
    open_positions: &[BrokerOpenPositionIdentityV3],
) -> String {
    let mut payload = Vec::new();
    payload.extend_from_slice(&CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3.to_be_bytes());
    append_string(&mut payload, environment);
    payload.extend_from_slice(&account_id.to_be_bytes());
    append_string(&mut payload, account_currency);
    payload.extend_from_slice(&money_digits.to_be_bytes());
    payload.extend_from_slice(&target_position_id.to_be_bytes());
    payload.extend_from_slice(&observed_at_unix_ms.to_be_bytes());
    payload.extend_from_slice(&(open_positions.len() as u64).to_be_bytes());
    for position in open_positions {
        position.append_hash_bytes(&mut payload);
    }
    sha256_hex("neoethos-broker-flat-reconcile-v3", &payload)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClosedPositionJournalReceiptV3 {
    schema_version: u32,
    artifact_class: JournalMoneyArtifactClassV3,
    monetary_authority: JournalMonetaryAuthorityV3,
    promotion_eligibility: JournalMoneyPromotionEligibilityV3,
    lifecycle: BrokerPositionLifecycleIdentityV3,
    fills: Vec<DurableBrokerDealMoneyV3>,
    flat_reconcile_evidence: BrokerFlatReconcileEvidenceV3,
    gross_profit_raw_scaled: i64,
    commission_raw_scaled_signed: i64,
    swap_raw_scaled_signed: i64,
    pnl_conversion_fee_raw_scaled_signed: i64,
    component_sum_raw_scaled: i64,
    closed_filled_volume_raw_centi_units: i64,
    receipt_identity_sha256: String,
}

impl ClosedPositionJournalReceiptV3 {
    fn build(
        lifecycle: &BrokerPositionLifecycleIdentityV3,
        fills: Vec<DurableBrokerDealMoneyV3>,
        flat_reconcile_evidence: BrokerFlatReconcileEvidenceV3,
    ) -> Result<Self, JournalMoneyErrorV3> {
        let totals = DealTotalsV3::from_fills(&fills)?;
        if totals.closed_filled_volume_raw_centi_units
            != lifecycle.expected_entry_filled_volume_raw_centi_units
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::FilledVolumeMismatch,
                format!(
                    "verified close fills total {} raw centi-units; exact entry volume is {}",
                    totals.closed_filled_volume_raw_centi_units,
                    lifecycle.expected_entry_filled_volume_raw_centi_units
                ),
            ));
        }
        flat_reconcile_evidence.validate_against(lifecycle)?;
        let mut receipt = Self {
            schema_version: CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3,
            artifact_class: JournalMoneyArtifactClassV3::VerifiedBrokerDealMoney,
            monetary_authority: JournalMonetaryAuthorityV3::VerifiedBrokerDealComponents,
            promotion_eligibility: JournalMoneyPromotionEligibilityV3::EligibleForRiskAndPromotion,
            lifecycle: lifecycle.clone(),
            fills,
            flat_reconcile_evidence,
            gross_profit_raw_scaled: totals.gross_profit_raw_scaled,
            commission_raw_scaled_signed: totals.commission_raw_scaled_signed,
            swap_raw_scaled_signed: totals.swap_raw_scaled_signed,
            pnl_conversion_fee_raw_scaled_signed: totals.pnl_conversion_fee_raw_scaled_signed,
            component_sum_raw_scaled: totals.component_sum_raw_scaled,
            closed_filled_volume_raw_centi_units: totals.closed_filled_volume_raw_centi_units,
            receipt_identity_sha256: String::new(),
        };
        receipt.receipt_identity_sha256 = receipt.canonical_hash();
        receipt.validate()?;
        Ok(receipt)
    }

    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub const fn artifact_class(&self) -> JournalMoneyArtifactClassV3 {
        self.artifact_class
    }

    pub const fn monetary_authority(&self) -> JournalMonetaryAuthorityV3 {
        self.monetary_authority
    }

    pub const fn promotion_eligibility(&self) -> JournalMoneyPromotionEligibilityV3 {
        self.promotion_eligibility
    }

    pub fn lifecycle(&self) -> &BrokerPositionLifecycleIdentityV3 {
        &self.lifecycle
    }

    pub fn fills(&self) -> &[DurableBrokerDealMoneyV3] {
        &self.fills
    }

    pub fn flat_reconcile_evidence(&self) -> &BrokerFlatReconcileEvidenceV3 {
        &self.flat_reconcile_evidence
    }

    pub const fn gross_profit_raw_scaled(&self) -> i64 {
        self.gross_profit_raw_scaled
    }

    pub const fn commission_raw_scaled_signed(&self) -> i64 {
        self.commission_raw_scaled_signed
    }

    pub const fn swap_raw_scaled_signed(&self) -> i64 {
        self.swap_raw_scaled_signed
    }

    pub const fn pnl_conversion_fee_raw_scaled_signed(&self) -> i64 {
        self.pnl_conversion_fee_raw_scaled_signed
    }

    pub const fn component_sum_raw_scaled(&self) -> i64 {
        self.component_sum_raw_scaled
    }

    pub const fn closed_filled_volume_raw_centi_units(&self) -> i64 {
        self.closed_filled_volume_raw_centi_units
    }

    pub fn component_sum_account_currency(&self) -> f64 {
        self.component_sum_raw_scaled as f64 / 10.0_f64.powi(self.lifecycle.money_digits as i32)
    }

    pub fn receipt_identity_sha256(&self) -> &str {
        &self.receipt_identity_sha256
    }

    pub fn exit_timestamp_ms(&self) -> i64 {
        self.fills
            .iter()
            .map(|fill| fill.execution_timestamp_ms)
            .max()
            .unwrap_or(self.lifecycle.entry_timestamp_ms)
    }

    fn canonical_hash(&self) -> String {
        let mut payload = Vec::new();
        payload.extend_from_slice(&self.schema_version.to_be_bytes());
        payload.push(match self.artifact_class {
            JournalMoneyArtifactClassV3::VerifiedBrokerDealMoney => 1,
            JournalMoneyArtifactClassV3::DisplayOnly => 0,
        });
        payload.push(match self.monetary_authority {
            JournalMonetaryAuthorityV3::VerifiedBrokerDealComponents => 1,
            JournalMonetaryAuthorityV3::Refused => 0,
        });
        payload.push(match self.promotion_eligibility {
            JournalMoneyPromotionEligibilityV3::EligibleForRiskAndPromotion => 1,
            JournalMoneyPromotionEligibilityV3::NotPromotionEligible => 0,
        });
        append_string(&mut payload, &self.lifecycle.lifecycle_identity_sha256);
        payload.extend_from_slice(&(self.fills.len() as u64).to_be_bytes());
        for fill in &self.fills {
            append_string(&mut payload, &fill.durable_fill_identity_sha256);
        }
        append_string(
            &mut payload,
            &self
                .flat_reconcile_evidence
                .runtime_snapshot_identity_sha256,
        );
        payload.extend_from_slice(&self.gross_profit_raw_scaled.to_be_bytes());
        payload.extend_from_slice(&self.commission_raw_scaled_signed.to_be_bytes());
        payload.extend_from_slice(&self.swap_raw_scaled_signed.to_be_bytes());
        payload.extend_from_slice(&self.pnl_conversion_fee_raw_scaled_signed.to_be_bytes());
        payload.extend_from_slice(&self.component_sum_raw_scaled.to_be_bytes());
        payload.extend_from_slice(&self.closed_filled_volume_raw_centi_units.to_be_bytes());
        sha256_hex("neoethos-closed-position-journal-receipt-v3", &payload)
    }

    fn validate(&self) -> Result<(), JournalMoneyErrorV3> {
        if self.schema_version != CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3 {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::UnsupportedSchemaVersion,
                format!("closed receipt schema {} is not V3", self.schema_version),
            ));
        }
        if self.artifact_class != JournalMoneyArtifactClassV3::VerifiedBrokerDealMoney
            || self.monetary_authority != JournalMonetaryAuthorityV3::VerifiedBrokerDealComponents
            || self.promotion_eligibility
                != JournalMoneyPromotionEligibilityV3::EligibleForRiskAndPromotion
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                "closed receipt authority or eligibility markers are invalid",
            ));
        }
        self.lifecycle.validate()?;
        self.flat_reconcile_evidence
            .validate_against(&self.lifecycle)?;
        let totals = DealTotalsV3::from_fills_against(&self.fills, &self.lifecycle)?;
        if self.fills.is_empty()
            || totals.gross_profit_raw_scaled != self.gross_profit_raw_scaled
            || totals.commission_raw_scaled_signed != self.commission_raw_scaled_signed
            || totals.swap_raw_scaled_signed != self.swap_raw_scaled_signed
            || totals.pnl_conversion_fee_raw_scaled_signed
                != self.pnl_conversion_fee_raw_scaled_signed
            || totals.component_sum_raw_scaled != self.component_sum_raw_scaled
            || totals.closed_filled_volume_raw_centi_units
                != self.closed_filled_volume_raw_centi_units
            || totals.closed_filled_volume_raw_centi_units
                != self.lifecycle.expected_entry_filled_volume_raw_centi_units
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                "closed receipt totals differ from its exact durable fills",
            ));
        }
        validate_hash("closed receipt identity", &self.receipt_identity_sha256)?;
        if self.canonical_hash() != self.receipt_identity_sha256 {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                "closed receipt identity hash does not match its exact fields",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DealTotalsV3 {
    gross_profit_raw_scaled: i64,
    commission_raw_scaled_signed: i64,
    swap_raw_scaled_signed: i64,
    pnl_conversion_fee_raw_scaled_signed: i64,
    component_sum_raw_scaled: i64,
    closed_filled_volume_raw_centi_units: i64,
}

impl DealTotalsV3 {
    fn from_fills(fills: &[DurableBrokerDealMoneyV3]) -> Result<Self, JournalMoneyErrorV3> {
        let mut totals = Self::default();
        let mut deal_ids = BTreeSet::new();
        for fill in fills {
            if !deal_ids.insert(fill.deal_id) {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::CorruptLedger,
                    format!("durable deal {} occurs more than once", fill.deal_id),
                ));
            }
            totals.gross_profit_raw_scaled = checked_add(
                totals.gross_profit_raw_scaled,
                fill.gross_profit_raw_scaled,
                "position gross profit",
            )?;
            totals.commission_raw_scaled_signed = checked_add(
                totals.commission_raw_scaled_signed,
                fill.commission_raw_scaled_signed,
                "position commission",
            )?;
            totals.swap_raw_scaled_signed = checked_add(
                totals.swap_raw_scaled_signed,
                fill.swap_raw_scaled_signed,
                "position swap",
            )?;
            totals.pnl_conversion_fee_raw_scaled_signed = checked_add(
                totals.pnl_conversion_fee_raw_scaled_signed,
                fill.pnl_conversion_fee.raw_scaled_signed(),
                "position conversion fee",
            )?;
            totals.component_sum_raw_scaled = checked_add(
                totals.component_sum_raw_scaled,
                fill.component_sum_raw_scaled,
                "position component sum",
            )?;
            totals.closed_filled_volume_raw_centi_units = checked_add(
                totals.closed_filled_volume_raw_centi_units,
                fill.filled_volume_raw_centi_units,
                "position closed raw volume",
            )?;
        }
        Ok(totals)
    }

    fn from_fills_against(
        fills: &[DurableBrokerDealMoneyV3],
        lifecycle: &BrokerPositionLifecycleIdentityV3,
    ) -> Result<Self, JournalMoneyErrorV3> {
        let mut previous_deal = None;
        for fill in fills {
            fill.validate_against(lifecycle)?;
            if previous_deal.is_some_and(|previous| fill.deal_id <= previous) {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::CorruptLedger,
                    "closed receipt durable fills are not strictly sorted by deal id",
                ));
            }
            previous_deal = Some(fill.deal_id);
        }
        Self::from_fills(fills)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JournalDealObservationV3 {
    Added,
    Duplicate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JournalPositionStateV3 {
    deal_count: usize,
    closed_filled_volume_raw_centi_units: i64,
    component_sum_raw_scaled: i64,
    finalized: bool,
}

impl JournalPositionStateV3 {
    pub const fn deal_count(&self) -> usize {
        self.deal_count
    }

    pub const fn closed_filled_volume_raw_centi_units(&self) -> i64 {
        self.closed_filled_volume_raw_centi_units
    }

    pub const fn component_sum_raw_scaled(&self) -> i64 {
        self.component_sum_raw_scaled
    }

    pub const fn is_finalized(&self) -> bool {
        self.finalized
    }

    pub const fn is_empty(&self) -> bool {
        self.deal_count == 0 && !self.finalized
    }
}

#[derive(Clone, Debug)]
pub struct JournalMoneyV3Store {
    data_dir: PathBuf,
}

impl JournalMoneyV3Store {
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
        }
    }

    fn positions_root(&self) -> PathBuf {
        self.data_dir
            .join(JOURNAL_DIRECTORY)
            .join(MONEY_V3_DIRECTORY)
            .join(POSITIONS_DIRECTORY)
    }

    fn position_dir(&self, lifecycle: &BrokerPositionLifecycleIdentityV3) -> PathBuf {
        self.positions_root()
            .join(&lifecycle.lifecycle_identity_sha256)
    }

    fn manifest_path(&self, lifecycle: &BrokerPositionLifecycleIdentityV3) -> PathBuf {
        self.position_dir(lifecycle).join(MANIFEST_FILE)
    }

    fn deals_dir(&self, lifecycle: &BrokerPositionLifecycleIdentityV3) -> PathBuf {
        self.position_dir(lifecycle).join(DEALS_DIRECTORY)
    }

    fn deal_path(&self, lifecycle: &BrokerPositionLifecycleIdentityV3, deal_id: i64) -> PathBuf {
        self.deals_dir(lifecycle).join(format!("{deal_id}.v3.json"))
    }

    fn receipt_path(&self, lifecycle: &BrokerPositionLifecycleIdentityV3) -> PathBuf {
        self.position_dir(lifecycle).join(RECEIPT_FILE)
    }

    pub fn record_close_fill(
        &self,
        lifecycle: &BrokerPositionLifecycleIdentityV3,
        fill: &BrokerDealMoneyEvidenceV1,
    ) -> Result<JournalDealObservationV3, JournalMoneyErrorV3> {
        lifecycle.validate()?;
        let durable = DurableBrokerDealMoneyV3::from_broker_evidence(lifecycle, fill)?;
        let receipt_path = self.receipt_path(lifecycle);
        if receipt_path.exists() {
            let receipt: ClosedPositionJournalReceiptV3 = read_strict_json(&receipt_path)?;
            receipt.validate()?;
            return Err(journal_error(
                JournalMoneyErrorCodeV3::AlreadyFinalized,
                format!(
                    "position {} already has finalized receipt {}",
                    lifecycle.position_id, receipt.receipt_identity_sha256
                ),
            ));
        }
        self.ensure_manifest(lifecycle)?;
        let deal_path = self.deal_path(lifecycle, durable.deal_id);
        match persist_immutable_json(&deal_path, &durable)? {
            ImmutablePersistOutcomeV3::Added => Ok(JournalDealObservationV3::Added),
            ImmutablePersistOutcomeV3::AlreadyExists => {
                let existing: DurableBrokerDealMoneyV3 = read_strict_json(&deal_path)?;
                existing.validate_against(lifecycle)?;
                if existing.deal_id == durable.deal_id
                    && existing.deal_identity_sha256 == durable.deal_identity_sha256
                    && existing.durable_fill_identity_sha256 == durable.durable_fill_identity_sha256
                    && existing == durable
                {
                    Ok(JournalDealObservationV3::Duplicate)
                } else {
                    Err(journal_error(
                        JournalMoneyErrorCodeV3::DuplicateDealIdentityMismatch,
                        format!(
                            "deal {} already exists with a different immutable identity",
                            durable.deal_id
                        ),
                    ))
                }
            }
        }
    }

    pub fn position_state(
        &self,
        lifecycle: &BrokerPositionLifecycleIdentityV3,
    ) -> Result<JournalPositionStateV3, JournalMoneyErrorV3> {
        lifecycle.validate()?;
        let position_dir = self.position_dir(lifecycle);
        if !position_dir.exists() {
            return Ok(JournalPositionStateV3 {
                deal_count: 0,
                closed_filled_volume_raw_centi_units: 0,
                component_sum_raw_scaled: 0,
                finalized: false,
            });
        }
        self.load_and_match_manifest(lifecycle)?;
        validate_position_directory_shape(&position_dir)?;
        let fills = self.load_durable_fills(lifecycle)?;
        let totals = DealTotalsV3::from_fills_against(&fills, lifecycle)?;
        let receipt_path = self.receipt_path(lifecycle);
        let finalized = if receipt_path.exists() {
            let receipt: ClosedPositionJournalReceiptV3 = read_strict_json(&receipt_path)?;
            validate_receipt_against_disk(&receipt, lifecycle, &fills)?;
            true
        } else {
            false
        };
        Ok(JournalPositionStateV3 {
            deal_count: fills.len(),
            closed_filled_volume_raw_centi_units: totals.closed_filled_volume_raw_centi_units,
            component_sum_raw_scaled: totals.component_sum_raw_scaled,
            finalized,
        })
    }

    pub fn finalize_from_account_runtime(
        &self,
        lifecycle: &BrokerPositionLifecycleIdentityV3,
        runtime: &CTraderAccountRuntimeSnapshot,
    ) -> Result<ClosedPositionJournalReceiptV3, JournalMoneyErrorV3> {
        lifecycle.validate()?;
        let receipt_path = self.receipt_path(lifecycle);
        if receipt_path.exists() {
            let receipt: ClosedPositionJournalReceiptV3 = read_strict_json(&receipt_path)?;
            let fills = self.load_durable_fills(lifecycle)?;
            validate_receipt_against_disk(&receipt, lifecycle, &fills)?;
            return Ok(receipt);
        }
        self.load_and_match_manifest(lifecycle)?;
        let mut fills = self.load_durable_fills(lifecycle)?;
        if fills.is_empty() {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::MissingDealEvidence,
                "no durable broker close-deal evidence exists for this position",
            ));
        }
        fills.sort_by_key(|fill| fill.deal_id);
        let flat_reconcile_evidence =
            BrokerFlatReconcileEvidenceV3::from_account_runtime(lifecycle, runtime)?;
        let totals = DealTotalsV3::from_fills_against(&fills, lifecycle)?;
        if totals.closed_filled_volume_raw_centi_units
            != lifecycle.expected_entry_filled_volume_raw_centi_units
        {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::FilledVolumeMismatch,
                format!(
                    "durable close fills total {} raw centi-units; exact entry volume is {}",
                    totals.closed_filled_volume_raw_centi_units,
                    lifecycle.expected_entry_filled_volume_raw_centi_units
                ),
            ));
        }
        let candidate =
            ClosedPositionJournalReceiptV3::build(lifecycle, fills, flat_reconcile_evidence)?;
        match persist_immutable_json(&receipt_path, &candidate)? {
            ImmutablePersistOutcomeV3::Added => Ok(candidate),
            ImmutablePersistOutcomeV3::AlreadyExists => {
                let existing: ClosedPositionJournalReceiptV3 = read_strict_json(&receipt_path)?;
                let fills = self.load_durable_fills(lifecycle)?;
                validate_receipt_against_disk(&existing, lifecycle, &fills)?;
                if existing == candidate {
                    Ok(existing)
                } else {
                    Err(journal_error(
                        JournalMoneyErrorCodeV3::AlreadyFinalized,
                        "a different immutable receipt won the concurrent finalization race",
                    ))
                }
            }
        }
    }

    pub fn load_finalized_receipts_strict(
        &self,
    ) -> Result<FinalizedClosedPositionJournalV3, JournalMoneyErrorV3> {
        let positions_root = self.positions_root();
        if !positions_root.exists() {
            return Ok(FinalizedClosedPositionJournalV3 {
                receipts: Vec::new(),
            });
        }
        let metadata = fs::metadata(&positions_root)
            .map_err(|error| io_error("inspect journal positions root", &positions_root, error))?;
        if !metadata.is_dir() {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                format!(
                    "journal positions root {} is not a directory",
                    positions_root.display()
                ),
            ));
        }
        let mut position_dirs = Vec::new();
        for entry in fs::read_dir(&positions_root)
            .map_err(|error| io_error("read journal positions root", &positions_root, error))?
        {
            let entry = entry.map_err(|error| {
                io_error(
                    "read journal position directory entry",
                    &positions_root,
                    error,
                )
            })?;
            let file_type = entry.file_type().map_err(|error| {
                io_error(
                    "inspect journal position directory entry",
                    &entry.path(),
                    error,
                )
            })?;
            if !file_type.is_dir() {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::CorruptLedger,
                    format!(
                        "unexpected non-directory entry in V3 positions root: {}",
                        entry.path().display()
                    ),
                ));
            }
            position_dirs.push(entry.path());
        }
        position_dirs.sort();
        let mut receipts = Vec::new();
        for position_dir in position_dirs {
            validate_position_directory_shape(&position_dir)?;
            let manifest_path = position_dir.join(MANIFEST_FILE);
            let lifecycle: BrokerPositionLifecycleIdentityV3 = read_strict_json(&manifest_path)?;
            lifecycle.validate()?;
            if position_dir.file_name().and_then(|value| value.to_str())
                != Some(lifecycle.lifecycle_identity_sha256())
            {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::CorruptLedger,
                    format!(
                        "position directory {} differs from its lifecycle identity",
                        position_dir.display()
                    ),
                ));
            }
            let fills = self.load_durable_fills(&lifecycle)?;
            let receipt_path = position_dir.join(RECEIPT_FILE);
            if receipt_path.exists() {
                let receipt: ClosedPositionJournalReceiptV3 = read_strict_json(&receipt_path)?;
                validate_receipt_against_disk(&receipt, &lifecycle, &fills)?;
                receipts.push(receipt);
            } else {
                DealTotalsV3::from_fills_against(&fills, &lifecycle)?;
            }
        }
        receipts.sort_by(|left, right| {
            left.exit_timestamp_ms()
                .cmp(&right.exit_timestamp_ms())
                .then_with(|| {
                    left.lifecycle
                        .lifecycle_identity_sha256
                        .cmp(&right.lifecycle.lifecycle_identity_sha256)
                })
        });
        Ok(FinalizedClosedPositionJournalV3 { receipts })
    }

    fn ensure_manifest(
        &self,
        lifecycle: &BrokerPositionLifecycleIdentityV3,
    ) -> Result<(), JournalMoneyErrorV3> {
        let path = self.manifest_path(lifecycle);
        match persist_immutable_json(&path, lifecycle)? {
            ImmutablePersistOutcomeV3::Added => Ok(()),
            ImmutablePersistOutcomeV3::AlreadyExists => self.load_and_match_manifest(lifecycle),
        }
    }

    fn load_and_match_manifest(
        &self,
        lifecycle: &BrokerPositionLifecycleIdentityV3,
    ) -> Result<(), JournalMoneyErrorV3> {
        let path = self.manifest_path(lifecycle);
        if !path.exists() {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                format!("position ledger is missing manifest {}", path.display()),
            ));
        }
        let persisted: BrokerPositionLifecycleIdentityV3 = read_strict_json(&path)?;
        persisted.validate()?;
        if &persisted != lifecycle {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::IdentityMismatch,
                "persisted lifecycle manifest differs from the requested identity",
            ));
        }
        Ok(())
    }

    fn load_durable_fills(
        &self,
        lifecycle: &BrokerPositionLifecycleIdentityV3,
    ) -> Result<Vec<DurableBrokerDealMoneyV3>, JournalMoneyErrorV3> {
        let deals_dir = self.deals_dir(lifecycle);
        if !deals_dir.exists() {
            return Ok(Vec::new());
        }
        let metadata = fs::metadata(&deals_dir)
            .map_err(|error| io_error("inspect durable deals directory", &deals_dir, error))?;
        if !metadata.is_dir() {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                format!(
                    "durable deals path {} is not a directory",
                    deals_dir.display()
                ),
            ));
        }
        let mut paths = Vec::new();
        for entry in fs::read_dir(&deals_dir)
            .map_err(|error| io_error("read durable deals directory", &deals_dir, error))?
        {
            let entry = entry.map_err(|error| {
                io_error("read durable deal directory entry", &deals_dir, error)
            })?;
            let file_type = entry.file_type().map_err(|error| {
                io_error("inspect durable deal directory entry", &entry.path(), error)
            })?;
            if !file_type.is_file() {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::CorruptLedger,
                    format!("unexpected non-file deal entry {}", entry.path().display()),
                ));
            }
            paths.push(entry.path());
        }
        paths.sort();
        let mut fills = Vec::with_capacity(paths.len());
        let mut deal_ids = BTreeSet::new();
        for path in paths {
            let fill: DurableBrokerDealMoneyV3 = read_strict_json(&path)?;
            fill.validate_against(lifecycle)?;
            let expected_file_name = format!("{}.v3.json", fill.deal_id);
            if path.file_name().and_then(|value| value.to_str())
                != Some(expected_file_name.as_str())
                || !deal_ids.insert(fill.deal_id)
            {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::CorruptLedger,
                    format!(
                        "durable deal file {} has a mismatched or duplicate deal id",
                        path.display()
                    ),
                ));
            }
            fills.push(fill);
        }
        fills.sort_by_key(|fill| fill.deal_id);
        Ok(fills)
    }
}

fn validate_receipt_against_disk(
    receipt: &ClosedPositionJournalReceiptV3,
    lifecycle: &BrokerPositionLifecycleIdentityV3,
    fills: &[DurableBrokerDealMoneyV3],
) -> Result<(), JournalMoneyErrorV3> {
    receipt.validate()?;
    if &receipt.lifecycle != lifecycle || receipt.fills != fills {
        return Err(journal_error(
            JournalMoneyErrorCodeV3::CorruptLedger,
            "final receipt differs from its immutable manifest or deal files",
        ));
    }
    Ok(())
}

fn validate_position_directory_shape(position_dir: &Path) -> Result<(), JournalMoneyErrorV3> {
    let mut saw_manifest = false;
    for entry in fs::read_dir(position_dir)
        .map_err(|error| io_error("read V3 position directory", position_dir, error))?
    {
        let entry = entry
            .map_err(|error| io_error("read V3 position directory entry", position_dir, error))?;
        let name = entry.file_name();
        let name = name.to_str().ok_or_else(|| {
            journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                format!(
                    "position directory {} has a non-UTF8 entry",
                    position_dir.display()
                ),
            )
        })?;
        let file_type = entry.file_type().map_err(|error| {
            io_error("inspect V3 position directory entry", &entry.path(), error)
        })?;
        match name {
            MANIFEST_FILE if file_type.is_file() => saw_manifest = true,
            RECEIPT_FILE if file_type.is_file() => {}
            DEALS_DIRECTORY if file_type.is_dir() => {}
            _ => {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::CorruptLedger,
                    format!("unexpected V3 position artifact {}", entry.path().display()),
                ));
            }
        }
    }
    if !saw_manifest {
        return Err(journal_error(
            JournalMoneyErrorCodeV3::CorruptLedger,
            format!(
                "V3 position directory {} has no manifest",
                position_dir.display()
            ),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImmutablePersistOutcomeV3 {
    Added,
    AlreadyExists,
}

fn persist_immutable_json<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<ImmutablePersistOutcomeV3, JournalMoneyErrorV3> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        journal_error(
            JournalMoneyErrorCodeV3::CorruptLedger,
            format!(
                "serialize immutable journal artifact {}: {error}",
                path.display()
            ),
        )
    })?;
    let parent = path.parent().ok_or_else(|| {
        journal_error(
            JournalMoneyErrorCodeV3::Io,
            format!("immutable journal path {} has no parent", path.display()),
        )
    })?;
    fs::create_dir_all(parent)
        .map_err(|error| io_error("create immutable journal directory", parent, error))?;

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            journal_error(
                JournalMoneyErrorCodeV3::Io,
                format!(
                    "immutable journal path {} has no UTF-8 file name",
                    path.display()
                ),
            )
        })?;
    let temp_path = parent.join(format!(
        ".{file_name}.tmp.{}.{}.{}",
        std::process::id(),
        timestamp,
        sequence
    ));
    let mut temp_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .map_err(|error| io_error("create immutable journal temporary file", &temp_path, error))?;
    let write_result = (|| -> Result<(), JournalMoneyErrorV3> {
        temp_file.write_all(&bytes).map_err(|error| {
            io_error("write immutable journal temporary file", &temp_path, error)
        })?;
        temp_file.sync_all().map_err(|error| {
            io_error("sync immutable journal temporary file", &temp_path, error)
        })?;
        Ok(())
    })();
    drop(temp_file);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }

    let result = match fs::hard_link(&temp_path, path) {
        Ok(()) => Ok(ImmutablePersistOutcomeV3::Added),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists || path.exists() => {
            Ok(ImmutablePersistOutcomeV3::AlreadyExists)
        }
        Err(error) => Err(io_error("publish immutable journal artifact", path, error)),
    };
    let cleanup = fs::remove_file(&temp_path);
    match (result, cleanup) {
        (Ok(outcome), Ok(())) => Ok(outcome),
        (Ok(_), Err(error)) => Err(io_error(
            "remove immutable journal temporary file",
            &temp_path,
            error,
        )),
        (Err(error), _) => Err(error),
    }
}

fn read_strict_json<T>(path: &Path) -> Result<T, JournalMoneyErrorV3>
where
    T: for<'de> Deserialize<'de>,
{
    let bytes =
        fs::read(path).map_err(|error| io_error("read strict journal artifact", path, error))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        journal_error(
            JournalMoneyErrorCodeV3::CorruptLedger,
            format!("parse strict journal JSON {}: {error}", path.display()),
        )
    })?;
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            journal_error(
                JournalMoneyErrorCodeV3::CorruptLedger,
                format!(
                    "strict journal artifact {} has no schema_version",
                    path.display()
                ),
            )
        })?;
    if schema_version != u64::from(CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3) {
        return Err(journal_error(
            JournalMoneyErrorCodeV3::UnsupportedSchemaVersion,
            format!(
                "strict journal artifact {} has unsupported schema version {}",
                path.display(),
                schema_version
            ),
        ));
    }
    serde_json::from_value(value).map_err(|error| {
        journal_error(
            JournalMoneyErrorCodeV3::CorruptLedger,
            format!(
                "strict journal artifact {} violates its V3 wire: {error}",
                path.display()
            ),
        )
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalAccountScopeV3 {
    environment: String,
    account_id: i64,
    account_currency: String,
    money_digits: u32,
}

impl JournalAccountScopeV3 {
    pub fn new(
        environment: impl Into<String>,
        account_id: i64,
        account_currency: impl Into<String>,
        money_digits: u32,
    ) -> Result<Self, JournalMoneyErrorV3> {
        let environment = environment.into();
        let account_currency = account_currency.into();
        canonical_environment(&environment)?;
        validate_currency(&account_currency)?;
        if account_id <= 0 {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::InvalidIdentity,
                "journal account scope requires a positive broker account id",
            ));
        }
        if money_digits > MAX_MONEY_DIGITS {
            return Err(journal_error(
                JournalMoneyErrorCodeV3::InvalidMoneyDigits,
                "journal account scope moneyDigits is outside [0, 10]",
            ));
        }
        Ok(Self {
            environment,
            account_id,
            account_currency,
            money_digits,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AccountPeriodLossesV3 {
    account_currency: String,
    money_digits: u32,
    day_loss_raw_scaled: i64,
    week_loss_raw_scaled: i64,
    month_loss_raw_scaled: i64,
}

impl AccountPeriodLossesV3 {
    pub fn account_currency(&self) -> &str {
        &self.account_currency
    }

    pub const fn money_digits(&self) -> u32 {
        self.money_digits
    }

    pub const fn day_loss_raw_scaled(&self) -> i64 {
        self.day_loss_raw_scaled
    }

    pub const fn week_loss_raw_scaled(&self) -> i64 {
        self.week_loss_raw_scaled
    }

    pub const fn month_loss_raw_scaled(&self) -> i64 {
        self.month_loss_raw_scaled
    }

    pub fn day_loss_account_currency(&self) -> f64 {
        self.day_loss_raw_scaled as f64 / 10.0_f64.powi(self.money_digits as i32)
    }

    pub fn week_loss_account_currency(&self) -> f64 {
        self.week_loss_raw_scaled as f64 / 10.0_f64.powi(self.money_digits as i32)
    }

    pub fn month_loss_account_currency(&self) -> f64 {
        self.month_loss_raw_scaled as f64 / 10.0_f64.powi(self.money_digits as i32)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FinalizedClosedPositionJournalV3 {
    receipts: Vec<ClosedPositionJournalReceiptV3>,
}

impl FinalizedClosedPositionJournalV3 {
    pub fn receipts(&self) -> &[ClosedPositionJournalReceiptV3] {
        &self.receipts
    }

    pub fn len(&self) -> usize {
        self.receipts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }

    pub fn period_losses(
        &self,
        scope: &JournalAccountScopeV3,
        now_ms: i64,
    ) -> Result<AccountPeriodLossesV3, JournalMoneyErrorV3> {
        let (day_start, week_start, month_start) = period_starts_ms(now_ms)?;
        let mut day_loss_raw_scaled = 0_i64;
        let mut week_loss_raw_scaled = 0_i64;
        let mut month_loss_raw_scaled = 0_i64;
        for receipt in &self.receipts {
            let lifecycle = &receipt.lifecycle;
            if lifecycle.environment != scope.environment
                || lifecycle.account_id != scope.account_id
            {
                continue;
            }
            if lifecycle.account_currency != scope.account_currency
                || lifecycle.money_digits != scope.money_digits
            {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::IdentityMismatch,
                    "a matching account/environment receipt has a different currency or moneyDigits",
                ));
            }
            if receipt.component_sum_raw_scaled >= 0 {
                continue;
            }
            let exit_timestamp_ms = receipt.exit_timestamp_ms();
            if exit_timestamp_ms <= 0 || exit_timestamp_ms > now_ms {
                return Err(journal_error(
                    JournalMoneyErrorCodeV3::InvalidTimestamp,
                    "a finalized receipt has a non-positive or future exit timestamp",
                ));
            }
            let loss = receipt
                .component_sum_raw_scaled
                .checked_abs()
                .ok_or_else(|| {
                    journal_error(
                        JournalMoneyErrorCodeV3::InvalidMoney,
                        "receipt loss magnitude overflows signed broker raw money",
                    )
                })?;
            if exit_timestamp_ms >= day_start {
                day_loss_raw_scaled = checked_add(day_loss_raw_scaled, loss, "daily account loss")?;
            }
            if exit_timestamp_ms >= week_start {
                week_loss_raw_scaled =
                    checked_add(week_loss_raw_scaled, loss, "weekly account loss")?;
            }
            if exit_timestamp_ms >= month_start {
                month_loss_raw_scaled =
                    checked_add(month_loss_raw_scaled, loss, "monthly account loss")?;
            }
        }
        Ok(AccountPeriodLossesV3 {
            account_currency: scope.account_currency.clone(),
            money_digits: scope.money_digits,
            day_loss_raw_scaled,
            week_loss_raw_scaled,
            month_loss_raw_scaled,
        })
    }
}

fn period_starts_ms(now_ms: i64) -> Result<(i64, i64, i64), JournalMoneyErrorV3> {
    let now = Utc.timestamp_millis_opt(now_ms).single().ok_or_else(|| {
        journal_error(
            JournalMoneyErrorCodeV3::InvalidTimestamp,
            "period-loss timestamp is outside chrono's supported range",
        )
    })?;
    let day_start = Utc
        .with_ymd_and_hms(now.year(), now.month(), now.day(), 0, 0, 0)
        .single()
        .ok_or_else(|| {
            journal_error(
                JournalMoneyErrorCodeV3::InvalidTimestamp,
                "cannot construct UTC day boundary",
            )
        })?;
    let days_from_monday = i64::from(now.weekday().num_days_from_monday());
    let week_start_ms = day_start
        .timestamp_millis()
        .checked_sub(days_from_monday * 86_400_000)
        .ok_or_else(|| {
            journal_error(
                JournalMoneyErrorCodeV3::InvalidTimestamp,
                "UTC week boundary overflows i64 milliseconds",
            )
        })?;
    let month_start_ms = Utc
        .with_ymd_and_hms(now.year(), now.month(), 1, 0, 0, 0)
        .single()
        .ok_or_else(|| {
            journal_error(
                JournalMoneyErrorCodeV3::InvalidTimestamp,
                "cannot construct UTC month boundary",
            )
        })?
        .timestamp_millis();
    Ok((day_start.timestamp_millis(), week_start_ms, month_start_ms))
}
