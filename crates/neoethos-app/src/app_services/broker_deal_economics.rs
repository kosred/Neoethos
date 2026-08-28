//! Typed live cTrader deal money and optional historical-research parity.
//!
//! The live builder consumes only broker wire fields and the exact broker
//! symbol volume scale. It does not require, construct, or imply historical
//! Bid/Ask authority. Comparing that live evidence with a pre-existing
//! quote-validated research ledger is a separate parity-only operation.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use neoethos_broker_truth::{
    AccountMoneyV1, ExecutionEconomicsArtifactClassV1, ExecutionEconomicsPromotionEligibilityV1,
    QuoteValidatedExecutionEconomicsLedgerV1,
};
use sha2::{Digest, Sha256};

pub const BROKER_DEAL_MONEY_SCHEMA_VERSION_V1: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerDealMoneyErrorCodeV1 {
    InvalidEnvironment,
    InvalidCurrency,
    InvalidMoneyDigits,
    InvalidFilledVolume,
    FilledVolumeMismatch,
    InvalidContract,
    InvalidMoney,
    MissingGrossProfit,
    MissingCommission,
    MissingSwap,
    FillIdentityMismatch,
    CurrencyMismatch,
    MoneyMismatch,
    DuplicateDeal,
    UnverifiedFill,
    AlreadyFinalized,
    HistoricalParityUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerDealMoneyErrorV1 {
    code: BrokerDealMoneyErrorCodeV1,
    detail: String,
}

impl BrokerDealMoneyErrorV1 {
    fn new(code: BrokerDealMoneyErrorCodeV1, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> BrokerDealMoneyErrorCodeV1 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for BrokerDealMoneyErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "broker deal money V1: {}", self.detail)
    }
}

impl Error for BrokerDealMoneyErrorV1 {}

fn money_error(
    code: BrokerDealMoneyErrorCodeV1,
    detail: impl Into<String>,
) -> BrokerDealMoneyErrorV1 {
    BrokerDealMoneyErrorV1::new(code, detail)
}

fn canonical_environment(value: &str) -> Result<&str, BrokerDealMoneyErrorV1> {
    match value {
        "demo" => Ok("demo"),
        "live" => Ok("live"),
        _ => Err(money_error(
            BrokerDealMoneyErrorCodeV1::InvalidEnvironment,
            "environment must be the canonical lowercase value `demo` or `live`",
        )),
    }
}

fn validate_currency(value: &str) -> Result<(), BrokerDealMoneyErrorV1> {
    if value.len() != 3 || !value.bytes().all(|byte| byte.is_ascii_uppercase()) {
        return Err(money_error(
            BrokerDealMoneyErrorCodeV1::InvalidCurrency,
            "account currency must be exactly three uppercase ASCII letters",
        ));
    }
    Ok(())
}

fn append_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn append_optional_f64(bytes: &mut Vec<u8>, value: Option<f64>) {
    match value {
        Some(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value.to_bits().to_be_bytes());
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerPnlConversionFeeV1 {
    Charged { raw_scaled_signed: i64 },
    NotApplied,
}

impl BrokerPnlConversionFeeV1 {
    pub(crate) const fn raw_scaled_signed(self) -> i64 {
        match self {
            Self::Charged { raw_scaled_signed } => raw_scaled_signed,
            Self::NotApplied => 0,
        }
    }

    const fn wire_tag(self) -> u8 {
        match self {
            Self::Charged { .. } => 1,
            Self::NotApplied => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerSymbolVolumeScaleEvidenceV1 {
    environment: String,
    account_id: i64,
    symbol_id: i64,
    symbol_name: String,
    lot_size_raw_centi_units: i64,
    volume_scale_identity_sha256: String,
}

impl BrokerSymbolVolumeScaleEvidenceV1 {
    pub fn new(
        environment: impl Into<String>,
        account_id: i64,
        symbol_id: i64,
        symbol_name: impl Into<String>,
        lot_size_raw_centi_units: i64,
    ) -> Result<Self, BrokerDealMoneyErrorV1> {
        let environment = environment.into();
        canonical_environment(&environment)?;
        let symbol_name = symbol_name.into();
        if account_id <= 0
            || symbol_id <= 0
            || symbol_name.trim().is_empty()
            || symbol_name.trim() != symbol_name
            || lot_size_raw_centi_units <= 0
        {
            return Err(money_error(
                BrokerDealMoneyErrorCodeV1::InvalidContract,
                "broker symbol volume-scale identity is incomplete or invalid",
            ));
        }

        let mut payload = Vec::new();
        append_string(&mut payload, &environment);
        payload.extend_from_slice(&account_id.to_be_bytes());
        payload.extend_from_slice(&symbol_id.to_be_bytes());
        append_string(&mut payload, &symbol_name);
        payload.extend_from_slice(&lot_size_raw_centi_units.to_be_bytes());
        let volume_scale_identity_sha256 = sha256_hex("neoethos-broker-volume-scale-v1", &payload);

        Ok(Self {
            environment,
            account_id,
            symbol_id,
            symbol_name,
            lot_size_raw_centi_units,
            volume_scale_identity_sha256,
        })
    }

    pub fn environment(&self) -> &str {
        &self.environment
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn symbol_id(&self) -> i64 {
        self.symbol_id
    }

    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }

    pub const fn lot_size_raw_centi_units(&self) -> i64 {
        self.lot_size_raw_centi_units
    }

    pub fn contract_units_per_lot(&self) -> f64 {
        self.lot_size_raw_centi_units as f64 / 100.0
    }

    pub fn volume_scale_identity_sha256(&self) -> &str {
        &self.volume_scale_identity_sha256
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrokerDealWireSnapshotV1 {
    pub environment: String,
    pub account_id: i64,
    pub deal_id: i64,
    pub order_id: i64,
    pub position_id: i64,
    pub symbol_id: i64,
    pub symbol_name: String,
    pub deal_status: String,
    pub trade_side: String,
    pub filled_volume_raw_centi_units: i64,
    pub execution_timestamp_ms: i64,
    pub execution_price: Option<f64>,
    pub entry_price: Option<f64>,
    pub money_digits: Option<u32>,
    pub gross_profit_raw_scaled: Option<i64>,
    pub commission_raw_scaled_signed: Option<i64>,
    pub swap_raw_scaled_signed: Option<i64>,
    pub pnl_conversion_fee: BrokerPnlConversionFeeV1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrokerDealMoneyEvidenceV1 {
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
    contract_units_per_lot: f64,
    actual_filled_lots: f64,
    execution_timestamp_ms: i64,
    execution_price: f64,
    entry_price: f64,
    money_digits: u32,
    gross_profit_raw_scaled: i64,
    commission_raw_scaled_signed: i64,
    swap_raw_scaled_signed: i64,
    component_sum_raw_scaled: i64,
    gross_profit_account_currency: AccountMoneyV1,
    commission_account_currency_signed: AccountMoneyV1,
    swap_account_currency_signed: AccountMoneyV1,
    pnl_conversion_fee_state: BrokerPnlConversionFeeV1,
    pnl_conversion_fee_account_currency_signed: AccountMoneyV1,
    component_sum_account_currency: AccountMoneyV1,
    volume_scale_identity_sha256: String,
    deal_identity_sha256: String,
}

impl BrokerDealMoneyEvidenceV1 {
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn environment(&self) -> &str {
        &self.environment
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn deal_id(&self) -> i64 {
        self.deal_id
    }

    pub const fn order_id(&self) -> i64 {
        self.order_id
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

    pub fn trade_side(&self) -> &str {
        &self.trade_side
    }

    pub const fn filled_volume_raw_centi_units(&self) -> i64 {
        self.filled_volume_raw_centi_units
    }

    pub const fn contract_units_per_lot(&self) -> f64 {
        self.contract_units_per_lot
    }

    pub const fn actual_filled_lots(&self) -> f64 {
        self.actual_filled_lots
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

    pub const fn gross_profit_raw_scaled(&self) -> i64 {
        self.gross_profit_raw_scaled
    }

    pub const fn commission_raw_scaled_signed(&self) -> i64 {
        self.commission_raw_scaled_signed
    }

    pub const fn swap_raw_scaled_signed(&self) -> i64 {
        self.swap_raw_scaled_signed
    }

    pub const fn component_sum_raw_scaled(&self) -> i64 {
        self.component_sum_raw_scaled
    }

    pub fn account_currency(&self) -> &str {
        self.component_sum_account_currency.currency()
    }

    pub fn gross_profit_account_currency(&self) -> &AccountMoneyV1 {
        &self.gross_profit_account_currency
    }

    pub fn commission_account_currency_signed(&self) -> &AccountMoneyV1 {
        &self.commission_account_currency_signed
    }

    pub fn swap_account_currency_signed(&self) -> &AccountMoneyV1 {
        &self.swap_account_currency_signed
    }

    pub const fn pnl_conversion_fee_state(&self) -> BrokerPnlConversionFeeV1 {
        self.pnl_conversion_fee_state
    }

    pub fn pnl_conversion_fee_account_currency_signed(&self) -> &AccountMoneyV1 {
        &self.pnl_conversion_fee_account_currency_signed
    }

    pub fn component_sum_account_currency(&self) -> &AccountMoneyV1 {
        &self.component_sum_account_currency
    }

    pub fn volume_scale_identity_sha256(&self) -> &str {
        &self.volume_scale_identity_sha256
    }

    pub fn deal_identity_sha256(&self) -> &str {
        &self.deal_identity_sha256
    }
}

fn account_money(currency: &str, amount: f64) -> Result<AccountMoneyV1, BrokerDealMoneyErrorV1> {
    AccountMoneyV1::new(currency, amount)
        .map_err(|error| money_error(BrokerDealMoneyErrorCodeV1::InvalidMoney, error.to_string()))
}

fn required_positive_price(label: &str, value: Option<f64>) -> Result<f64, BrokerDealMoneyErrorV1> {
    match value {
        Some(value) if value.is_finite() && value > 0.0 => Ok(value),
        _ => Err(money_error(
            BrokerDealMoneyErrorCodeV1::FillIdentityMismatch,
            format!("{label} is missing, non-finite, or non-positive"),
        )),
    }
}

fn deal_identity_sha256(
    deal: &BrokerDealWireSnapshotV1,
    account_currency: &str,
    money_digits: u32,
    gross_profit_raw_scaled: i64,
    commission_raw_scaled_signed: i64,
    swap_raw_scaled_signed: i64,
    volume_scale_identity_sha256: &str,
) -> String {
    let mut payload = Vec::new();
    append_string(&mut payload, &deal.environment);
    payload.extend_from_slice(&deal.account_id.to_be_bytes());
    payload.extend_from_slice(&deal.deal_id.to_be_bytes());
    payload.extend_from_slice(&deal.order_id.to_be_bytes());
    payload.extend_from_slice(&deal.position_id.to_be_bytes());
    payload.extend_from_slice(&deal.symbol_id.to_be_bytes());
    append_string(&mut payload, &deal.symbol_name);
    append_string(&mut payload, &deal.deal_status);
    append_string(&mut payload, &deal.trade_side);
    payload.extend_from_slice(&deal.filled_volume_raw_centi_units.to_be_bytes());
    payload.extend_from_slice(&deal.execution_timestamp_ms.to_be_bytes());
    append_optional_f64(&mut payload, deal.execution_price);
    append_optional_f64(&mut payload, deal.entry_price);
    payload.extend_from_slice(&money_digits.to_be_bytes());
    payload.extend_from_slice(&gross_profit_raw_scaled.to_be_bytes());
    payload.extend_from_slice(&commission_raw_scaled_signed.to_be_bytes());
    payload.extend_from_slice(&swap_raw_scaled_signed.to_be_bytes());
    payload.push(deal.pnl_conversion_fee.wire_tag());
    payload.extend_from_slice(&deal.pnl_conversion_fee.raw_scaled_signed().to_be_bytes());
    append_string(&mut payload, account_currency);
    append_string(&mut payload, volume_scale_identity_sha256);
    sha256_hex("neoethos-broker-deal-money-evidence-v1", &payload)
}

pub fn build_broker_deal_money_evidence_v1(
    deal: &BrokerDealWireSnapshotV1,
    volume_scale: &BrokerSymbolVolumeScaleEvidenceV1,
    account_currency: &str,
) -> Result<BrokerDealMoneyEvidenceV1, BrokerDealMoneyErrorV1> {
    canonical_environment(&deal.environment)?;
    validate_currency(account_currency)?;
    if deal.environment != volume_scale.environment
        || deal.account_id != volume_scale.account_id
        || deal.symbol_id != volume_scale.symbol_id
        || deal.symbol_name != volume_scale.symbol_name
    {
        return Err(money_error(
            BrokerDealMoneyErrorCodeV1::FillIdentityMismatch,
            "deal environment/account/symbol identity differs from exact broker volume scale",
        ));
    }
    if deal.deal_id <= 0
        || deal.order_id <= 0
        || deal.position_id <= 0
        || deal.execution_timestamp_ms <= 0
        || deal.deal_status != "FILLED"
        || !matches!(deal.trade_side.as_str(), "BUY" | "SELL")
    {
        return Err(money_error(
            BrokerDealMoneyErrorCodeV1::FillIdentityMismatch,
            "deal identifiers, status, side, or timestamp are invalid",
        ));
    }
    if deal.filled_volume_raw_centi_units <= 0 {
        return Err(money_error(
            BrokerDealMoneyErrorCodeV1::InvalidFilledVolume,
            "broker filled volume must be a positive raw centi-unit integer",
        ));
    }
    let money_digits = deal.money_digits.ok_or_else(|| {
        money_error(
            BrokerDealMoneyErrorCodeV1::InvalidMoneyDigits,
            "closing deal omitted its wire moneyDigits",
        )
    })?;
    if money_digits > 10 {
        return Err(money_error(
            BrokerDealMoneyErrorCodeV1::InvalidMoneyDigits,
            "closing deal moneyDigits is outside the supported [0, 10] range",
        ));
    }
    let gross_profit_raw_scaled = deal.gross_profit_raw_scaled.ok_or_else(|| {
        money_error(
            BrokerDealMoneyErrorCodeV1::MissingGrossProfit,
            "closing deal omitted gross profit",
        )
    })?;
    let commission_raw_scaled_signed = deal.commission_raw_scaled_signed.ok_or_else(|| {
        money_error(
            BrokerDealMoneyErrorCodeV1::MissingCommission,
            "closing deal omitted commission",
        )
    })?;
    let swap_raw_scaled_signed = deal.swap_raw_scaled_signed.ok_or_else(|| {
        money_error(
            BrokerDealMoneyErrorCodeV1::MissingSwap,
            "closing deal omitted swap",
        )
    })?;
    let broker_conversion_fee_account_currency_signed = match deal.pnl_conversion_fee {
        BrokerPnlConversionFeeV1::Charged { raw_scaled_signed } => raw_scaled_signed,
        BrokerPnlConversionFeeV1::NotApplied => 0,
    };
    let component_sum_raw_scaled = gross_profit_raw_scaled
        .checked_add(commission_raw_scaled_signed)
        .and_then(|sum| sum.checked_add(swap_raw_scaled_signed))
        .and_then(|sum| sum.checked_add(broker_conversion_fee_account_currency_signed))
        .ok_or_else(|| {
            money_error(
                BrokerDealMoneyErrorCodeV1::InvalidMoney,
                "signed broker money components overflow i64",
            )
        })?;

    let execution_price = required_positive_price("execution price", deal.execution_price)?;
    let entry_price = required_positive_price("entry price", deal.entry_price)?;
    let divisor = 10.0_f64.powi(money_digits as i32);
    let gross_profit_account_currency = gross_profit_raw_scaled as f64 / divisor;
    let broker_fee_account_currency_signed = commission_raw_scaled_signed as f64 / divisor;
    let broker_swap_account_currency_signed = swap_raw_scaled_signed as f64 / divisor;
    let broker_conversion_fee_account_currency_signed =
        broker_conversion_fee_account_currency_signed as f64 / divisor;
    let component_sum_account_currency = component_sum_raw_scaled as f64 / divisor;
    let contract_units_per_lot = volume_scale.contract_units_per_lot();
    let actual_filled_lots =
        deal.filled_volume_raw_centi_units as f64 / volume_scale.lot_size_raw_centi_units as f64;
    if !contract_units_per_lot.is_finite()
        || contract_units_per_lot <= 0.0
        || !actual_filled_lots.is_finite()
        || actual_filled_lots <= 0.0
    {
        return Err(money_error(
            BrokerDealMoneyErrorCodeV1::InvalidFilledVolume,
            "broker lotSize cannot resolve a finite positive actual fill in standard lots",
        ));
    }

    let volume_scale_identity_sha256 = volume_scale.volume_scale_identity_sha256.clone();
    let deal_identity_sha256 = deal_identity_sha256(
        deal,
        account_currency,
        money_digits,
        gross_profit_raw_scaled,
        commission_raw_scaled_signed,
        swap_raw_scaled_signed,
        &volume_scale_identity_sha256,
    );
    Ok(BrokerDealMoneyEvidenceV1 {
        schema_version: BROKER_DEAL_MONEY_SCHEMA_VERSION_V1,
        environment: deal.environment.clone(),
        account_id: deal.account_id,
        deal_id: deal.deal_id,
        order_id: deal.order_id,
        position_id: deal.position_id,
        symbol_id: deal.symbol_id,
        symbol_name: deal.symbol_name.clone(),
        trade_side: deal.trade_side.clone(),
        filled_volume_raw_centi_units: deal.filled_volume_raw_centi_units,
        contract_units_per_lot,
        actual_filled_lots,
        execution_timestamp_ms: deal.execution_timestamp_ms,
        execution_price,
        entry_price,
        money_digits,
        gross_profit_raw_scaled,
        commission_raw_scaled_signed,
        swap_raw_scaled_signed,
        component_sum_raw_scaled,
        gross_profit_account_currency: account_money(
            account_currency,
            gross_profit_account_currency,
        )?,
        commission_account_currency_signed: account_money(
            account_currency,
            broker_fee_account_currency_signed,
        )?,
        swap_account_currency_signed: account_money(
            account_currency,
            broker_swap_account_currency_signed,
        )?,
        pnl_conversion_fee_state: deal.pnl_conversion_fee,
        pnl_conversion_fee_account_currency_signed: account_money(
            account_currency,
            broker_conversion_fee_account_currency_signed,
        )?,
        component_sum_account_currency: account_money(
            account_currency,
            component_sum_account_currency,
        )?,
        volume_scale_identity_sha256,
        deal_identity_sha256,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerDealObservationV1 {
    Added,
    Duplicate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrokerClosedPositionMoneyV1 {
    position_id: i64,
    account_currency: String,
    deal_count: usize,
    filled_volume_raw_centi_units: i64,
    money_digits: u32,
    component_sum_raw_scaled: i64,
    actual_filled_lots: f64,
    component_sum_account_currency: AccountMoneyV1,
}

impl BrokerClosedPositionMoneyV1 {
    pub const fn position_id(&self) -> i64 {
        self.position_id
    }

    pub fn account_currency(&self) -> &str {
        &self.account_currency
    }

    pub const fn deal_count(&self) -> usize {
        self.deal_count
    }

    pub const fn filled_volume_raw_centi_units(&self) -> i64 {
        self.filled_volume_raw_centi_units
    }

    pub const fn money_digits(&self) -> u32 {
        self.money_digits
    }

    pub const fn component_sum_raw_scaled(&self) -> i64 {
        self.component_sum_raw_scaled
    }

    pub const fn actual_filled_lots(&self) -> f64 {
        self.actual_filled_lots
    }

    pub fn component_sum_account_currency(&self) -> &AccountMoneyV1 {
        &self.component_sum_account_currency
    }
}

#[derive(Clone, Debug)]
pub struct BrokerPositionMoneyAccumulatorV1 {
    environment: String,
    account_id: i64,
    position_id: i64,
    symbol_id: i64,
    symbol_name: String,
    account_currency: String,
    volume_scale_identity_sha256: String,
    seen_deal_ids: BTreeMap<i64, String>,
    filled_volume_raw_centi_units: i64,
    money_digits: u32,
    component_sum_raw_scaled: i64,
    actual_filled_lots: f64,
    has_unverified_fill: bool,
    finalized: bool,
}

impl BrokerPositionMoneyAccumulatorV1 {
    pub fn new(seed: &BrokerDealMoneyEvidenceV1) -> Self {
        Self {
            environment: seed.environment.clone(),
            account_id: seed.account_id,
            position_id: seed.position_id,
            symbol_id: seed.symbol_id,
            symbol_name: seed.symbol_name.clone(),
            account_currency: seed.account_currency().to_owned(),
            volume_scale_identity_sha256: seed.volume_scale_identity_sha256.clone(),
            seen_deal_ids: BTreeMap::new(),
            filled_volume_raw_centi_units: 0,
            money_digits: seed.money_digits,
            component_sum_raw_scaled: 0,
            actual_filled_lots: 0.0,
            has_unverified_fill: false,
            finalized: false,
        }
    }

    pub fn observe_fill(
        &mut self,
        fill: &BrokerDealMoneyEvidenceV1,
    ) -> Result<BrokerDealObservationV1, BrokerDealMoneyErrorV1> {
        if self.finalized {
            return Err(money_error(
                BrokerDealMoneyErrorCodeV1::AlreadyFinalized,
                "cannot append a fill after position money finalized",
            ));
        }
        if fill.environment != self.environment
            || fill.account_id != self.account_id
            || fill.position_id != self.position_id
            || fill.symbol_id != self.symbol_id
            || fill.symbol_name != self.symbol_name
            || fill.account_currency() != self.account_currency
            || fill.volume_scale_identity_sha256 != self.volume_scale_identity_sha256
            || fill.money_digits != self.money_digits
        {
            return Err(money_error(
                BrokerDealMoneyErrorCodeV1::FillIdentityMismatch,
                "partial-close fill identity differs from its position accumulator",
            ));
        }
        if let Some(identity) = self.seen_deal_ids.get(&fill.deal_id) {
            if identity == &fill.deal_identity_sha256 {
                return Ok(BrokerDealObservationV1::Duplicate);
            }
            return Err(money_error(
                BrokerDealMoneyErrorCodeV1::DuplicateDeal,
                "one broker deal id was observed with two different evidence identities",
            ));
        }

        let next_filled_volume_raw_centi_units = self
            .filled_volume_raw_centi_units
            .checked_add(fill.filled_volume_raw_centi_units)
            .ok_or_else(|| {
                money_error(
                    BrokerDealMoneyErrorCodeV1::InvalidFilledVolume,
                    "partial-close raw filled-volume sum overflowed i64",
                )
            })?;
        let next_lots = self.actual_filled_lots + fill.actual_filled_lots;
        let next_component_sum_raw_scaled = self
            .component_sum_raw_scaled
            .checked_add(fill.component_sum_raw_scaled)
            .ok_or_else(|| {
                money_error(
                    BrokerDealMoneyErrorCodeV1::InvalidMoney,
                    "partial-close raw money-component sum overflowed i64",
                )
            })?;
        if !next_lots.is_finite() {
            return Err(money_error(
                BrokerDealMoneyErrorCodeV1::InvalidMoney,
                "partial-close accumulator overflowed finite lots",
            ));
        }
        self.seen_deal_ids
            .insert(fill.deal_id, fill.deal_identity_sha256.clone());
        self.filled_volume_raw_centi_units = next_filled_volume_raw_centi_units;
        self.actual_filled_lots = next_lots;
        self.component_sum_raw_scaled = next_component_sum_raw_scaled;
        Ok(BrokerDealObservationV1::Added)
    }

    pub fn refuse_unverified_fill(&mut self) {
        self.has_unverified_fill = true;
    }

    pub fn verify_complete_filled_volume(
        &self,
        expected_entry_filled_volume_raw_centi_units: i64,
    ) -> Result<(), BrokerDealMoneyErrorV1> {
        if expected_entry_filled_volume_raw_centi_units <= 0
            || self.filled_volume_raw_centi_units != expected_entry_filled_volume_raw_centi_units
        {
            return Err(money_error(
                BrokerDealMoneyErrorCodeV1::FilledVolumeMismatch,
                format!(
                    "verified closing fills total {} raw centi-units; exact entry fill was {}",
                    self.filled_volume_raw_centi_units,
                    expected_entry_filled_volume_raw_centi_units
                ),
            ));
        }
        Ok(())
    }

    pub fn finalize_if_position_closed(
        &mut self,
        position_still_open: bool,
    ) -> Result<Option<BrokerClosedPositionMoneyV1>, BrokerDealMoneyErrorV1> {
        if position_still_open {
            return Ok(None);
        }
        if self.finalized {
            return Err(money_error(
                BrokerDealMoneyErrorCodeV1::AlreadyFinalized,
                "position money was already finalized",
            ));
        }
        if self.has_unverified_fill || self.seen_deal_ids.is_empty() {
            return Err(money_error(
                BrokerDealMoneyErrorCodeV1::UnverifiedFill,
                "a flat broker position has missing or unverified close-fill money",
            ));
        }
        let component_sum_account_currency = account_money(
            &self.account_currency,
            self.component_sum_raw_scaled as f64 / 10.0_f64.powi(self.money_digits as i32),
        )?;
        let result = BrokerClosedPositionMoneyV1 {
            position_id: self.position_id,
            account_currency: self.account_currency.clone(),
            deal_count: self.seen_deal_ids.len(),
            filled_volume_raw_centi_units: self.filled_volume_raw_centi_units,
            money_digits: self.money_digits,
            component_sum_raw_scaled: self.component_sum_raw_scaled,
            actual_filled_lots: self.actual_filled_lots,
            component_sum_account_currency,
        };
        self.finalized = true;
        Ok(Some(result))
    }

    pub const fn is_finalized(&self) -> bool {
        self.finalized
    }

    pub const fn filled_volume_raw_centi_units(&self) -> i64 {
        self.filled_volume_raw_centi_units
    }

    pub fn component_sum_account_currency(&self) -> f64 {
        self.component_sum_raw_scaled as f64 / 10.0_f64.powi(self.money_digits as i32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerDealEconomicsParityAuthorityV1 {
    ParityOnly,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerDealEconomicsParityV1 {
    authority: BrokerDealEconomicsParityAuthorityV1,
    promotion_eligibility: ExecutionEconomicsPromotionEligibilityV1,
    deal_identity_sha256: String,
    execution_economics_ledger_sha256: String,
}

impl BrokerDealEconomicsParityV1 {
    pub const fn authority(&self) -> BrokerDealEconomicsParityAuthorityV1 {
        self.authority
    }

    pub const fn promotion_eligibility(&self) -> ExecutionEconomicsPromotionEligibilityV1 {
        self.promotion_eligibility
    }

    pub fn deal_identity_sha256(&self) -> &str {
        &self.deal_identity_sha256
    }

    pub fn execution_economics_ledger_sha256(&self) -> &str {
        &self.execution_economics_ledger_sha256
    }
}

fn amount_as_raw(amount: f64, money_digits: u32) -> Result<i64, BrokerDealMoneyErrorV1> {
    let scaled = amount * 10.0_f64.powi(money_digits as i32);
    let rounded = scaled.round();
    let tolerance = scaled.abs().max(1.0) * f64::EPSILON * 16.0;
    if !scaled.is_finite()
        || rounded < i64::MIN as f64
        || rounded > i64::MAX as f64
        || (scaled - rounded).abs() > tolerance
    {
        return Err(money_error(
            BrokerDealMoneyErrorCodeV1::MoneyMismatch,
            "historical money is not exactly representable at broker moneyDigits",
        ));
    }
    Ok(rounded as i64)
}

fn require_money_equal(
    label: &str,
    actual: f64,
    expected: f64,
    money_digits: u32,
) -> Result<(), BrokerDealMoneyErrorV1> {
    if amount_as_raw(actual, money_digits)? != amount_as_raw(expected, money_digits)? {
        return Err(money_error(
            BrokerDealMoneyErrorCodeV1::MoneyMismatch,
            format!("broker and historical {label} differ at broker moneyDigits"),
        ));
    }
    Ok(())
}

pub fn reconcile_broker_deal_economics_v1(
    actual: &BrokerDealMoneyEvidenceV1,
    economics: &QuoteValidatedExecutionEconomicsLedgerV1,
) -> Result<BrokerDealEconomicsParityV1, BrokerDealMoneyErrorV1> {
    if economics.artifact_class() != ExecutionEconomicsArtifactClassV1::ResearchOnly
        || economics.promotion_eligibility()
            != ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible
    {
        return Err(money_error(
            BrokerDealMoneyErrorCodeV1::HistoricalParityUnavailable,
            "historical economics input is not research-only/non-promotable",
        ));
    }
    if actual.account_currency() != economics.account_currency() {
        return Err(money_error(
            BrokerDealMoneyErrorCodeV1::CurrencyMismatch,
            "live deal and historical economics account currencies differ",
        ));
    }
    if actual.symbol_name() != economics.symbol_contract().symbol_name()
        || actual.contract_units_per_lot().to_bits()
            != economics
                .symbol_contract()
                .contract_units_per_lot()
                .to_bits()
        || actual.actual_filled_lots().to_bits() != economics.filled_lots().to_bits()
        || actual.execution_timestamp_ms() != economics.exit_fill_timestamp_unix_ms()
        || actual.execution_price().to_bits() != economics.modeled_exit_price().to_bits()
        || actual.entry_price().to_bits() != economics.modeled_entry_price().to_bits()
    {
        return Err(money_error(
            BrokerDealMoneyErrorCodeV1::FillIdentityMismatch,
            "live broker fill tuple differs from the historical quote-ledger fill tuple",
        ));
    }

    let commission_signed = -(economics.entry_commission_account_currency().amount()
        + economics.exit_commission_account_currency().amount());
    let conversion_fee_signed = -economics.pnl_conversion_fee_account_currency().amount();
    require_money_equal(
        "gross profit",
        actual.gross_profit_account_currency().amount(),
        economics.gross_pnl_account_currency().amount(),
        actual.money_digits(),
    )?;
    require_money_equal(
        "commission",
        actual.commission_account_currency_signed().amount(),
        commission_signed,
        actual.money_digits(),
    )?;
    require_money_equal(
        "swap",
        actual.swap_account_currency_signed().amount(),
        economics.swap_account_currency_signed().amount(),
        actual.money_digits(),
    )?;
    require_money_equal(
        "PnL conversion fee",
        actual.pnl_conversion_fee_account_currency_signed().amount(),
        conversion_fee_signed,
        actual.money_digits(),
    )?;
    require_money_equal(
        "component sum",
        actual.component_sum_account_currency().amount(),
        economics.net_pnl_account_currency().amount(),
        actual.money_digits(),
    )?;

    let execution_economics_ledger_sha256 = economics.ledger_sha256().to_owned();
    Ok(BrokerDealEconomicsParityV1 {
        authority: BrokerDealEconomicsParityAuthorityV1::ParityOnly,
        promotion_eligibility: ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible,
        deal_identity_sha256: actual.deal_identity_sha256().to_owned(),
        execution_economics_ledger_sha256,
    })
}
