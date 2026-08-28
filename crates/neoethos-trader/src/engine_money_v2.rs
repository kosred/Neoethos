use std::{error::Error, fmt};

use neoethos_broker_truth::{
    AccountMoneyV1, ExecutionEconomicsArtifactClassV1, ExecutionEconomicsPromotionEligibilityV1,
    QuoteValidatedExecutionEconomicsLedgerV1,
};
use serde::{Deserialize, Serialize};

use crate::contracts::ExecReport;
use crate::engine::EngineStats;

pub const ENGINE_MONEY_SCHEMA_VERSION_V2: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineMoneyErrorCodeV2 {
    InvalidLots,
    InvalidPrice,
    InvalidTimestamp,
    InvalidIdentity,
    InvalidPosition,
    InvalidMoney,
    MissingPosition,
    ExceedsRemainingLots,
    MissingEconomicsLedger,
    MissingMark,
    MissingConversionEvidence,
    CurrencyMismatch,
    EconomicsLedgerMismatch,
    UnsupportedSchemaVersion,
    LegacyMoneyWireRefused,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineMoneyErrorV2 {
    code: EngineMoneyErrorCodeV2,
    detail: String,
}

impl EngineMoneyErrorV2 {
    fn new(code: EngineMoneyErrorCodeV2, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> EngineMoneyErrorCodeV2 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for EngineMoneyErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "engine money V2: {}", self.detail)
    }
}

impl Error for EngineMoneyErrorV2 {}

fn money_error(code: EngineMoneyErrorCodeV2, detail: impl Into<String>) -> EngineMoneyErrorV2 {
    EngineMoneyErrorV2::new(code, detail)
}

fn validate_sha256(label: &str, value: &str) -> Result<(), EngineMoneyErrorV2> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(money_error(
            EngineMoneyErrorCodeV2::InvalidIdentity,
            format!("{label} must be exactly 64 lowercase hexadecimal characters"),
        ));
    }
    Ok(())
}

fn validate_position_text(label: &str, value: &str) -> Result<(), EngineMoneyErrorV2> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(money_error(
            EngineMoneyErrorCodeV2::InvalidPosition,
            format!("{label} must be non-empty and already trimmed"),
        ));
    }
    Ok(())
}

fn validate_money_currency(
    label: &str,
    money: &AccountMoneyV1,
    account_currency: &str,
) -> Result<(), EngineMoneyErrorV2> {
    if money.currency() != account_currency {
        return Err(money_error(
            EngineMoneyErrorCodeV2::CurrencyMismatch,
            format!("{label} currency differs from the engine account currency"),
        ));
    }
    if !money.amount().is_finite() {
        return Err(money_error(
            EngineMoneyErrorCodeV2::InvalidMoney,
            format!("{label} amount must be finite"),
        ));
    }
    Ok(())
}

fn account_money(
    account_currency: &str,
    amount: f64,
) -> Result<AccountMoneyV1, EngineMoneyErrorV2> {
    AccountMoneyV1::new(account_currency, amount).map_err(|error| {
        money_error(
            EngineMoneyErrorCodeV2::InvalidMoney,
            format!("cannot construct typed account money: {error}"),
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StandardLotsV1(f64);

impl StandardLotsV1 {
    pub fn new(value: f64) -> Result<Self, EngineMoneyErrorV2> {
        if !value.is_finite() || value <= 0.0 {
            return Err(money_error(
                EngineMoneyErrorCodeV2::InvalidLots,
                "standard lots must be finite and strictly positive",
            ));
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> f64 {
        self.0
    }

    fn validate(self) -> Result<(), EngineMoneyErrorV2> {
        Self::new(self.0).map(|_| ())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExecutionPriceV1(f64);

impl ExecutionPriceV1 {
    pub fn new(value: f64) -> Result<Self, EngineMoneyErrorV2> {
        if !value.is_finite() || value <= 0.0 {
            return Err(money_error(
                EngineMoneyErrorCodeV2::InvalidPrice,
                "execution price must be finite and strictly positive",
            ));
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> f64 {
        self.0
    }

    fn validate(self) -> Result<(), EngineMoneyErrorV2> {
        Self::new(self.0).map(|_| ())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FillSideV2 {
    Entry,
    Exit,
}

trait ExecutionEconomicsViewV2 {
    fn account_currency(&self) -> &str;
    fn symbol(&self) -> &str;
    fn filled_lots(&self) -> f64;
    fn modeled_entry_price(&self) -> f64;
    fn modeled_exit_price(&self) -> f64;
    fn entry_fill_timestamp_unix_ms(&self) -> i64;
    fn exit_fill_timestamp_unix_ms(&self) -> i64;
    fn entry_fill_identity_sha256(&self) -> &str;
    fn exit_fill_identity_sha256(&self) -> &str;
    fn ledger_sha256(&self) -> &str;
    fn entry_commission_account_currency(&self) -> &AccountMoneyV1;
    fn exit_commission_account_currency(&self) -> &AccountMoneyV1;
    fn net_pnl_account_currency(&self) -> &AccountMoneyV1;
    fn artifact_class(&self) -> ExecutionEconomicsArtifactClassV1;
    fn promotion_eligibility(&self) -> ExecutionEconomicsPromotionEligibilityV1;
}

impl ExecutionEconomicsViewV2 for QuoteValidatedExecutionEconomicsLedgerV1 {
    fn account_currency(&self) -> &str {
        QuoteValidatedExecutionEconomicsLedgerV1::account_currency(self)
    }

    fn symbol(&self) -> &str {
        self.symbol_contract().symbol_name()
    }

    fn filled_lots(&self) -> f64 {
        QuoteValidatedExecutionEconomicsLedgerV1::filled_lots(self)
    }

    fn modeled_entry_price(&self) -> f64 {
        QuoteValidatedExecutionEconomicsLedgerV1::modeled_entry_price(self)
    }

    fn modeled_exit_price(&self) -> f64 {
        QuoteValidatedExecutionEconomicsLedgerV1::modeled_exit_price(self)
    }

    fn entry_fill_timestamp_unix_ms(&self) -> i64 {
        QuoteValidatedExecutionEconomicsLedgerV1::entry_fill_timestamp_unix_ms(self)
    }

    fn exit_fill_timestamp_unix_ms(&self) -> i64 {
        QuoteValidatedExecutionEconomicsLedgerV1::exit_fill_timestamp_unix_ms(self)
    }

    fn entry_fill_identity_sha256(&self) -> &str {
        QuoteValidatedExecutionEconomicsLedgerV1::entry_fill_identity_sha256(self)
    }

    fn exit_fill_identity_sha256(&self) -> &str {
        QuoteValidatedExecutionEconomicsLedgerV1::exit_fill_identity_sha256(self)
    }

    fn ledger_sha256(&self) -> &str {
        QuoteValidatedExecutionEconomicsLedgerV1::ledger_sha256(self)
    }

    fn entry_commission_account_currency(&self) -> &AccountMoneyV1 {
        QuoteValidatedExecutionEconomicsLedgerV1::entry_commission_account_currency(self)
    }

    fn exit_commission_account_currency(&self) -> &AccountMoneyV1 {
        QuoteValidatedExecutionEconomicsLedgerV1::exit_commission_account_currency(self)
    }

    fn net_pnl_account_currency(&self) -> &AccountMoneyV1 {
        QuoteValidatedExecutionEconomicsLedgerV1::net_pnl_account_currency(self)
    }

    fn artifact_class(&self) -> ExecutionEconomicsArtifactClassV1 {
        QuoteValidatedExecutionEconomicsLedgerV1::artifact_class(self)
    }

    fn promotion_eligibility(&self) -> ExecutionEconomicsPromotionEligibilityV1 {
        QuoteValidatedExecutionEconomicsLedgerV1::promotion_eligibility(self)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilledExecutionReportV2 {
    schema_version: u32,
    fill_identity_sha256: String,
    position_id: String,
    symbol: String,
    fill_side: FillSideV2,
    actual_filled_lots: StandardLotsV1,
    fill_price: ExecutionPriceV1,
    filled_at_unix_ms: i64,
    execution_economics_ledger_sha256: String,
}

impl FilledExecutionReportV2 {
    pub fn from_economics_ledger(
        position_id: impl Into<String>,
        fill_side: FillSideV2,
        economics_ledger: &QuoteValidatedExecutionEconomicsLedgerV1,
    ) -> Result<Self, EngineMoneyErrorV2> {
        Self::from_economics_view(position_id, fill_side, economics_ledger)
    }

    fn from_economics_view(
        position_id: impl Into<String>,
        fill_side: FillSideV2,
        economics_ledger: &impl ExecutionEconomicsViewV2,
    ) -> Result<Self, EngineMoneyErrorV2> {
        let (fill_identity_sha256, fill_price, filled_at_unix_ms) = match fill_side {
            FillSideV2::Entry => (
                economics_ledger.entry_fill_identity_sha256(),
                economics_ledger.modeled_entry_price(),
                economics_ledger.entry_fill_timestamp_unix_ms(),
            ),
            FillSideV2::Exit => (
                economics_ledger.exit_fill_identity_sha256(),
                economics_ledger.modeled_exit_price(),
                economics_ledger.exit_fill_timestamp_unix_ms(),
            ),
        };
        let report = Self {
            schema_version: ENGINE_MONEY_SCHEMA_VERSION_V2,
            fill_identity_sha256: fill_identity_sha256.to_owned(),
            position_id: position_id.into(),
            symbol: economics_ledger.symbol().to_owned(),
            fill_side,
            actual_filled_lots: StandardLotsV1::new(economics_ledger.filled_lots())?,
            fill_price: ExecutionPriceV1::new(fill_price)?,
            filled_at_unix_ms,
            execution_economics_ledger_sha256: economics_ledger.ledger_sha256().to_owned(),
        };
        report.validate_against_view(economics_ledger)?;
        Ok(report)
    }

    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn fill_identity_sha256(&self) -> &str {
        &self.fill_identity_sha256
    }

    pub fn position_id(&self) -> &str {
        &self.position_id
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub const fn fill_side(&self) -> FillSideV2 {
        self.fill_side
    }

    pub const fn actual_filled_lots(&self) -> StandardLotsV1 {
        self.actual_filled_lots
    }

    pub const fn fill_price(&self) -> ExecutionPriceV1 {
        self.fill_price
    }

    pub const fn filled_at_unix_ms(&self) -> i64 {
        self.filled_at_unix_ms
    }

    pub fn execution_economics_ledger_sha256(&self) -> &str {
        &self.execution_economics_ledger_sha256
    }

    fn validate_against_view(
        &self,
        economics_ledger: &impl ExecutionEconomicsViewV2,
    ) -> Result<(), EngineMoneyErrorV2> {
        if self.schema_version != ENGINE_MONEY_SCHEMA_VERSION_V2 {
            return Err(money_error(
                EngineMoneyErrorCodeV2::UnsupportedSchemaVersion,
                "filled execution report is not schema V2",
            ));
        }
        validate_sha256("fill identity", &self.fill_identity_sha256)?;
        validate_sha256(
            "execution economics ledger",
            &self.execution_economics_ledger_sha256,
        )?;
        validate_position_text("position id", &self.position_id)?;
        validate_position_text("symbol", &self.symbol)?;
        self.actual_filled_lots.validate()?;
        self.fill_price.validate()?;
        if self.filled_at_unix_ms <= 0 {
            return Err(money_error(
                EngineMoneyErrorCodeV2::InvalidTimestamp,
                "fill timestamp must be strictly positive",
            ));
        }
        if self.execution_economics_ledger_sha256 != economics_ledger.ledger_sha256()
            || self.symbol != economics_ledger.symbol()
            || self.actual_filled_lots.get().to_bits() != economics_ledger.filled_lots().to_bits()
        {
            return Err(money_error(
                EngineMoneyErrorCodeV2::EconomicsLedgerMismatch,
                "fill report does not match its exact execution economics ledger",
            ));
        }
        let (expected_fill_identity, expected_fill_price, expected_fill_timestamp_unix_ms) =
            match self.fill_side {
                FillSideV2::Entry => (
                    economics_ledger.entry_fill_identity_sha256(),
                    economics_ledger.modeled_entry_price(),
                    economics_ledger.entry_fill_timestamp_unix_ms(),
                ),
                FillSideV2::Exit => (
                    economics_ledger.exit_fill_identity_sha256(),
                    economics_ledger.modeled_exit_price(),
                    economics_ledger.exit_fill_timestamp_unix_ms(),
                ),
            };
        if self.fill_identity_sha256 != expected_fill_identity
            || self.fill_price.get().to_bits() != expected_fill_price.to_bits()
            || self.filled_at_unix_ms != expected_fill_timestamp_unix_ms
        {
            return Err(money_error(
                EngineMoneyErrorCodeV2::EconomicsLedgerMismatch,
                "fill identity, price or timestamp differs from the sealed-ledger economics",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoneyUnavailableReasonV2 {
    MissingMark,
    MissingConversionEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "availability", content = "evidence", rename_all = "snake_case")]
pub enum MoneyAvailabilityV2 {
    Available(AccountMoneyV1),
    Unavailable(MoneyUnavailableReasonV2),
}

impl MoneyAvailabilityV2 {
    fn validate_for_currency(&self, account_currency: &str) -> Result<(), EngineMoneyErrorV2> {
        match self {
            Self::Available(money) => {
                validate_money_currency("available money", money, account_currency)
            }
            Self::Unavailable(_) => Ok(()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineMoneyStatsV2 {
    schema_version: u32,
    account_currency: String,
    quote_execution_economics_ledger_sha256: String,
    realized_pnl_account_currency: AccountMoneyV1,
    unrealized_pnl_account_currency: MoneyAvailabilityV2,
    balance_account_currency: AccountMoneyV1,
    equity_account_currency: MoneyAvailabilityV2,
    entry_commission_account_currency: AccountMoneyV1,
    exit_commission_account_currency: AccountMoneyV1,
    artifact_class: ExecutionEconomicsArtifactClassV1,
    promotion_eligibility: ExecutionEconomicsPromotionEligibilityV1,
}

impl EngineMoneyStatsV2 {
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn account_currency(&self) -> &str {
        &self.account_currency
    }

    pub fn quote_execution_economics_ledger_sha256(&self) -> &str {
        &self.quote_execution_economics_ledger_sha256
    }

    pub fn realized_pnl_account_currency(&self) -> &AccountMoneyV1 {
        &self.realized_pnl_account_currency
    }

    pub fn unrealized_pnl_account_currency(&self) -> &MoneyAvailabilityV2 {
        &self.unrealized_pnl_account_currency
    }

    pub fn balance_account_currency(&self) -> &AccountMoneyV1 {
        &self.balance_account_currency
    }

    pub fn equity_account_currency(&self) -> &MoneyAvailabilityV2 {
        &self.equity_account_currency
    }

    pub fn entry_commission_account_currency(&self) -> &AccountMoneyV1 {
        &self.entry_commission_account_currency
    }

    pub fn exit_commission_account_currency(&self) -> &AccountMoneyV1 {
        &self.exit_commission_account_currency
    }

    pub const fn artifact_class(&self) -> ExecutionEconomicsArtifactClassV1 {
        self.artifact_class
    }

    pub const fn promotion_eligibility(&self) -> ExecutionEconomicsPromotionEligibilityV1 {
        self.promotion_eligibility
    }

    fn validate(&self) -> Result<(), EngineMoneyErrorV2> {
        if self.schema_version != ENGINE_MONEY_SCHEMA_VERSION_V2 {
            return Err(money_error(
                EngineMoneyErrorCodeV2::UnsupportedSchemaVersion,
                "engine money stats are not schema V2",
            ));
        }
        validate_sha256(
            "quote execution economics ledger",
            &self.quote_execution_economics_ledger_sha256,
        )?;
        for (label, money) in [
            ("realized PnL", &self.realized_pnl_account_currency),
            ("balance", &self.balance_account_currency),
            ("entry commission", &self.entry_commission_account_currency),
            ("exit commission", &self.exit_commission_account_currency),
        ] {
            validate_money_currency(label, money, &self.account_currency)?;
        }
        self.unrealized_pnl_account_currency
            .validate_for_currency(&self.account_currency)?;
        self.equity_account_currency
            .validate_for_currency(&self.account_currency)?;
        if self.artifact_class != ExecutionEconomicsArtifactClassV1::ResearchOnly
            || self.promotion_eligibility
                != ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible
        {
            return Err(money_error(
                EngineMoneyErrorCodeV2::EconomicsLedgerMismatch,
                "engine money V2 must remain research-only and not promotion eligible",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct MoneyPositionV2 {
    position_id: String,
    symbol: String,
    remaining_lots: f64,
}

#[derive(Clone, Debug)]
pub struct EngineMoneyStateV2 {
    account_currency: String,
    opening_balance_account_currency: AccountMoneyV1,
    realized_pnl_account_currency: AccountMoneyV1,
    entry_commission_account_currency: AccountMoneyV1,
    exit_commission_account_currency: AccountMoneyV1,
    positions: Vec<MoneyPositionV2>,
    latest_execution_economics_ledger_sha256: Option<String>,
}

impl EngineMoneyStateV2 {
    pub fn new(
        opening_balance_account_currency: AccountMoneyV1,
    ) -> Result<Self, EngineMoneyErrorV2> {
        let account_currency = opening_balance_account_currency.currency().to_owned();
        validate_money_currency(
            "opening balance",
            &opening_balance_account_currency,
            &account_currency,
        )?;
        Ok(Self {
            account_currency: account_currency.clone(),
            opening_balance_account_currency,
            realized_pnl_account_currency: account_money(&account_currency, 0.0)?,
            entry_commission_account_currency: account_money(&account_currency, 0.0)?,
            exit_commission_account_currency: account_money(&account_currency, 0.0)?,
            positions: Vec::new(),
            latest_execution_economics_ledger_sha256: None,
        })
    }

    pub fn account_currency(&self) -> &str {
        &self.account_currency
    }

    pub fn remaining_lots(&self, position_id: &str) -> Result<StandardLotsV1, EngineMoneyErrorV2> {
        let position = self
            .positions
            .iter()
            .find(|position| position.position_id == position_id)
            .ok_or_else(|| {
                money_error(
                    EngineMoneyErrorCodeV2::MissingPosition,
                    "position is absent from engine money V2 state",
                )
            })?;
        StandardLotsV1::new(position.remaining_lots)
    }

    pub fn stats_with_unrealized(
        &self,
        unrealized_pnl_account_currency: MoneyAvailabilityV2,
    ) -> Result<EngineMoneyStatsV2, EngineMoneyErrorV2> {
        unrealized_pnl_account_currency.validate_for_currency(&self.account_currency)?;
        let quote_execution_economics_ledger_sha256 = self
            .latest_execution_economics_ledger_sha256
            .as_deref()
            .ok_or_else(|| {
                money_error(
                    EngineMoneyErrorCodeV2::MissingEconomicsLedger,
                    "money stats require at least one paired execution economics ledger",
                )
            })?;
        let balance_account_currency = account_money(
            &self.account_currency,
            self.opening_balance_account_currency.amount()
                + self.realized_pnl_account_currency.amount(),
        )?;
        let equity_account_currency = match &unrealized_pnl_account_currency {
            MoneyAvailabilityV2::Available(unrealized) => {
                MoneyAvailabilityV2::Available(account_money(
                    &self.account_currency,
                    balance_account_currency.amount() + unrealized.amount(),
                )?)
            }
            MoneyAvailabilityV2::Unavailable(reason) => MoneyAvailabilityV2::Unavailable(*reason),
        };
        let stats = EngineMoneyStatsV2 {
            schema_version: ENGINE_MONEY_SCHEMA_VERSION_V2,
            account_currency: self.account_currency.clone(),
            quote_execution_economics_ledger_sha256: quote_execution_economics_ledger_sha256
                .to_owned(),
            realized_pnl_account_currency: self.realized_pnl_account_currency.clone(),
            unrealized_pnl_account_currency,
            balance_account_currency,
            equity_account_currency,
            entry_commission_account_currency: self.entry_commission_account_currency.clone(),
            exit_commission_account_currency: self.exit_commission_account_currency.clone(),
            artifact_class: ExecutionEconomicsArtifactClassV1::ResearchOnly,
            promotion_eligibility: ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible,
        };
        stats.validate()?;
        Ok(stats)
    }

    pub fn stats_without_mark(&self) -> Result<EngineMoneyStatsV2, EngineMoneyErrorV2> {
        self.stats_with_unrealized(MoneyAvailabilityV2::Unavailable(
            MoneyUnavailableReasonV2::MissingMark,
        ))
    }

    pub fn stats_without_conversion(&self) -> Result<EngineMoneyStatsV2, EngineMoneyErrorV2> {
        self.stats_with_unrealized(MoneyAvailabilityV2::Unavailable(
            MoneyUnavailableReasonV2::MissingConversionEvidence,
        ))
    }
}

pub fn apply_filled_execution_v2(
    state: &mut EngineMoneyStateV2,
    report: &FilledExecutionReportV2,
    economics_ledger: &QuoteValidatedExecutionEconomicsLedgerV1,
) -> Result<EngineMoneyStatsV2, EngineMoneyErrorV2> {
    let remaining_lots = state
        .positions
        .iter()
        .find(|position| position.position_id == report.position_id())
        .map(|position| position.remaining_lots);
    let execution_economics_ledger_sha256 = economics_ledger.ledger_sha256();
    let contract_evidence = (
        report.actual_filled_lots(),
        remaining_lots,
        execution_economics_ledger_sha256,
        economics_ledger.entry_commission_account_currency(),
        economics_ledger.exit_commission_account_currency(),
        economics_ledger.net_pnl_account_currency(),
    );
    let _ = contract_evidence;
    apply_filled_execution_view_v2(state, report, economics_ledger)
}

fn apply_filled_execution_view_v2(
    state: &mut EngineMoneyStateV2,
    report: &FilledExecutionReportV2,
    economics_ledger: &impl ExecutionEconomicsViewV2,
) -> Result<EngineMoneyStatsV2, EngineMoneyErrorV2> {
    report.validate_against_view(economics_ledger)?;
    if economics_ledger.artifact_class() != ExecutionEconomicsArtifactClassV1::ResearchOnly
        || economics_ledger.promotion_eligibility()
            != ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible
    {
        return Err(money_error(
            EngineMoneyErrorCodeV2::EconomicsLedgerMismatch,
            "trader money V2 accepts only research-only broker economics",
        ));
    }
    if economics_ledger.account_currency() != state.account_currency() {
        return Err(money_error(
            EngineMoneyErrorCodeV2::CurrencyMismatch,
            "execution economics account currency differs from trader state",
        ));
    }
    let actual_filled_lots = report.actual_filled_lots().get();
    if actual_filled_lots.to_bits() != economics_ledger.filled_lots().to_bits() {
        return Err(money_error(
            EngineMoneyErrorCodeV2::EconomicsLedgerMismatch,
            "actual filled lots differ from the paired economics ledger",
        ));
    }
    let execution_economics_ledger_sha256 = economics_ledger.ledger_sha256();
    if report.execution_economics_ledger_sha256() != execution_economics_ledger_sha256 {
        return Err(money_error(
            EngineMoneyErrorCodeV2::EconomicsLedgerMismatch,
            "filled report and economics ledger identities differ",
        ));
    }
    let entry_commission_account_currency = economics_ledger.entry_commission_account_currency();
    let exit_commission_account_currency = economics_ledger.exit_commission_account_currency();
    let net_pnl_account_currency = economics_ledger.net_pnl_account_currency();
    for (label, money) in [
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
        validate_money_currency(label, money, state.account_currency())?;
    }

    match report.fill_side() {
        FillSideV2::Entry => {
            let existing_index = state
                .positions
                .iter()
                .position(|position| position.position_id == report.position_id());
            let remaining_lots = if let Some(index) = existing_index {
                let position = &state.positions[index];
                if position.symbol != report.symbol() {
                    return Err(money_error(
                        EngineMoneyErrorCodeV2::InvalidPosition,
                        "partial entry symbol differs from the open money position",
                    ));
                }
                position.remaining_lots + actual_filled_lots
            } else {
                actual_filled_lots
            };
            if !remaining_lots.is_finite() {
                return Err(money_error(
                    EngineMoneyErrorCodeV2::InvalidLots,
                    "remaining lots overflow after applying the entry fill",
                ));
            }
            let next_entry_commission = account_money(
                state.account_currency(),
                state.entry_commission_account_currency.amount()
                    + entry_commission_account_currency.amount(),
            )?;
            let next_realized = account_money(
                state.account_currency(),
                state.realized_pnl_account_currency.amount()
                    - entry_commission_account_currency.amount(),
            )?;
            if let Some(index) = existing_index {
                state.positions[index].remaining_lots = remaining_lots;
            } else {
                state.positions.push(MoneyPositionV2 {
                    position_id: report.position_id().to_owned(),
                    symbol: report.symbol().to_owned(),
                    remaining_lots,
                });
            }
            state.entry_commission_account_currency = next_entry_commission;
            state.realized_pnl_account_currency = next_realized;
        }
        FillSideV2::Exit => {
            let position_index = state
                .positions
                .iter()
                .position(|position| position.position_id == report.position_id())
                .ok_or_else(|| {
                    money_error(
                        EngineMoneyErrorCodeV2::MissingPosition,
                        "exit fill has no matching open money position",
                    )
                })?;
            let position = &state.positions[position_index];
            if position.symbol != report.symbol() {
                return Err(money_error(
                    EngineMoneyErrorCodeV2::InvalidPosition,
                    "exit fill symbol differs from the open money position",
                ));
            }
            if actual_filled_lots > position.remaining_lots {
                return Err(money_error(
                    EngineMoneyErrorCodeV2::ExceedsRemainingLots,
                    "exit fill exceeds the position's remaining standard lots",
                ));
            }
            let remaining_lots =
                if actual_filled_lots.to_bits() == position.remaining_lots.to_bits() {
                    0.0
                } else {
                    position.remaining_lots - actual_filled_lots
                };
            let next_realized = account_money(
                state.account_currency(),
                state.realized_pnl_account_currency.amount()
                    + net_pnl_account_currency.amount()
                    + entry_commission_account_currency.amount(),
            )?;
            let next_exit_commission = account_money(
                state.account_currency(),
                state.exit_commission_account_currency.amount()
                    + exit_commission_account_currency.amount(),
            )?;
            if remaining_lots == 0.0 {
                state.positions.remove(position_index);
            } else {
                state.positions[position_index].remaining_lots = remaining_lots;
            }
            state.realized_pnl_account_currency = next_realized;
            state.exit_commission_account_currency = next_exit_commission;
        }
    }
    state.latest_execution_economics_ledger_sha256 =
        Some(execution_economics_ledger_sha256.to_owned());
    state.stats_without_mark()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyMonetaryAuthorityV2 {
    Refused,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyMoneyArtifactClassV2 {
    LoopDiagnosticsOnly,
}

pub fn try_from_legacy_exec_report(
    _legacy: &ExecReport,
) -> Result<FilledExecutionReportV2, EngineMoneyErrorV2> {
    let authority = LegacyMonetaryAuthorityV2::Refused;
    let artifact_class = LegacyMoneyArtifactClassV2::LoopDiagnosticsOnly;
    Err(money_error(
        EngineMoneyErrorCodeV2::LegacyMoneyWireRefused,
        format!("legacy ExecReport is {artifact_class:?} with monetary authority {authority:?}"),
    ))
}

pub fn try_from_legacy_engine_stats(
    _legacy: &EngineStats,
) -> Result<EngineMoneyStatsV2, EngineMoneyErrorV2> {
    let authority = LegacyMonetaryAuthorityV2::Refused;
    let artifact_class = LegacyMoneyArtifactClassV2::LoopDiagnosticsOnly;
    Err(money_error(
        EngineMoneyErrorCodeV2::LegacyMoneyWireRefused,
        format!("legacy EngineStats is {artifact_class:?} with monetary authority {authority:?}"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EconomicsFixtureV2 {
        account_currency: String,
        symbol: String,
        filled_lots: f64,
        modeled_entry_price: f64,
        modeled_exit_price: f64,
        entry_fill_timestamp_unix_ms: i64,
        exit_fill_timestamp_unix_ms: i64,
        entry_fill_identity_sha256: String,
        exit_fill_identity_sha256: String,
        ledger_sha256: String,
        entry_commission_account_currency: AccountMoneyV1,
        exit_commission_account_currency: AccountMoneyV1,
        net_pnl_account_currency: AccountMoneyV1,
    }

    impl ExecutionEconomicsViewV2 for EconomicsFixtureV2 {
        fn account_currency(&self) -> &str {
            &self.account_currency
        }

        fn symbol(&self) -> &str {
            &self.symbol
        }

        fn filled_lots(&self) -> f64 {
            self.filled_lots
        }

        fn modeled_entry_price(&self) -> f64 {
            self.modeled_entry_price
        }

        fn modeled_exit_price(&self) -> f64 {
            self.modeled_exit_price
        }

        fn entry_fill_timestamp_unix_ms(&self) -> i64 {
            self.entry_fill_timestamp_unix_ms
        }

        fn exit_fill_timestamp_unix_ms(&self) -> i64 {
            self.exit_fill_timestamp_unix_ms
        }

        fn entry_fill_identity_sha256(&self) -> &str {
            &self.entry_fill_identity_sha256
        }

        fn exit_fill_identity_sha256(&self) -> &str {
            &self.exit_fill_identity_sha256
        }

        fn ledger_sha256(&self) -> &str {
            &self.ledger_sha256
        }

        fn entry_commission_account_currency(&self) -> &AccountMoneyV1 {
            &self.entry_commission_account_currency
        }

        fn exit_commission_account_currency(&self) -> &AccountMoneyV1 {
            &self.exit_commission_account_currency
        }

        fn net_pnl_account_currency(&self) -> &AccountMoneyV1 {
            &self.net_pnl_account_currency
        }

        fn artifact_class(&self) -> ExecutionEconomicsArtifactClassV1 {
            ExecutionEconomicsArtifactClassV1::ResearchOnly
        }

        fn promotion_eligibility(&self) -> ExecutionEconomicsPromotionEligibilityV1 {
            ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible
        }
    }

    fn fixture(
        filled_lots: f64,
        entry_commission: f64,
        exit_commission: f64,
        net_pnl: f64,
        digest_byte: char,
    ) -> EconomicsFixtureV2 {
        EconomicsFixtureV2 {
            account_currency: "USD".to_owned(),
            symbol: "EURUSD".to_owned(),
            filled_lots,
            modeled_entry_price: 1.1000,
            modeled_exit_price: 1.1010,
            entry_fill_timestamp_unix_ms: 1_700_000_000_100,
            exit_fill_timestamp_unix_ms: 1_700_000_000_200,
            entry_fill_identity_sha256: digest_byte.to_string().repeat(64),
            exit_fill_identity_sha256: digest_byte.to_string().repeat(64),
            ledger_sha256: digest_byte.to_string().repeat(64),
            entry_commission_account_currency: AccountMoneyV1::new("USD", entry_commission)
                .expect("valid entry commission fixture"),
            exit_commission_account_currency: AccountMoneyV1::new("USD", exit_commission)
                .expect("valid exit commission fixture"),
            net_pnl_account_currency: AccountMoneyV1::new("USD", net_pnl)
                .expect("valid net-PnL fixture"),
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn report_derives_exact_fill_timestamp_and_refuses_tamper() {
        let economics = fixture(1.0, 7.0, 7.0, 86.0, 'a');
        let mut report = FilledExecutionReportV2::from_economics_view(
            "position-1",
            FillSideV2::Entry,
            &economics,
        )
        .expect("fixture report must build");
        assert_eq!(
            report.filled_at_unix_ms(),
            economics.entry_fill_timestamp_unix_ms
        );

        report.filled_at_unix_ms += 1;
        let error = report
            .validate_against_view(&economics)
            .expect_err("caller-chosen timestamp must fail closed");
        assert_eq!(
            error.code(),
            EngineMoneyErrorCodeV2::EconomicsLedgerMismatch
        );
    }

    #[test]
    fn entry_and_partial_exit_cashflows_keep_intermediate_and_final_balance_exact() {
        let mut state = EngineMoneyStateV2::new(
            AccountMoneyV1::new("USD", 1_000.0).expect("valid opening balance"),
        )
        .expect("state must build");

        let entry_economics = fixture(1.0, 7.0, 7.0, 86.0, 'a');
        let entry_report = FilledExecutionReportV2::from_economics_view(
            "position-1",
            FillSideV2::Entry,
            &entry_economics,
        )
        .expect("entry report must build");
        let entry_stats =
            apply_filled_execution_view_v2(&mut state, &entry_report, &entry_economics)
                .expect("entry fill must apply");
        assert_close(entry_stats.realized_pnl_account_currency().amount(), -7.0);
        assert_close(entry_stats.balance_account_currency().amount(), 993.0);
        assert_close(
            entry_stats.entry_commission_account_currency().amount(),
            7.0,
        );

        let first_exit_economics = fixture(0.6, 4.2, 4.2, 51.6, 'b');
        let first_exit_report = FilledExecutionReportV2::from_economics_view(
            "position-1",
            FillSideV2::Exit,
            &first_exit_economics,
        )
        .expect("first exit report must build");
        let first_exit_stats =
            apply_filled_execution_view_v2(&mut state, &first_exit_report, &first_exit_economics)
                .expect("first partial exit must apply");
        assert_close(
            state
                .remaining_lots("position-1")
                .expect("position remains open")
                .get(),
            0.4,
        );
        assert_close(
            first_exit_stats.realized_pnl_account_currency().amount(),
            48.8,
        );
        assert_close(
            first_exit_stats.balance_account_currency().amount(),
            1_048.8,
        );
        assert_close(
            first_exit_stats.exit_commission_account_currency().amount(),
            4.2,
        );

        let final_exit_economics = fixture(0.4, 2.8, 2.8, 34.4, 'c');
        let final_exit_report = FilledExecutionReportV2::from_economics_view(
            "position-1",
            FillSideV2::Exit,
            &final_exit_economics,
        )
        .expect("final exit report must build");
        let final_stats =
            apply_filled_execution_view_v2(&mut state, &final_exit_report, &final_exit_economics)
                .expect("final partial exit must apply");
        assert_close(final_stats.realized_pnl_account_currency().amount(), 86.0);
        assert_close(final_stats.balance_account_currency().amount(), 1_086.0);
        assert_close(final_stats.exit_commission_account_currency().amount(), 7.0);
        assert_eq!(
            state
                .remaining_lots("position-1")
                .expect_err("fully closed position must be absent")
                .code(),
            EngineMoneyErrorCodeV2::MissingPosition
        );
    }
}
