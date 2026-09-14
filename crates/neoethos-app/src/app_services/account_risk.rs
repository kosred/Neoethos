//! Process-wide, account-scoped ownership of live risk state.
//!
//! A live strategy engine is a portfolio worker; it is not a broker account.
//! All workers trading the same `(environment, account_id)` therefore share one
//! [`AccountRiskAuthority`] for prop-firm drawdown/sizing state, while
//! [`AccountEntryAuthority`] is the one mode-independent durable daily-entry
//! counter used by both prop-firm and Risky Mode. Mutable state is atomically
//! checkpointed before an entry can leave the process, so a second worker or an
//! application restart cannot reset either authority.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Datelike, TimeZone, Utc};
use neoethos_core::domain::prop_firm::PropFirmPreset;
use neoethos_core::domain::risk::{
    ClosedTrade, PositionSizingInput, RiskManager, RiskManagerCheckpointV1, TradeGateInput,
};
use serde::{Deserialize, Serialize};

const ACCOUNT_RISK_CHECKPOINT_SCHEMA_V1: u32 = 1;
const ACCOUNT_RISK_CHECKPOINT_SCHEMA_V2: u32 = 2;
const ACCOUNT_ENTRY_CHECKPOINT_SCHEMA_V1: u32 = 1;
const ACCOUNT_ENTRY_CHECKPOINT_SCHEMA_V2: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountRiskIdentity {
    pub environment: String,
    pub account_id: i64,
    pub account_currency: String,
}

impl AccountRiskIdentity {
    pub fn new(environment: &str, account_id: i64, account_currency: &str) -> Result<Self> {
        let environment = environment.trim().to_ascii_lowercase();
        if !matches!(environment.as_str(), "demo" | "live") {
            return Err(anyhow!(
                "account-risk environment must be `demo` or `live`; got {environment:?}"
            ));
        }
        if account_id <= 0 {
            return Err(anyhow!(
                "account-risk account id must be positive; got {account_id}"
            ));
        }
        let account_currency = account_currency.trim().to_ascii_uppercase();
        if account_currency.is_empty() {
            return Err(anyhow!("account-risk currency is empty"));
        }
        Ok(Self {
            environment,
            account_id,
            account_currency,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccountRiskSnapshot {
    pub balance: f64,
    pub equity: f64,
}

impl AccountRiskSnapshot {
    fn validate(self) -> Result<()> {
        if !self.balance.is_finite()
            || self.balance <= 0.0
            || !self.equity.is_finite()
            || self.equity <= 0.0
        {
            return Err(anyhow!(
                "account-risk snapshot requires positive finite balance/equity; got {}/{}",
                self.balance,
                self.equity
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropFirmPeriod {
    pub day_id: u32,
    pub month_id: u32,
    pub start_utc_ms: i64,
    pub reset_zone: &'static str,
}

/// Resolve the firm's current calendar period and its exact UTC boundary.
pub fn prop_firm_period(preset: PropFirmPreset, now: DateTime<Utc>) -> Result<PropFirmPeriod> {
    let (date, start_utc, reset_zone) = match preset {
        PropFirmPreset::Ftmo => {
            let zone = chrono_tz::Europe::Prague;
            let date = now.with_timezone(&zone).date_naive();
            let local_midnight = date
                .and_hms_opt(0, 0, 0)
                .context("construct FTMO local midnight")?;
            let start = zone
                .from_local_datetime(&local_midnight)
                .single()
                .context("FTMO midnight is ambiguous or nonexistent in Europe/Prague")?
                .with_timezone(&Utc);
            (date, start, "Europe/Prague (CE(S)T)")
        }
        _ => {
            let date = now.date_naive();
            let start = date
                .and_hms_opt(0, 0, 0)
                .context("construct UTC midnight")?
                .and_utc();
            (date, start, "UTC (preset has no authoritative reset zone)")
        }
    };
    let day_id = (date.year().max(0) as u32) * 10_000 + date.month() * 100 + date.day();
    let month_id = (date.year().max(0) as u32) * 100 + date.month();
    Ok(PropFirmPeriod {
        day_id,
        month_id,
        start_utc_ms: start_utc.timestamp_millis(),
        reset_zone,
    })
}

/// Broker proof used only when no durable anchor exists for the active day.
/// Any deal, cash flow, or truncated deal result makes the anchor unknown and
/// blocks entries for that day; the code never guesses a midnight balance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccountRiskAnchorEvidence {
    pub account_id: i64,
    pub from_utc_ms: i64,
    pub to_utc_ms: i64,
    pub deals_seen: usize,
    pub deals_has_more: bool,
    pub cash_flows_seen: usize,
}

impl AccountRiskAnchorEvidence {
    fn prove_unchanged_since(
        &self,
        identity: &AccountRiskIdentity,
        period: PropFirmPeriod,
    ) -> Result<()> {
        if self.account_id != identity.account_id
            || self.from_utc_ms != period.start_utc_ms
            || self.to_utc_ms < self.from_utc_ms
        {
            return Err(anyhow!(
                "daily-anchor evidence identity/window mismatch: account {}, window {}..{}, expected account {}, start {}",
                self.account_id,
                self.from_utc_ms,
                self.to_utc_ms,
                identity.account_id,
                period.start_utc_ms
            ));
        }
        if self.deals_seen != 0 || self.deals_has_more || self.cash_flows_seen != 0 {
            return Err(anyhow!(
                "daily balance anchor is unrecoverable: no persisted checkpoint exists for day {}, and the broker reports {} deal row(s), hasMore={}, and {} cash-flow row(s) since {}. Refusing new entries until a day boundary can be observed without prior balance-changing activity",
                period.day_id,
                self.deals_seen,
                self.deals_has_more,
                self.cash_flows_seen,
                period.start_utc_ms
            ));
        }
        Ok(())
    }
}

/// Query the authoritative broker endpoints needed to prove that current
/// balance still equals the balance at the firm's reset boundary.
pub fn fetch_anchor_evidence_blocking(
    identity: &AccountRiskIdentity,
    period: PropFirmPeriod,
    to_utc_ms: i64,
) -> Result<AccountRiskAnchorEvidence> {
    if to_utc_ms < period.start_utc_ms {
        return Err(anyhow!(
            "daily-anchor evidence window ends before it starts: {} < {}",
            to_utc_ms,
            period.start_utc_ms
        ));
    }
    // One row is enough: this path needs proof of absence, not a financial
    // reconstruction. `hasMore` is preserved so truncation can never look empty.
    let deals = crate::app_services::broker_api::fetch_broker_deal_history_blocking(
        period.start_utc_ms,
        to_utc_ms,
        1,
    )?;
    let cash_flows = crate::app_services::broker_api::fetch_broker_cash_flow_history_blocking(
        period.start_utc_ms,
        to_utc_ms,
    )?;
    if deals.account_id != identity.account_id || cash_flows.account_id != identity.account_id {
        return Err(anyhow!(
            "daily-anchor evidence belongs to deal/cash-flow accounts {}/{}, expected {}",
            deals.account_id,
            cash_flows.account_id,
            identity.account_id
        ));
    }
    Ok(AccountRiskAnchorEvidence {
        account_id: identity.account_id,
        from_utc_ms: period.start_utc_ms,
        to_utc_ms,
        deals_seen: deals.deals.len(),
        deals_has_more: deals.has_more,
        cash_flows_seen: cash_flows.entries.len(),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AccountRiskPolicyV1 {
    preset: PropFirmPreset,
    challenge_phase: String,
    challenge_mode: bool,
    recovery_mode_enabled: bool,
    max_total_loss_pct: f64,
    daily_dd_stop_trading_pct: f64,
    daily_dd_warning_pct: f64,
    max_risk_per_trade: f64,
    min_confidence_threshold: f64,
    monthly_profit_target_pct: f64,
    challenge_target_return_pct: f64,
    initial_balance: f64,
}

impl AccountRiskPolicyV1 {
    fn from_manager(manager: &RiskManager) -> Self {
        Self {
            preset: manager.preset,
            challenge_phase: manager.challenge_phase.clone(),
            challenge_mode: manager.challenge_mode,
            recovery_mode_enabled: manager.recovery_mode_enabled,
            max_total_loss_pct: manager.max_total_loss_pct,
            daily_dd_stop_trading_pct: manager.daily_dd_stop_trading_pct,
            daily_dd_warning_pct: manager.daily_dd_warning_pct,
            max_risk_per_trade: manager.max_risk_per_trade,
            min_confidence_threshold: manager.min_confidence_threshold,
            monthly_profit_target_pct: manager.monthly_profit_target_pct,
            challenge_target_return_pct: manager.challenge_target_return_pct,
            initial_balance: manager.initial_balance,
        }
    }
}

/// Legacy combined checkpoint. Read only to migrate the entry counter into its
/// own authority before rewriting the prop-firm state as V2.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AccountRiskCheckpointFileV1 {
    schema_version: u32,
    identity: AccountRiskIdentity,
    policy: AccountRiskPolicyV1,
    manager: RiskManagerCheckpointV1,
    entry_day_id: Option<u32>,
    entries_reserved_or_filled: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AccountRiskCheckpointFileV2 {
    schema_version: u32,
    identity: AccountRiskIdentity,
    policy: AccountRiskPolicyV1,
    manager: RiskManagerCheckpointV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AccountEntryCheckpointFileV1 {
    schema_version: u32,
    identity: AccountRiskIdentity,
    day_id: Option<u32>,
    entries_reserved_or_filled: u32,
}

/// This extends the existing entry checkpoint, not a new journal or permission.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AccountEntryCheckpointFileV2 {
    schema_version: u32,
    identity: AccountRiskIdentity,
    day_id: Option<u32>,
    entries_reserved_or_filled: u32,
    unresolved_intent: Option<AccountEntryIntent>,
    last_verified_entry: Option<VerifiedEntryReference>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AccountEntryIntent {
    client_order_id: String,
    symbol_id: i64,
    trade_side: String,
    volume_raw_centi_units: i64,
    reserved_day_id: u32,
    created_at_utc_ms: i64,
    accepted_order_id: Option<i64>,
}

impl AccountEntryIntent {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.client_order_id.is_empty()
                && self.client_order_id.len() <= 50
                && !self.client_order_id.contains('\0')
                && self.symbol_id > 0
                && matches!(self.trade_side.as_str(), "BUY" | "SELL")
                && self.volume_raw_centi_units > 0
                && self.reserved_day_id > 0
                && self.created_at_utc_ms > 0
                && self.accepted_order_id.is_none_or(|id| id > 0),
            "invalid persisted account entry intent"
        );
        Ok(())
    }
}

/// Bounded diagnostic linkage only, NOT restoration of owned position state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct VerifiedEntryReference {
    intent: AccountEntryIntent,
    order_id: i64,
    position_id: i64,
    deal_id: i64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CheckpointSchema {
    schema_version: u32,
}

#[derive(Debug, Clone)]
pub struct AccountRiskSummary {
    pub preset: PropFirmPreset,
    pub challenge_phase: String,
    pub challenge_mode: bool,
    pub recovery_mode_enabled: bool,
    pub daily_drawdown_limit: f64,
    pub total_drawdown_limit: f64,
    pub max_risk_per_trade: f64,
    pub phase_advisory_max_risk_per_trade: f64,
    pub min_confidence_threshold: f64,
    pub day_start_balance: f64,
    pub last_session_date_id: Option<u32>,
    pub recovery_mode: bool,
    pub circuit_breaker_latched: bool,
    pub revenge_window: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccountRiskRefusal {
    pub rule: &'static str,
    pub detail: String,
}

impl AccountRiskRefusal {
    fn new(rule: &'static str, detail: impl Into<String>) -> Self {
        Self {
            rule,
            detail: detail.into(),
        }
    }
}

pub struct AccountRiskAuthority {
    identity: AccountRiskIdentity,
    checkpoint_path: PathBuf,
    policy: AccountRiskPolicyV1,
    manager: RiskManager,
}

impl AccountRiskAuthority {
    fn checkpoint(&self) -> AccountRiskCheckpointFileV2 {
        AccountRiskCheckpointFileV2 {
            schema_version: ACCOUNT_RISK_CHECKPOINT_SCHEMA_V2,
            identity: self.identity.clone(),
            policy: self.policy.clone(),
            manager: self.manager.checkpoint_v1(),
        }
    }

    fn persist(&self) -> Result<()> {
        if let Some(parent) = self.checkpoint_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        neoethos_core::storage::json::write_json_atomic(&self.checkpoint_path, &self.checkpoint())
            .with_context(|| format!("persist account risk at {}", self.checkpoint_path.display()))
    }

    pub fn needs_period(&self, period: PropFirmPeriod) -> bool {
        self.manager.last_session_date_id != Some(period.day_id)
            || self.manager.last_month_id != Some(period.month_id)
    }

    /// Prepare or refresh the current firm-local period. A new period is
    /// accepted only with broker evidence proving current balance still equals
    /// the reset-boundary balance.
    pub fn prepare_period(
        &mut self,
        period: PropFirmPeriod,
        snapshot: AccountRiskSnapshot,
        anchor_evidence: Option<&AccountRiskAnchorEvidence>,
    ) -> Result<()> {
        snapshot.validate()?;
        if self.needs_period(period) {
            anchor_evidence
                .context("account risk has no durable anchor for the active firm-local day")?
                .prove_unchanged_since(&self.identity, period)?;
            self.manager.roll_periods(
                period.day_id,
                period.month_id,
                snapshot.balance,
                snapshot.equity,
            );
        } else {
            self.manager.roll_periods(
                period.day_id,
                period.month_id,
                snapshot.balance,
                snapshot.equity,
            );
        }
        self.persist()
    }

    pub fn gate_and_size(
        &mut self,
        period: PropFirmPeriod,
        mut input: TradeGateInput,
        base_risk_pct: f64,
        entries_before: usize,
    ) -> Result<f64, AccountRiskRefusal> {
        if self.needs_period(period) {
            return Err(AccountRiskRefusal::new(
                "risk.daily_anchor_unprepared",
                format!(
                    "account risk is not prepared for firm-local day {}",
                    period.day_id
                ),
            ));
        }
        input.entries_today = entries_before;
        let decision = self.manager.check_trade_allowed(input);
        if let Err(error) = self.persist() {
            return Err(AccountRiskRefusal::new(
                "risk.state_persistence",
                format!("risk decision could not be persisted before order send: {error:#}"),
            ));
        }
        if let Err(refusal) = decision {
            return Err(AccountRiskRefusal::new(refusal.rule, refusal.detail));
        }
        let allowed = self.manager.calculate_position_size(PositionSizingInput {
            equity: input.equity,
            base_risk_pct,
        });
        if let Err(error) = self.persist() {
            return Err(AccountRiskRefusal::new(
                "risk.state_persistence",
                format!("risk size state could not be persisted before order send: {error:#}"),
            ));
        }
        Ok(base_risk_pct.min(allowed))
    }

    pub fn record_closed_trade(&mut self, trade: ClosedTrade) -> Result<usize> {
        self.manager.record_closed_trade(trade);
        self.persist()?;
        Ok(self.manager.revenge_detector.tracked())
    }

    pub fn summary(&self) -> AccountRiskSummary {
        AccountRiskSummary {
            preset: self.manager.preset,
            challenge_phase: self.manager.challenge_phase.clone(),
            challenge_mode: self.manager.challenge_mode,
            recovery_mode_enabled: self.manager.recovery_mode_enabled,
            daily_drawdown_limit: self.manager.daily_dd_stop_trading_pct,
            total_drawdown_limit: self.manager.max_total_loss_pct,
            max_risk_per_trade: self.manager.max_risk_per_trade,
            phase_advisory_max_risk_per_trade: self.manager.phase_max_risk_per_trade,
            min_confidence_threshold: self.manager.min_confidence_threshold,
            day_start_balance: self.manager.day_start_balance,
            last_session_date_id: self.manager.last_session_date_id,
            recovery_mode: self.manager.recovery_mode,
            circuit_breaker_latched: self.manager.circuit_breaker_triggered,
            revenge_window: self.manager.revenge_detector.tracked(),
        }
    }
}

pub type SharedAccountRiskAuthority = Arc<Mutex<AccountRiskAuthority>>;

/// The single durable, account-scoped daily-entry counter used by every live
/// mode. The caller supplies the authoritative day id (UTC for Risky Mode or
/// the firm's local day for prop-firm mode); the authority never invents a
/// reset boundary.
pub struct AccountEntryAuthority {
    identity: AccountRiskIdentity,
    checkpoint_path: PathBuf,
    day_id: Option<u32>,
    entries_reserved_or_filled: u32,
    unresolved_intent: Option<AccountEntryIntent>,
    last_verified_entry: Option<VerifiedEntryReference>,
    /// Process-local proof link to the existing attempt marker. Never restored from disk.
    current_submission_marker: Option<Arc<AtomicBool>>,
}

impl AccountEntryAuthority {
    fn checkpoint(&self) -> AccountEntryCheckpointFileV2 {
        AccountEntryCheckpointFileV2 {
            schema_version: ACCOUNT_ENTRY_CHECKPOINT_SCHEMA_V2,
            identity: self.identity.clone(),
            day_id: self.day_id,
            entries_reserved_or_filled: self.entries_reserved_or_filled,
            unresolved_intent: self.unresolved_intent.clone(),
            last_verified_entry: self.last_verified_entry.clone(),
        }
    }

    fn persist(&self) -> Result<()> {
        if let Some(parent) = self.checkpoint_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        neoethos_core::storage::json::write_json_atomic(&self.checkpoint_path, &self.checkpoint())
            .with_context(|| {
                format!(
                    "persist account entry state at {}",
                    self.checkpoint_path.display()
                )
            })
    }

    /// Select the caller-proved accounting day. A changed day resets only the
    /// count; unresolved intent survives every rollover and still blocks entry.
    pub fn prepare_day(&mut self, day_id: u32) -> Result<u32> {
        if day_id == 0 {
            return Err(anyhow!("account-entry day id must be non-zero"));
        }
        if self.day_id == Some(day_id) {
            return Ok(self.entries_reserved_or_filled);
        }
        let previous_day = self.day_id;
        let previous_count = self.entries_reserved_or_filled;
        self.day_id = Some(day_id);
        self.entries_reserved_or_filled = 0;
        if let Err(error) = self.persist() {
            self.day_id = previous_day;
            self.entries_reserved_or_filled = previous_count;
            return Err(error);
        }
        Ok(0)
    }

    /// Reserve and persist one entry before order work begins. A crash may
    /// conservatively over-count an unfilled reservation, but cannot under-count
    /// a filled order.
    pub fn try_reserve_entry(
        &mut self,
        day_id: u32,
        cap: Option<u32>,
    ) -> Result<usize, AccountRiskRefusal> {
        if let Some(intent) = &self.unresolved_intent {
            return Err(AccountRiskRefusal::new(
                "risk.unresolved_entry",
                format!(
                    "entry {} remains unresolved; broker reconciliation is required, no retry",
                    intent.client_order_id
                ),
            ));
        }
        if self.day_id != Some(day_id) {
            return Err(AccountRiskRefusal::new(
                "risk.daily_entry_period_unprepared",
                format!(
                    "account entry state is prepared for day {:?}, not {day_id}",
                    self.day_id
                ),
            ));
        }
        if let Some(cap) = cap
            && self.entries_reserved_or_filled >= cap
        {
            return Err(AccountRiskRefusal::new(
                "risk.max_trades_per_day",
                format!(
                    "account-wide daily entry cap reached: {}/{} on day {}",
                    self.entries_reserved_or_filled, cap, day_id
                ),
            ));
        }
        let entries_before = self.entries_reserved_or_filled as usize;
        self.entries_reserved_or_filled = self.entries_reserved_or_filled.saturating_add(1);
        if let Err(error) = self.persist() {
            self.entries_reserved_or_filled = self.entries_reserved_or_filled.saturating_sub(1);
            return Err(AccountRiskRefusal::new(
                "risk.state_persistence",
                format!("entry reservation could not be persisted: {error:#}"),
            ));
        }
        Ok(entries_before)
    }

    /// Release only a definitely unsent reservation. Ambiguous sends are not
    /// releasable by the daily-counter API.
    pub fn release_entry(&mut self, day_id: u32) -> Result<()> {
        anyhow::ensure!(
            self.unresolved_intent.is_none(),
            "cannot release an account entry while submission remains unresolved"
        );
        if self.day_id != Some(day_id) {
            return Err(anyhow!(
                "cannot release account-entry reservation for day {day_id}; active day is {:?}",
                self.day_id
            ));
        }
        if self.entries_reserved_or_filled == 0 {
            return Err(anyhow!(
                "cannot release account-entry reservation for day {day_id}; the durable count is already zero"
            ));
        }
        self.entries_reserved_or_filled -= 1;
        if let Err(error) = self.persist() {
            self.entries_reserved_or_filled = self.entries_reserved_or_filled.saturating_add(1);
            return Err(error);
        }
        Ok(())
    }

    /// Only the completed local attempt may cancel its definitely unsent intent.
    /// A fresh false marker cannot release restored uncertainty or another attempt.
    pub(crate) fn release_unsent_entry(
        &mut self,
        reserved_day_id: u32,
        client_order_id: &str,
        submission_marker: &Arc<AtomicBool>,
    ) -> Result<()> {
        anyhow::ensure!(
            !submission_marker.load(Ordering::Acquire),
            "cannot release an entry after its executor may have started"
        );
        let Some(intent) = self.unresolved_intent.as_ref() else {
            return self.release_entry(reserved_day_id);
        };
        anyhow::ensure!(
            intent.client_order_id == client_order_id
                && intent.reserved_day_id == reserved_day_id
                && intent.accepted_order_id.is_none()
                && self
                    .current_submission_marker
                    .as_ref()
                    .is_some_and(|bound| Arc::ptr_eq(bound, submission_marker)),
            "unsent release lacks the exact current-attempt client/day/marker proof"
        );
        anyhow::ensure!(self.day_id.is_some(), "unsent intent has no accounting day");
        let previous_count = self.entries_reserved_or_filled;
        if self.day_id == Some(reserved_day_id) {
            anyhow::ensure!(
                previous_count > 0,
                "unsent intent reservation count is already zero"
            );
            self.entries_reserved_or_filled -= 1;
        }
        // prepare_day already dropped the old day's counter on rollover. Never
        // subtract this old reservation from the new day's independent count.
        let pending = self.unresolved_intent.take();
        let marker = self.current_submission_marker.take();
        if let Err(error) = self.persist() {
            self.entries_reserved_or_filled = previous_count;
            self.unresolved_intent = pending;
            self.current_submission_marker = marker;
            return Err(error);
        }
        Ok(())
    }

    /// Bind the exact outgoing account/environment/client/order facts before
    /// the single backend invocation. No credentials are persisted.
    pub(crate) fn begin_submission(
        &mut self,
        day_id: u32,
        request: &super::ctrader_execution::CTraderExecutionRuntimeRequest,
        submission_marker: &Arc<AtomicBool>,
    ) -> Result<()> {
        use super::ctrader_execution::CTraderExecutionRequest;
        use super::ctrader_live_auth::CTraderEnvironment;
        use super::ctrader_messages::CTraderOrderType;
        anyhow::ensure!(
            !submission_marker.load(Ordering::Acquire),
            "cannot begin a durable intent after the executor may have started"
        );
        let CTraderExecutionRequest::NewOrder(order) = &request.request else {
            anyhow::bail!("entry intent requires a new order");
        };
        let environment = match request.environment {
            CTraderEnvironment::Demo => "demo",
            CTraderEnvironment::Live => "live",
        };
        anyhow::ensure!(
            environment == self.identity.environment
                && request.account_id.parse::<i64>()? == self.identity.account_id
                && order.account_id == self.identity.account_id
                && matches!(
                    order.order_type,
                    CTraderOrderType::Market | CTraderOrderType::MarketRange
                ),
            "entry intent differs from its exact account/environment authority"
        );
        let intent = AccountEntryIntent {
            client_order_id: order
                .client_order_id
                .clone()
                .context("entry requires clientOrderId")?,
            symbol_id: order.symbol_id,
            trade_side: order.trade_side.label().to_owned(),
            volume_raw_centi_units: order.volume,
            reserved_day_id: day_id,
            created_at_utc_ms: Utc::now().timestamp_millis(),
            accepted_order_id: None,
        };
        self.begin_intent(intent, Some(submission_marker.clone()))
    }

    fn begin_intent(
        &mut self,
        intent: AccountEntryIntent,
        marker: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        intent.validate()?;
        anyhow::ensure!(
            self.unresolved_intent.is_none()
                && self.day_id == Some(intent.reserved_day_id)
                && self.entries_reserved_or_filled > 0,
            "entry intent has no current unused reservation or another intent remains unresolved"
        );
        self.unresolved_intent = Some(intent);
        self.current_submission_marker = marker;
        // Keep the in-memory block even on an ambiguous persistence failure.
        // The caller does not reach execute unless this durable write succeeds.
        self.persist()
    }

    pub(crate) fn unresolved_client_order_id(&self) -> Option<&str> {
        self.unresolved_intent
            .as_ref()
            .map(|intent| intent.client_order_id.as_str())
    }

    /// Carry typed collector evidence only; never parse order ids from error text.
    /// No context/known id means uncertainty remains with the original client id.
    pub(crate) fn record_unresolved_error(
        &mut self,
        client_order_id: &str,
        error: &anyhow::Error,
    ) -> Result<()> {
        let Some(context) =
            error.downcast_ref::<super::ctrader_execution::CTraderUnresolvedExecution>()
        else {
            return Ok(());
        };
        self.record_unresolved_context(client_order_id, context)
    }

    /// Preserve a request-validated Filled/NETTED order reference without claiming
    /// it opened an owned position. The submitting call supplies its pinned environment.
    pub(crate) fn record_unresolved_outcome(
        &mut self,
        client_order_id: &str,
        environment: super::ctrader_live_auth::CTraderEnvironment,
        outcome: &super::ctrader_execution::CTraderExecutionOutcome,
    ) -> Result<()> {
        let intent = self
            .unresolved_intent
            .as_ref()
            .context("missing durable entry intent")?;
        anyhow::ensure!(
            outcome.symbol_id == Some(intent.symbol_id)
                && outcome.trade_side.as_deref() == Some(intent.trade_side.as_str()),
            "unresolved outcome symbol/side differs from the durable entry intent"
        );
        self.record_unresolved_context(
            client_order_id,
            &super::ctrader_execution::CTraderUnresolvedExecution {
                environment,
                account_id: outcome.account_id,
                client_order_id: Some(client_order_id.to_owned()),
                accepted_order_id: outcome.order_id,
            },
        )
    }

    fn record_unresolved_context(
        &mut self,
        client_order_id: &str,
        context: &super::ctrader_execution::CTraderUnresolvedExecution,
    ) -> Result<()> {
        let intent = self
            .unresolved_intent
            .as_mut()
            .context("missing durable entry intent")?;
        let environment = match context.environment {
            super::ctrader_live_auth::CTraderEnvironment::Demo => "demo",
            super::ctrader_live_auth::CTraderEnvironment::Live => "live",
        };
        anyhow::ensure!(
            context.account_id == self.identity.account_id
                && environment == self.identity.environment
                && context.client_order_id.as_deref() == Some(client_order_id)
                && intent.client_order_id == client_order_id,
            "unresolved execution context differs from the durable entry intent"
        );
        if let Some(order_id) = context.accepted_order_id {
            anyhow::ensure!(
                order_id > 0 && intent.accepted_order_id.is_none_or(|id| id == order_id),
                "conflicting accepted order identity; retain unresolved intent"
            );
            intent.accepted_order_id = Some(order_id);
            self.persist()?;
        }
        Ok(())
    }

    /// Called only after live's unchanged exact opening/account/volume proof.
    /// Clearing uncertainty is not reconstruction of the position after restart.
    pub(crate) fn confirm_verified_opening(
        &mut self,
        client_order_id: &str,
        opening: &super::ctrader_execution::CTraderOpeningFillEvidenceV1,
    ) -> Result<()> {
        let intent = self
            .unresolved_intent
            .as_ref()
            .context("missing durable entry intent")?;
        anyhow::ensure!(
            intent.client_order_id == client_order_id
                && opening.account_id() == self.identity.account_id
                && opening.symbol_id() == intent.symbol_id
                && opening.trade_side() == intent.trade_side
                && opening.filled_volume_raw_centi_units() == intent.volume_raw_centi_units
                && intent
                    .accepted_order_id
                    .is_none_or(|id| id == opening.order_id()),
            "verified opening differs from the durable entry intent"
        );
        let completed = VerifiedEntryReference {
            intent: intent.clone(),
            order_id: opening.order_id(),
            position_id: opening.position_id(),
            deal_id: opening.deal_id(),
        };
        self.persist_verified_reference(completed)
    }

    fn persist_verified_reference(&mut self, completed: VerifiedEntryReference) -> Result<()> {
        anyhow::ensure!(
            self.unresolved_intent.as_ref() == Some(&completed.intent)
                && completed.order_id > 0
                && completed.position_id > 0
                && completed.deal_id > 0
                && completed
                    .intent
                    .accepted_order_id
                    .is_none_or(|id| id == completed.order_id),
            "verified reference no longer matches the unresolved intent"
        );
        let previous = self.last_verified_entry.replace(completed);
        let pending = self.unresolved_intent.take();
        if let Err(error) = self.persist() {
            self.unresolved_intent = pending;
            self.last_verified_entry = previous;
            return Err(error);
        }
        self.current_submission_marker = None;
        Ok(())
    }

    pub fn day_id(&self) -> Option<u32> {
        self.day_id
    }

    pub fn entries_today(&self) -> u32 {
        self.entries_reserved_or_filled
    }
}

pub type SharedAccountEntryAuthority = Arc<Mutex<AccountEntryAuthority>>;

/// Ownership key. Currency is broker identity evidence, not a namespace: two
/// currencies reported for one environment/account must conflict rather than
/// create two authorities that write the same checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AccountRiskKey {
    environment: String,
    account_id: i64,
}

impl From<&AccountRiskIdentity> for AccountRiskKey {
    fn from(identity: &AccountRiskIdentity) -> Self {
        Self {
            environment: identity.environment.clone(),
            account_id: identity.account_id,
        }
    }
}

#[derive(Default)]
pub struct AccountRiskRegistry {
    risk_authorities: Mutex<HashMap<AccountRiskKey, SharedAccountRiskAuthority>>,
    entry_authorities: Mutex<HashMap<AccountRiskKey, SharedAccountEntryAuthority>>,
}

impl AccountRiskRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the one mode-independent daily-entry authority for this exact
    /// broker account. Only a zero-count legacy checkpoint can be migrated;
    /// positive V1 counts require reconciliation without rewriting either file.
    pub fn acquire_entry(
        &self,
        identity: AccountRiskIdentity,
        data_dir: &Path,
    ) -> Result<SharedAccountEntryAuthority> {
        let checkpoint_path = data_dir.join("runtime").join("account-entry").join(format!(
            "{}-{}.json",
            identity.environment, identity.account_id
        ));
        let legacy_risk_path = data_dir.join("runtime").join("account-risk").join(format!(
            "{}-{}.json",
            identity.environment, identity.account_id
        ));
        let key = AccountRiskKey::from(&identity);
        let mut registry = self
            .entry_authorities
            .lock()
            .map_err(|_| anyhow!("account-entry registry lock is poisoned"))?;

        if let Some(existing) = registry.get(&key) {
            let existing_guard = existing
                .lock()
                .map_err(|_| anyhow!("account-entry authority lock is poisoned"))?;
            if existing_guard.identity != identity {
                return Err(anyhow!(
                    "broker identity changed for {}/account {}: active currency is {}, new currency is {}; refusing a second account-entry authority",
                    identity.environment,
                    identity.account_id,
                    existing_guard.identity.account_currency,
                    identity.account_currency
                ));
            }
            if existing_guard.checkpoint_path != checkpoint_path {
                return Err(anyhow!(
                    "account {} already has an entry authority at {}, not {}",
                    identity.account_id,
                    existing_guard.checkpoint_path.display(),
                    checkpoint_path.display()
                ));
            }
            drop(existing_guard);
            return Ok(existing.clone());
        }

        let mut authority = AccountEntryAuthority {
            identity: identity.clone(),
            checkpoint_path: checkpoint_path.clone(),
            day_id: None,
            entries_reserved_or_filled: 0,
            unresolved_intent: None,
            last_verified_entry: None,
            current_submission_marker: None,
        };
        let mut must_persist = !checkpoint_path.exists();
        if checkpoint_path.exists() {
            let raw = std::fs::read(&checkpoint_path)
                .with_context(|| format!("read {}", checkpoint_path.display()))?;
            let schema: CheckpointSchema = serde_json::from_slice(&raw)
                .with_context(|| format!("read schema from {}", checkpoint_path.display()))?;
            let checkpoint = match schema.schema_version {
                ACCOUNT_ENTRY_CHECKPOINT_SCHEMA_V1 => {
                    let legacy: AccountEntryCheckpointFileV1 = serde_json::from_slice(&raw)
                        .with_context(|| format!("parse {}", checkpoint_path.display()))?;
                    anyhow::ensure!(
                        legacy.entries_reserved_or_filled == 0,
                        "legacy account-entry V1 has positive entries; unresolved submission cannot be excluded; reconciliation required before migration"
                    );
                    // Compatibility only: a zero legacy counter is not proof of
                    // historical non-exposure or a completed reconciliation.
                    must_persist = true;
                    AccountEntryCheckpointFileV2 {
                        schema_version: ACCOUNT_ENTRY_CHECKPOINT_SCHEMA_V2,
                        identity: legacy.identity,
                        day_id: legacy.day_id,
                        entries_reserved_or_filled: legacy.entries_reserved_or_filled,
                        unresolved_intent: None,
                        last_verified_entry: None,
                    }
                }
                ACCOUNT_ENTRY_CHECKPOINT_SCHEMA_V2 => {
                    let fields: serde_json::Value = serde_json::from_slice(&raw)?;
                    anyhow::ensure!(
                        fields.get("unresolvedIntent").is_some()
                            && fields.get("lastVerifiedEntry").is_some(),
                        "account-entry V2 omitted recovery state; refusing an implicit empty intent"
                    );
                    serde_json::from_slice(&raw)
                        .with_context(|| format!("parse {}", checkpoint_path.display()))?
                }
                version => anyhow::bail!("unsupported account-entry checkpoint schema {version}"),
            };
            if checkpoint.identity != identity {
                return Err(anyhow!(
                    "account-entry checkpoint identity {:?} differs from active {:?}",
                    checkpoint.identity,
                    identity
                ));
            }
            if let Some(intent) = &checkpoint.unresolved_intent {
                intent.validate()?;
            }
            if let Some(reference) = &checkpoint.last_verified_entry {
                reference.intent.validate()?;
                anyhow::ensure!(
                    reference.order_id > 0 && reference.position_id > 0 && reference.deal_id > 0,
                    "invalid persisted verified entry reference"
                );
            }
            authority.day_id = checkpoint.day_id;
            authority.entries_reserved_or_filled = checkpoint.entries_reserved_or_filled;
            authority.unresolved_intent = checkpoint.unresolved_intent;
            authority.last_verified_entry = checkpoint.last_verified_entry;
        } else if legacy_risk_path.exists() {
            let raw = std::fs::read(&legacy_risk_path)
                .with_context(|| format!("read {}", legacy_risk_path.display()))?;
            let schema: CheckpointSchema = serde_json::from_slice(&raw)
                .with_context(|| format!("read schema from {}", legacy_risk_path.display()))?;
            match schema.schema_version {
                ACCOUNT_RISK_CHECKPOINT_SCHEMA_V1 => {
                    let legacy: AccountRiskCheckpointFileV1 = serde_json::from_slice(&raw)
                        .with_context(|| format!("parse {}", legacy_risk_path.display()))?;
                    if legacy.identity != identity {
                        return Err(anyhow!(
                            "legacy account-risk checkpoint identity {:?} differs from active {:?}",
                            legacy.identity,
                            identity
                        ));
                    }
                    anyhow::ensure!(
                        legacy.entries_reserved_or_filled == 0,
                        "legacy combined account-risk V1 has positive entries; unresolved submission cannot be excluded; reconciliation required before migration"
                    );
                    // Preserve the original file on refusal. Zero-count migration
                    // does not certify the absence of pre-upgrade exposure.
                    authority.day_id = legacy.entry_day_id;
                    authority.entries_reserved_or_filled = legacy.entries_reserved_or_filled;
                    must_persist = true;
                }
                ACCOUNT_RISK_CHECKPOINT_SCHEMA_V2 => {
                    return Err(anyhow!(
                        "account-risk V2 exists at {} but its required account-entry checkpoint {} is missing; refusing to reset an unknown live entry count",
                        legacy_risk_path.display(),
                        checkpoint_path.display()
                    ));
                }
                version => {
                    return Err(anyhow!(
                        "unsupported account-risk checkpoint schema {version} while recovering the daily-entry count"
                    ));
                }
            }
        }
        if must_persist {
            authority.persist()?;
        }

        let shared = Arc::new(Mutex::new(authority));
        registry.insert(key, shared.clone());
        Ok(shared)
    }

    /// Return the one authority for this account, restoring its durable state
    /// on first use. A policy or storage-path mismatch is refused instead of
    /// running two interpretations of one account in parallel.
    pub fn acquire(
        &self,
        identity: AccountRiskIdentity,
        settings: &neoethos_core::Settings,
        snapshot: AccountRiskSnapshot,
        data_dir: &Path,
    ) -> Result<SharedAccountRiskAuthority> {
        snapshot.validate()?;
        // Entry migration must happen before a combined V1 risk checkpoint is
        // rewritten as V2, otherwise a restart could erase today's count.
        let _entry_authority = self.acquire_entry(identity.clone(), data_dir)?;
        let manager = RiskManager::from_settings(settings, snapshot.balance, snapshot.equity)
            .map_err(|error| anyhow!("build account-wide prop-firm manager: {error}"))?;
        let policy = AccountRiskPolicyV1::from_manager(&manager);
        let checkpoint_path = data_dir.join("runtime").join("account-risk").join(format!(
            "{}-{}.json",
            identity.environment, identity.account_id
        ));
        let key = AccountRiskKey::from(&identity);

        let mut registry = self
            .risk_authorities
            .lock()
            .map_err(|_| anyhow!("account-risk registry lock is poisoned"))?;
        if let Some(existing) = registry.get(&key) {
            let existing_guard = existing
                .lock()
                .map_err(|_| anyhow!("account-risk authority lock is poisoned"))?;
            if existing_guard.identity != identity {
                return Err(anyhow!(
                    "broker identity changed for {}/account {}: active currency is {}, new currency is {}; refusing a second account-risk authority",
                    identity.environment,
                    identity.account_id,
                    existing_guard.identity.account_currency,
                    identity.account_currency
                ));
            }
            if existing_guard.policy != policy {
                return Err(anyhow!(
                    "account {} already has a live risk authority under different settings; stop all live engines before changing risk policy",
                    identity.account_id
                ));
            }
            if existing_guard.checkpoint_path != checkpoint_path {
                return Err(anyhow!(
                    "account {} already has a live risk authority at {}, not {}",
                    identity.account_id,
                    existing_guard.checkpoint_path.display(),
                    checkpoint_path.display()
                ));
            }
            drop(existing_guard);
            return Ok(existing.clone());
        }

        let mut authority = AccountRiskAuthority {
            identity: identity.clone(),
            checkpoint_path: checkpoint_path.clone(),
            policy: policy.clone(),
            manager,
        };
        if checkpoint_path.exists() {
            let raw = std::fs::read(&checkpoint_path)
                .with_context(|| format!("read {}", checkpoint_path.display()))?;
            let schema: CheckpointSchema = serde_json::from_slice(&raw)
                .with_context(|| format!("read schema from {}", checkpoint_path.display()))?;
            let (checkpoint_identity, checkpoint_policy, manager_checkpoint, migrated_v1) =
                match schema.schema_version {
                    ACCOUNT_RISK_CHECKPOINT_SCHEMA_V1 => {
                        let checkpoint: AccountRiskCheckpointFileV1 = serde_json::from_slice(&raw)
                            .with_context(|| format!("parse {}", checkpoint_path.display()))?;
                        (
                            checkpoint.identity,
                            checkpoint.policy,
                            checkpoint.manager,
                            true,
                        )
                    }
                    ACCOUNT_RISK_CHECKPOINT_SCHEMA_V2 => {
                        let checkpoint: AccountRiskCheckpointFileV2 = serde_json::from_slice(&raw)
                            .with_context(|| format!("parse {}", checkpoint_path.display()))?;
                        (
                            checkpoint.identity,
                            checkpoint.policy,
                            checkpoint.manager,
                            false,
                        )
                    }
                    version => {
                        return Err(anyhow!(
                            "unsupported account-risk checkpoint schema {version}; expected {} or {}",
                            ACCOUNT_RISK_CHECKPOINT_SCHEMA_V1,
                            ACCOUNT_RISK_CHECKPOINT_SCHEMA_V2
                        ));
                    }
                };
            if checkpoint_identity != identity {
                return Err(anyhow!(
                    "account-risk checkpoint identity {:?} differs from active {:?}",
                    checkpoint_identity,
                    identity
                ));
            }
            if checkpoint_policy != policy {
                return Err(anyhow!(
                    "persisted risk policy for account {} differs from current settings; refusing to reset or reinterpret live account state",
                    identity.account_id
                ));
            }
            authority
                .manager
                .restore_checkpoint_v1(&manager_checkpoint)
                .map_err(|error| anyhow!("restore account-risk checkpoint: {error}"))?;
            if migrated_v1 {
                authority.persist()?;
            }
        }

        let shared = Arc::new(Mutex::new(authority));
        registry.insert(key, shared.clone());
        Ok(shared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neoethos_core::domain::risk::TradeGateInput;

    fn temp_root(name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("neoethos-account-risk-{name}-{unique}"))
    }

    fn settings() -> neoethos_core::Settings {
        let mut settings = neoethos_core::Settings::default();
        settings.risk.initial_balance = 10_000.0;
        settings.risk.challenge_mode = true;
        settings
    }

    fn identity() -> AccountRiskIdentity {
        AccountRiskIdentity::new("demo", 712_345, "USD").expect("identity")
    }

    fn no_activity(period: PropFirmPeriod) -> AccountRiskAnchorEvidence {
        AccountRiskAnchorEvidence {
            account_id: 712_345,
            from_utc_ms: period.start_utc_ms,
            to_utc_ms: period.start_utc_ms + 1_000,
            deals_seen: 0,
            deals_has_more: false,
            cash_flows_seen: 0,
        }
    }

    fn synthetic_intent() -> AccountEntryIntent {
        AccountEntryIntent {
            client_order_id: "neo-fixture-entry-1".to_owned(),
            symbol_id: 14,
            trade_side: "BUY".to_owned(),
            volume_raw_centi_units: 10_000_000,
            reserved_day_id: 20_260_908,
            created_at_utc_ms: 1_788_825_600_000,
            accepted_order_id: None,
        }
    }

    #[test]
    fn unresolved_intent_survives_restart_and_day_rollover_without_auto_retry() {
        let root = temp_root("unresolved-restart");
        {
            let registry = AccountRiskRegistry::new();
            let shared = registry.acquire_entry(identity(), &root).unwrap();
            let mut state = shared.lock().unwrap();
            state.prepare_day(20_260_908).unwrap();
            state.try_reserve_entry(20_260_908, None).unwrap();
            state.begin_intent(synthetic_intent(), None).unwrap();
            assert!(state.release_entry(20_260_908).is_err());
        }
        for day in [20_260_908, 20_260_909] {
            let registry = AccountRiskRegistry::new();
            let shared = registry.acquire_entry(identity(), &root).unwrap();
            let mut state = shared.lock().unwrap();
            state.prepare_day(day).unwrap();
            assert_eq!(
                state.unresolved_client_order_id(),
                Some("neo-fixture-entry-1")
            );
            assert_eq!(
                state.try_reserve_entry(day, None).unwrap_err().rule,
                "risk.unresolved_entry"
            );
            assert!(state.begin_intent(synthetic_intent(), None).is_err());
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn unsent_clear_refuses_foreign_started_accepted_and_restarted_attempts_without_writes() {
        for case in ["client", "day", "marker", "started", "accepted", "restart"] {
            let root = temp_root(&format!("unsent-proof-{case}"));
            let shared = AccountRiskRegistry::new()
                .acquire_entry(identity(), &root)
                .unwrap();
            let marker = Arc::new(AtomicBool::new(false));
            {
                let mut state = shared.lock().unwrap();
                state.prepare_day(20_260_908).unwrap();
                state.try_reserve_entry(20_260_908, None).unwrap();
                state
                    .begin_intent(synthetic_intent(), Some(marker.clone()))
                    .unwrap();
                if case == "accepted" {
                    state.unresolved_intent.as_mut().unwrap().accepted_order_id = Some(8001);
                    state.persist().unwrap();
                }
            }
            let tested = if case == "restart" {
                AccountRiskRegistry::new()
                    .acquire_entry(identity(), &root)
                    .unwrap()
            } else {
                shared.clone()
            };
            let mut state = tested.lock().unwrap();
            let before = std::fs::read(&state.checkpoint_path).unwrap();
            let supplied_marker = if case == "marker" {
                Arc::new(AtomicBool::new(false))
            } else {
                marker.clone()
            };
            if case == "started" {
                marker.store(true, Ordering::Release);
            }
            let day = if case == "day" {
                20_260_909
            } else {
                20_260_908
            };
            let client = if case == "client" {
                "other-client"
            } else {
                "neo-fixture-entry-1"
            };
            assert!(
                state
                    .release_unsent_entry(day, client, &supplied_marker)
                    .is_err(),
                "{case}"
            );
            assert_eq!(state.entries_today(), 1, "{case}");
            assert_eq!(
                state.unresolved_client_order_id(),
                Some("neo-fixture-entry-1"),
                "{case}"
            );
            assert_eq!(
                state.try_reserve_entry(20_260_908, None).unwrap_err().rule,
                "risk.unresolved_entry"
            );
            assert_eq!(
                std::fs::read(&state.checkpoint_path).unwrap(),
                before,
                "{case}"
            );
            drop(state);
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn completed_unsent_rollover_clears_only_old_intent_not_the_new_days_counter() {
        let root = temp_root("unsent-rollover");
        let shared = AccountRiskRegistry::new()
            .acquire_entry(identity(), &root)
            .unwrap();
        let marker = Arc::new(AtomicBool::new(false));
        {
            let mut state = shared.lock().unwrap();
            state.prepare_day(20_260_908).unwrap();
            state.try_reserve_entry(20_260_908, None).unwrap();
            state
                .begin_intent(synthetic_intent(), Some(marker.clone()))
                .unwrap();
            state.prepare_day(20_260_909).unwrap();
            assert_eq!(state.entries_today(), 0);
            state
                .release_unsent_entry(20_260_908, "neo-fixture-entry-1", &marker)
                .unwrap();
            assert_eq!(state.day_id(), Some(20_260_909));
            assert_eq!(state.entries_today(), 0);
            assert_eq!(state.unresolved_client_order_id(), None);
        }
        let restored = AccountRiskRegistry::new()
            .acquire_entry(identity(), &root)
            .unwrap();
        let mut state = restored.lock().unwrap();
        assert_eq!(state.day_id(), Some(20_260_909));
        assert_eq!(state.try_reserve_entry(20_260_909, Some(1)).unwrap(), 0);
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn failed_unsent_clear_restores_count_intent_and_marker_until_atomic_write_succeeds() {
        let root = temp_root("unsent-clear-write");
        let shared = AccountRiskRegistry::new()
            .acquire_entry(identity(), &root)
            .unwrap();
        let marker = Arc::new(AtomicBool::new(false));
        let mut state = shared.lock().unwrap();
        state.prepare_day(20_260_908).unwrap();
        state.try_reserve_entry(20_260_908, None).unwrap();
        state
            .begin_intent(synthetic_intent(), Some(marker.clone()))
            .unwrap();
        let original_path = state.checkpoint_path.clone();
        let before = std::fs::read(&original_path).unwrap();
        let blocker = root.join("not-a-directory");
        std::fs::write(&blocker, b"owned test blocker").unwrap();
        state.checkpoint_path = blocker.join("entry.json");
        assert!(
            state
                .release_unsent_entry(20_260_908, "neo-fixture-entry-1", &marker)
                .is_err()
        );
        assert_eq!(state.entries_today(), 1);
        assert_eq!(
            state.unresolved_client_order_id(),
            Some("neo-fixture-entry-1")
        );
        assert!(Arc::ptr_eq(
            state.current_submission_marker.as_ref().unwrap(),
            &marker
        ));
        assert_eq!(
            state.try_reserve_entry(20_260_908, None).unwrap_err().rule,
            "risk.unresolved_entry"
        );
        assert_eq!(std::fs::read(&original_path).unwrap(), before);
        let restored = AccountRiskRegistry::new()
            .acquire_entry(identity(), &root)
            .unwrap();
        assert_eq!(
            restored.lock().unwrap().unresolved_client_order_id(),
            Some("neo-fixture-entry-1")
        );
        state.checkpoint_path = original_path;
        state
            .release_unsent_entry(20_260_908, "neo-fixture-entry-1", &marker)
            .unwrap();
        assert_eq!(state.entries_today(), 0);
        assert!(state.current_submission_marker.is_none());
        assert_eq!(state.unresolved_client_order_id(), None);
        drop(state);
        let restored = AccountRiskRegistry::new()
            .acquire_entry(identity(), &root)
            .unwrap();
        assert_eq!(
            restored
                .lock()
                .unwrap()
                .try_reserve_entry(20_260_908, Some(1))
                .unwrap(),
            0
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn accepted_order_context_is_persisted_by_type_and_cannot_be_rebound() {
        use super::super::ctrader_execution::CTraderUnresolvedExecution;
        use super::super::ctrader_live_auth::CTraderEnvironment;
        let root = temp_root("typed-order-context");
        let registry = AccountRiskRegistry::new();
        let shared = registry.acquire_entry(identity(), &root).unwrap();
        {
            let mut state = shared.lock().unwrap();
            state.prepare_day(20_260_908).unwrap();
            state.try_reserve_entry(20_260_908, None).unwrap();
            state.begin_intent(synthetic_intent(), None).unwrap();
            // Matching words alone must not become authority.
            state
                .record_unresolved_error(
                    "neo-fixture-entry-1",
                    &anyhow!("accepted_order_id=Some(9999)"),
                )
                .unwrap();
            assert_eq!(
                state.unresolved_intent.as_ref().unwrap().accepted_order_id,
                None
            );
            let context = CTraderUnresolvedExecution {
                environment: CTraderEnvironment::Demo,
                account_id: 712345,
                client_order_id: Some("neo-fixture-entry-1".to_owned()),
                accepted_order_id: Some(8001),
            };
            for case in ["account", "environment", "client"] {
                let mut foreign = context.clone();
                match case {
                    "account" => foreign.account_id = 99,
                    "environment" => foreign.environment = CTraderEnvironment::Live,
                    _ => foreign.client_order_id = Some("other".into()),
                }
                let error = anyhow!("synthetic read failed").context(foreign);
                assert!(
                    state
                        .record_unresolved_error("neo-fixture-entry-1", &error)
                        .is_err(),
                    "{case}"
                );
            }
            let error = anyhow!("synthetic deadline").context(context);
            state
                .record_unresolved_error("neo-fixture-entry-1", &error)
                .unwrap();
        }
        let restored = AccountRiskRegistry::new()
            .acquire_entry(identity(), &root)
            .unwrap();
        assert_eq!(
            restored
                .lock()
                .unwrap()
                .unresolved_intent
                .as_ref()
                .unwrap()
                .accepted_order_id,
            Some(8001)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn failed_intent_write_is_unsent_and_failed_completion_keeps_the_block() {
        let root = temp_root("intent-persistence");
        let registry = AccountRiskRegistry::new();
        let shared = registry.acquire_entry(identity(), &root).unwrap();
        let mut state = shared.lock().unwrap();
        state.prepare_day(20_260_908).unwrap();
        state.try_reserve_entry(20_260_908, None).unwrap();
        let original_path = state.checkpoint_path.clone();
        let blocker = root.join("not-a-directory");
        std::fs::write(&blocker, b"owned test blocker").unwrap();
        state.checkpoint_path = blocker.join("entry.json");
        assert!(state.begin_intent(synthetic_intent(), None).is_err());
        assert!(
            state.unresolved_intent.is_some(),
            "persistence ambiguity stays blocked in memory"
        );
        state.checkpoint_path = original_path.clone();
        state.persist().unwrap();
        let completed = VerifiedEntryReference {
            intent: state.unresolved_intent.clone().unwrap(),
            order_id: 8001,
            position_id: 9001,
            deal_id: 10001,
        };
        state.checkpoint_path = blocker.join("entry.json");
        assert!(state.persist_verified_reference(completed.clone()).is_err());
        assert!(state.unresolved_intent.is_some());
        assert!(state.last_verified_entry.is_none());
        state.checkpoint_path = original_path;
        state.persist_verified_reference(completed).unwrap();
        assert!(state.unresolved_intent.is_none());
        assert_eq!(
            state.entries_today(),
            1,
            "a proven fill still consumes its reservation"
        );
        drop(state);
        let restored = AccountRiskRegistry::new()
            .acquire_entry(identity(), &root)
            .unwrap();
        let state = restored.lock().unwrap();
        assert!(state.unresolved_intent.is_none());
        assert_eq!(
            state.last_verified_entry.as_ref().unwrap().position_id,
            9001
        );
        // This reference is not a reconstruction of owned position/protection state.
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn zero_v1_entry_count_migrates_but_incomplete_v2_recovery_fields_are_refused() {
        let root = temp_root("entry-schema");
        let path = root
            .join("runtime")
            .join("account-entry")
            .join("demo-712345.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let legacy = AccountEntryCheckpointFileV1 {
            schema_version: ACCOUNT_ENTRY_CHECKPOINT_SCHEMA_V1,
            identity: identity(),
            day_id: Some(20_260_908),
            entries_reserved_or_filled: 0,
        };
        neoethos_core::storage::json::write_json_atomic(&path, &legacy).unwrap();
        let state = AccountRiskRegistry::new()
            .acquire_entry(identity(), &root)
            .unwrap();
        assert_eq!(state.lock().unwrap().entries_today(), 0);
        // This compatibility migration does not prove no historical exposure.
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["schemaVersion"], 2);
        for missing in ["unresolvedIntent", "lastVerifiedEntry"] {
            let mut incomplete = saved.clone();
            incomplete.as_object_mut().unwrap().remove(missing);
            neoethos_core::storage::json::write_json_atomic(&path, &incomplete).unwrap();
            assert!(
                AccountRiskRegistry::new()
                    .acquire_entry(identity(), &root)
                    .is_err(),
                "{missing}"
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn positive_legacy_entry_and_combined_counts_require_reconciliation_without_rewrite() {
        for combined in [false, true] {
            let root = temp_root(if combined {
                "legacy-combined-positive"
            } else {
                "legacy-entry-positive"
            });
            let entry_path = root
                .join("runtime")
                .join("account-entry")
                .join("demo-712345.json");
            let risk_path = root
                .join("runtime")
                .join("account-risk")
                .join("demo-712345.json");
            let original_path = if combined { &risk_path } else { &entry_path };
            std::fs::create_dir_all(original_path.parent().unwrap()).unwrap();
            let snapshot = AccountRiskSnapshot {
                balance: 10_000.0,
                equity: 10_000.0,
            };
            if combined {
                let manager =
                    RiskManager::from_settings(&settings(), snapshot.balance, snapshot.equity)
                        .unwrap();
                let legacy = AccountRiskCheckpointFileV1 {
                    schema_version: ACCOUNT_RISK_CHECKPOINT_SCHEMA_V1,
                    identity: identity(),
                    policy: AccountRiskPolicyV1::from_manager(&manager),
                    manager: manager.checkpoint_v1(),
                    entry_day_id: Some(20_200_101), // A stale day must not erase uncertainty.
                    entries_reserved_or_filled: 5,
                };
                neoethos_core::storage::json::write_json_atomic(original_path, &legacy).unwrap();
            } else {
                let legacy = AccountEntryCheckpointFileV1 {
                    schema_version: ACCOUNT_ENTRY_CHECKPOINT_SCHEMA_V1,
                    identity: identity(),
                    day_id: Some(20_200_101),
                    entries_reserved_or_filled: 5,
                };
                neoethos_core::storage::json::write_json_atomic(original_path, &legacy).unwrap();
            }
            let original = std::fs::read(original_path).unwrap();
            // Neither direct entry acquisition nor PropFirm acquisition may rewrite
            // the source, even when a later day would ordinarily reset its counter.
            for _restart in 0..2 {
                let registry = AccountRiskRegistry::new();
                let error = registry
                    .acquire_entry(identity(), &root)
                    .err()
                    .expect("legacy uncertainty");
                assert!(error.to_string().contains("reconciliation required"));
                let error = registry
                    .acquire(identity(), &settings(), snapshot, &root)
                    .err()
                    .expect("legacy uncertainty");
                assert!(error.to_string().contains("reconciliation required"));
                assert_eq!(std::fs::read(original_path).unwrap(), original);
                assert!(registry.entry_authorities.lock().unwrap().is_empty());
                assert!(registry.risk_authorities.lock().unwrap().is_empty());
                assert!(!if combined {
                    entry_path.exists()
                } else {
                    risk_path.exists()
                });
            }
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn ftmo_period_follows_prague_dst_and_exposes_exact_utc_midnight() {
        let before = DateTime::parse_from_rfc3339("2026-03-29T21:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let after = DateTime::parse_from_rfc3339("2026-03-29T22:30:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let before_period = prop_firm_period(PropFirmPreset::Ftmo, before).unwrap();
        let after_period = prop_firm_period(PropFirmPreset::Ftmo, after).unwrap();
        assert_eq!(before_period.day_id, 20_260_329);
        assert_eq!(after_period.day_id, 20_260_330);
        assert_eq!(after_period.start_utc_ms, 1_774_821_600_000);
        assert_eq!(after_period.reset_zone, "Europe/Prague (CE(S)T)");
    }

    #[test]
    fn registry_returns_one_shared_risk_and_entry_authority_per_account() {
        let root = temp_root("shared");
        let registry = AccountRiskRegistry::new();
        let snapshot = AccountRiskSnapshot {
            balance: 10_000.0,
            equity: 10_000.0,
        };
        let first = registry
            .acquire(identity(), &settings(), snapshot, &root)
            .unwrap();
        let second = registry
            .acquire(identity(), &settings(), snapshot, &root)
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let first_entry = registry.acquire_entry(identity(), &root).unwrap();
        let second_entry = registry.acquire_entry(identity(), &root).unwrap();
        assert!(Arc::ptr_eq(&first_entry, &second_entry));

        let different_currency = AccountRiskIdentity::new("demo", 712_345, "EUR").unwrap();
        let error = match registry.acquire(different_currency, &settings(), snapshot, &root) {
            Ok(_) => panic!("one account must not receive a second currency/risk authority"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("broker identity changed"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn checkpoints_restore_risk_state_and_the_separate_entry_count() {
        let root = temp_root("restore");
        let now = DateTime::parse_from_rfc3339("2026-08-30T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let period = prop_firm_period(PropFirmPreset::Ftmo, now).unwrap();
        let snapshot = AccountRiskSnapshot {
            balance: 10_000.0,
            equity: 10_000.0,
        };
        {
            let registry = AccountRiskRegistry::new();
            let shared = registry
                .acquire(identity(), &settings(), snapshot, &root)
                .unwrap();
            let entries = registry.acquire_entry(identity(), &root).unwrap();
            entries.lock().unwrap().prepare_day(period.day_id).unwrap();
            assert_eq!(
                entries
                    .lock()
                    .unwrap()
                    .try_reserve_entry(period.day_id, Some(8))
                    .unwrap(),
                0
            );
            let mut authority = shared.lock().unwrap();
            authority
                .prepare_period(period, snapshot, Some(&no_activity(period)))
                .unwrap();
            authority
                .record_closed_trade(ClosedTrade {
                    entry_time_sec: 100,
                    exit_time_sec: 200,
                    pnl: -10.0,
                    size: 0.01,
                    direction: Some(1),
                })
                .unwrap();
        }

        let restored_registry = AccountRiskRegistry::new();
        let restored = restored_registry
            .acquire(identity(), &settings(), snapshot, &root)
            .unwrap();
        let restored_entries = restored_registry.acquire_entry(identity(), &root).unwrap();
        let summary = restored.lock().unwrap().summary();
        assert_eq!(summary.last_session_date_id, Some(period.day_id));
        assert_eq!(summary.day_start_balance, 10_000.0);
        assert_eq!(summary.revenge_window, 1);
        assert_eq!(
            restored_entries.lock().unwrap().day_id(),
            Some(period.day_id)
        );
        assert_eq!(restored_entries.lock().unwrap().entries_today(), 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn missing_anchor_with_broker_activity_fails_closed() {
        let root = temp_root("activity");
        let now = DateTime::parse_from_rfc3339("2026-08-30T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let period = prop_firm_period(PropFirmPreset::Ftmo, now).unwrap();
        let snapshot = AccountRiskSnapshot {
            balance: 9_900.0,
            equity: 9_900.0,
        };
        let registry = AccountRiskRegistry::new();
        let shared = registry
            .acquire(identity(), &settings(), snapshot, &root)
            .unwrap();
        let evidence = AccountRiskAnchorEvidence {
            deals_seen: 1,
            ..no_activity(period)
        };
        let error = shared
            .lock()
            .unwrap()
            .prepare_period(period, snapshot, Some(&evidence))
            .unwrap_err();
        assert!(error.to_string().contains("anchor is unrecoverable"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn gate_uses_the_persisted_account_wide_entry_count() {
        let root = temp_root("gate");
        let now = DateTime::parse_from_rfc3339("2026-08-30T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let period = prop_firm_period(PropFirmPreset::Ftmo, now).unwrap();
        let snapshot = AccountRiskSnapshot {
            balance: 10_000.0,
            equity: 10_000.0,
        };
        let registry = AccountRiskRegistry::new();
        let shared = registry
            .acquire(identity(), &settings(), snapshot, &root)
            .unwrap();
        let entries = registry.acquire_entry(identity(), &root).unwrap();
        entries.lock().unwrap().prepare_day(period.day_id).unwrap();
        let entries_before = entries
            .lock()
            .unwrap()
            .try_reserve_entry(period.day_id, Some(8))
            .unwrap();
        assert_eq!(entries_before, 0);
        let mut authority = shared.lock().unwrap();
        authority
            .prepare_period(period, snapshot, Some(&no_activity(period)))
            .unwrap();
        let allowed = authority
            .gate_and_size(
                period,
                TradeGateInput {
                    balance: 10_000.0,
                    equity: 10_000.0,
                    confidence: None,
                    current_time_sec: now.timestamp() as u64,
                    current_hour: 10,
                    entries_today: usize::MAX,
                    open_positions: 0,
                },
                0.001,
                entries_before,
            )
            .unwrap();
        assert_eq!(allowed, 0.001);
        drop(authority);
        entries
            .lock()
            .unwrap()
            .release_entry(period.day_id)
            .unwrap();
        assert_eq!(entries.lock().unwrap().entries_today(), 0);
        assert!(
            entries
                .lock()
                .unwrap()
                .release_entry(period.day_id)
                .is_err(),
            "a reservation token must not be releasable twice"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn entry_count_survives_restart_and_resets_only_for_a_new_caller_proved_day() {
        let root = temp_root("entry-rollover");
        let first_day = 20_260_830;
        let second_day = 20_260_831;
        {
            let registry = AccountRiskRegistry::new();
            let entries = registry.acquire_entry(identity(), &root).unwrap();
            let mut guard = entries.lock().unwrap();
            assert_eq!(guard.prepare_day(first_day).unwrap(), 0);
            assert_eq!(guard.try_reserve_entry(first_day, Some(8)).unwrap(), 0);
            assert_eq!(guard.prepare_day(first_day).unwrap(), 1);
        }

        let restored_registry = AccountRiskRegistry::new();
        let restored = restored_registry.acquire_entry(identity(), &root).unwrap();
        let mut guard = restored.lock().unwrap();
        assert_eq!(guard.day_id(), Some(first_day));
        assert_eq!(guard.entries_today(), 1);
        assert_eq!(guard.prepare_day(second_day).unwrap(), 0);
        assert_eq!(guard.day_id(), Some(second_day));
        assert_eq!(guard.entries_today(), 0);
        drop(guard);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn zero_legacy_combined_checkpoint_migrates_before_risk_v2_rewrite() {
        let root = temp_root("legacy-migration");
        let risk_path = root
            .join("runtime")
            .join("account-risk")
            .join("demo-712345.json");
        std::fs::create_dir_all(risk_path.parent().unwrap()).unwrap();
        let snapshot = AccountRiskSnapshot {
            balance: 10_000.0,
            equity: 10_000.0,
        };
        let manager = RiskManager::from_settings(&settings(), snapshot.balance, snapshot.equity)
            .expect("manager");
        let legacy = AccountRiskCheckpointFileV1 {
            schema_version: ACCOUNT_RISK_CHECKPOINT_SCHEMA_V1,
            identity: identity(),
            policy: AccountRiskPolicyV1::from_manager(&manager),
            manager: manager.checkpoint_v1(),
            entry_day_id: Some(20_260_830),
            entries_reserved_or_filled: 0,
        };
        neoethos_core::storage::json::write_json_atomic(&risk_path, &legacy).unwrap();

        let registry = AccountRiskRegistry::new();
        let _risk = registry
            .acquire(identity(), &settings(), snapshot, &root)
            .unwrap();
        let entries = registry.acquire_entry(identity(), &root).unwrap();
        assert_eq!(entries.lock().unwrap().day_id(), Some(20_260_830));
        assert_eq!(entries.lock().unwrap().entries_today(), 0);

        let migrated_risk: CheckpointSchema =
            serde_json::from_slice(&std::fs::read(&risk_path).unwrap()).unwrap();
        assert_eq!(
            migrated_risk.schema_version,
            ACCOUNT_RISK_CHECKPOINT_SCHEMA_V2
        );
        let entry_path = root
            .join("runtime")
            .join("account-entry")
            .join("demo-712345.json");
        let migrated_entries: CheckpointSchema =
            serde_json::from_slice(&std::fs::read(entry_path).unwrap()).unwrap();
        assert_eq!(
            migrated_entries.schema_version,
            ACCOUNT_ENTRY_CHECKPOINT_SCHEMA_V2
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
