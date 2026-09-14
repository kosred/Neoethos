//! Live autonomous trading service (Path A).
//!
//! Polls the broker for new closed bars, computes features, evaluates gene
//! signals, and places/closes orders via cTrader.
//!
//! PARITY, STATED HONESTLY. Signal and exit geometry are shared/pinned; broker
//! execution is deliberately a different boundary:
//!
//! - **Direction: shared implementation.** Live and replay net the same genes with
//!   `neoethos_trader::combine_gene_signals_with_archived_policy`; the replay nets the
//!   same genes over the same artifact-bound feature cube.
//! - **Exits: shared policy and closed-bar causality.** Discovery seals its
//!   break-even/trailing geometry into live-portfolio schema v4; replay and this
//!   loop both consume that immutable value. Replay applies the prior-bar stop
//!   locally; live advances its local stop only after cTrader confirms the amend.
//! - **Execution: NOT shared.** The replay fills at the mark through
//!   `MockExecutionAdapter` behind a `PermissiveRiskGate`. This loop pays a
//!   real broker and passes every gate in this file.
//!
//! The parity that IS load-bearing, and that this file must not break, is
//! live-vs-**discovery**: same genes, same features, same exit policy.
//!
//! Entry point: [`start`].  The returned [`Handle`] stops the loop.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

mod candidate_models;

use anyhow::{Context, Result, anyhow};
use neoethos_data::Ohlcv;
use neoethos_trader::Direction;
use serde::{Deserialize, Serialize};

use crate::app_services::account_risk::{
    AccountRiskIdentity, AccountRiskRegistry, AccountRiskSnapshot, AccountRiskSummary,
    PropFirmPeriod, SharedAccountEntryAuthority, SharedAccountRiskAuthority,
    fetch_anchor_evidence_blocking, prop_firm_period,
};
use crate::app_services::broker_api::{
    OrderSide, amend_position_sltp_expecting_account, close_position_blocking,
    fetch_broker_symbols_blocking, fetch_live_entry_context_blocking,
    fetch_recent_broker_trendbar_snapshot_blocking,
};
use crate::app_services::broker_deal_economics::{
    BrokerDealWireSnapshotV1, BrokerPositionMoneyAccumulatorV1, BrokerSymbolVolumeScaleEvidenceV1,
    build_broker_deal_money_evidence_v1,
};
use crate::app_services::ctrader_execution::{CTraderExecutionOutcome, CTraderExecutionStatus};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::live_spots::{self, SpotQuoteRefusal, SpotSessionId};

/// One entry reserved in the durable account authority shared by every live
/// portfolio and both risk modes.
struct EntryReservation {
    authority: SharedAccountEntryAuthority,
    day_id: u32,
    entries_before: usize,
}

impl EntryReservation {
    fn entries_before(&self) -> usize {
        self.entries_before
    }

    /// Return an unused slot. Failure is deliberately loud and fail-safe: the
    /// durable counter may over-count after a persistence failure, but can
    /// never under-count a filled order.
    fn release(self) {
        self.release_with(None);
    }

    /// Called only after the submitting task completed and its marker is false.
    fn release_unsent(self, client_order_id: &str, marker: &Arc<AtomicBool>) {
        self.release_with(Some((client_order_id, marker)));
    }

    fn release_with(self, attempt: Option<(&str, &Arc<AtomicBool>)>) {
        match self.authority.lock() {
            Ok(mut guard) => {
                let released = match attempt {
                    Some((client_order_id, marker)) => {
                        guard.release_unsent_entry(self.day_id, client_order_id, marker)
                    }
                    None => guard.release_entry(self.day_id),
                };
                if let Err(error) = released {
                    tracing::error!(
                        target: "neoethos_app::live_trading",
                        accounting_day = self.day_id,
                        error = %error,
                        "unused account entry reservation could not be persisted as released; durable state remains conservative"
                    );
                }
            }
            Err(_) => tracing::error!(
                target: "neoethos_app::live_trading",
                accounting_day = self.day_id,
                "unused account entry reservation could not be released because the account-entry lock is poisoned; durable state remains conservative"
            ),
        }
    }
}

// ── Public request type ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct StartRequest {
    /// Absolute or config-relative path to a `*.live_portfolio.json` file.
    pub portfolio_path: String,
    /// Position size sent to the broker, in lots. Default 0.01.
    #[serde(default = "default_lot_size")]
    pub lot_size: f64,
    /// Stop-loss pips. Pass `null` / omit for naked positions (requires
    /// the caller to also set `risky: true` in the future risk gate).
    pub stop_loss_pips: Option<f64>,
    /// Take-profit pips.
    pub take_profit_pips: Option<f64>,
    /// How many bars to fetch per TF for feature warmup. Default 1000.
    #[serde(default = "default_warmup_bars")]
    pub warmup_bars: usize,
    /// Auto-cull: after this many CONSECUTIVE losing trades, the engine stops
    /// itself and permanently retires the strategy (blacklist). Default 0/off:
    /// a loss streak alone is not evidence of negative expectancy.
    #[serde(default = "default_cull_losses")]
    pub cull_after_consecutive_losses: u32,
    /// Auto-cull, rolling-window criterion: over the last `cull_window_trades`
    /// closed trades, the win rate must stay ≥ this percent or the strategy is
    /// retired. Default 0/off: break-even win rate depends on payoff and costs,
    /// so no universal 57% floor is imposed on every discovered strategy.
    #[serde(default = "default_cull_min_win_rate_pct")]
    pub cull_min_win_rate_pct: f64,
    /// Rolling window size (closed trades) for the win-rate criterion. The
    /// check only fires once the window is FULL. Default 10.
    #[serde(default = "default_cull_window_trades")]
    pub cull_window_trades: usize,
}

pub fn default_lot_size() -> f64 {
    0.01
}
pub fn default_warmup_bars() -> usize {
    1000
}
pub fn default_cull_losses() -> u32 {
    0
}
pub fn default_cull_min_win_rate_pct() -> f64 {
    0.0
}
pub fn default_cull_window_trades() -> usize {
    10
}

// ── Status ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveTradingStatus {
    pub running: bool,
    /// Which portfolio file this engine is running — lets the supervisor
    /// identify each concurrent engine and the UI label its row.
    pub portfolio_path: Option<String>,
    pub symbol: Option<String>,
    pub base_tf: Option<String>,
    pub genes: usize,
    pub last_signal: Option<String>,
    pub open_position_id: Option<i64>,
    pub bars_evaluated: u64,
    /// Identity of the immutable discovery policy controlling this position.
    pub protection_policy_identity: Option<String>,
    /// `None` before the live artifact is loaded; thereafter the exact value
    /// discovery priced, not the current Settings value.
    pub trailing_enabled: Option<bool>,
    /// Human-readable state: waiting, armed, broker-confirmed, pending, or
    /// disabled by the validated search policy.
    pub protection_state: Option<String>,
    pub position_entry_price: Option<f64>,
    pub initial_stop_pips: Option<f64>,
    pub favorable_extreme_price: Option<f64>,
    pub favorable_move_r: Option<f64>,
    /// Last stop price acknowledged by an exact cTrader `ORDER_REPLACED`
    /// response. `None` means no trailing amend has been broker-confirmed.
    pub confirmed_stop_price: Option<f64>,
    pub last_protection_error: Option<String>,
    /// Honest coarse exit attribution. Broker-side SL versus TP is not guessed
    /// when the reconcile payload does not prove which protection fired.
    pub last_exit_reason: Option<String>,
    /// Current run of consecutive losing trades (resets to 0 on any win).
    pub consecutive_losses: u32,
    /// Win rate (%) over the rolling cull window, once ≥1 trade closed.
    pub window_win_rate_pct: Option<f64>,
    /// How many closed trades the rolling window currently holds.
    pub window_trades: u32,
    /// True once auto-cull retired this strategy (engine stopped + blacklisted).
    pub retired: bool,
}

impl Default for LiveTradingStatus {
    fn default() -> Self {
        Self {
            running: false,
            portfolio_path: None,
            symbol: None,
            base_tf: None,
            genes: 0,
            last_signal: None,
            open_position_id: None,
            bars_evaluated: 0,
            protection_policy_identity: None,
            trailing_enabled: None,
            protection_state: None,
            position_entry_price: None,
            initial_stop_pips: None,
            favorable_extreme_price: None,
            favorable_move_r: None,
            confirmed_stop_price: None,
            last_protection_error: None,
            last_exit_reason: None,
            consecutive_losses: 0,
            window_win_rate_pct: None,
            window_trades: 0,
            retired: false,
        }
    }
}

impl LiveTradingStatus {
    fn clear_position_telemetry(&mut self, trailing_enabled: bool) {
        self.position_entry_price = None;
        self.initial_stop_pips = None;
        self.favorable_extreme_price = None;
        self.favorable_move_r = None;
        self.confirmed_stop_price = None;
        self.last_protection_error = None;
        self.protection_state = Some(if trailing_enabled {
            "waiting_for_position".to_string()
        } else {
            "disabled_by_search".to_string()
        });
    }
}

// ── Handle ────────────────────────────────────────────────────────────────────

/// Returned by [`start`]. Call [`Handle::stop`] to request a graceful shutdown.
pub struct Handle {
    stop_flag: Arc<AtomicBool>,
    pub status: Arc<std::sync::Mutex<LiveTradingStatus>>,
}

impl Handle {
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }

    pub fn is_running(&self) -> bool {
        self.status.lock().map(|s| s.running).unwrap_or(false)
    }

    pub fn snapshot(&self) -> LiveTradingStatus {
        self.status.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Spawn the live trading loop and return a [`Handle`].  Returns immediately.
///
/// On a Live account the optional demo check applies only when explicitly
/// enabled. Both environments still require validated portfolio evidence,
/// broker identity, valid protection and risk-budgeted sizing in the loop.
pub fn start(req: StartRequest, account_risk: Arc<AccountRiskRegistry>) -> Result<Handle> {
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::LiveTrading)
        .map_err(anyhow::Error::new)?;

    // 2026-08-09 (W2, second half): CAPTURE the environment this admission
    // decision is made against, and hand it to the loop.
    //
    // The defect: the gate was evaluated here and never again, while
    // `submit_market_order_blocking` re-reads `ctrader.environment` from disk on
    // EVERY order (`broker_api.rs:218`, `:270`). Starting on Demo — where the
    // gate is an unconditional pass — and then flipping the environment to Live
    // in Settings put a REAL-money order through a running engine that had been
    // admitted against a demo account, by a gate never re-consulted.
    let gated_env_is_live = crate::app_services::live_gate::active_env_is_live();
    if gated_env_is_live {
        let decision = crate::app_services::live_gate::evaluate_for_portfolio(&req.portfolio_path)
            .context("evaluate demo forward-test gate")?;
        if !decision.eligible {
            anyhow::bail!(
                "LIVE blocked by the demo forward-test gate — {} \
                 This optional check is enabled in models.demo_forward_gate.",
                decision.summary
            );
        }
    }

    let stop_flag = Arc::new(AtomicBool::new(false));
    let status = Arc::new(std::sync::Mutex::new(LiveTradingStatus {
        running: true,
        portfolio_path: Some(req.portfolio_path.clone()),
        ..Default::default()
    }));

    let stop_clone = stop_flag.clone();
    let status_clone = status.clone();

    tokio::spawn(async move {
        if let Err(e) = run(
            req,
            stop_clone,
            status_clone.clone(),
            gated_env_is_live,
            account_risk,
        )
        .await
        {
            if let Ok(mut s) = status_clone.lock() {
                s.last_signal = Some(format!("STOPPED: {e}"));
            }
            tracing::error!(
                target: "neoethos_app::live_trading",
                error = %e,
                "live trading loop exited with error"
            );
        }
        if let Ok(mut s) = status_clone.lock() {
            s.running = false;
        }
    });

    Ok(Handle { stop_flag, status })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn tf_duration_ms(tf: &str) -> Result<i64> {
    let timeframe = tf
        .parse::<neoethos_core::CanonicalTimeframe>()
        .map_err(|_| anyhow!("unsupported live-trading timeframe {tf:?}"))?;
    timeframe.fixed_duration_ms().ok_or_else(|| {
        anyhow!(
            "live trading for calendar timeframe {timeframe} is disabled until its exact \
             broker session/calendar wake-up rule is evidenced"
        )
    })
}

fn favorable_excursion_r(
    entry_price: f64,
    favorable_extreme_price: f64,
    initial_stop_pips: f64,
    pip_size: f64,
    is_long: bool,
) -> Option<f64> {
    if !entry_price.is_finite()
        || entry_price <= 0.0
        || !favorable_extreme_price.is_finite()
        || favorable_extreme_price <= 0.0
        || !initial_stop_pips.is_finite()
        || initial_stop_pips <= 0.0
        || !pip_size.is_finite()
        || pip_size <= 0.0
    {
        return None;
    }
    let risk_distance = initial_stop_pips * pip_size;
    let favorable_move = if is_long {
        favorable_extreme_price - entry_price
    } else {
        entry_price - favorable_extreme_price
    };
    Some(favorable_move.max(0.0) / risk_distance)
}

/// Both risk modes must size against the current admitted account. A failed
/// fetch is not evidence of zero positions or of an unchanged startup balance.
fn validate_entry_account_values(
    environment: &str,
    account_id: i64,
    balance: f64,
    equity: f64,
    expected_environment: &str,
    expected_account_id: i64,
) -> Result<()> {
    anyhow::ensure!(
        environment.eq_ignore_ascii_case(expected_environment)
            && account_id == expected_account_id
            && balance.is_finite()
            && balance > 0.0
            && equity.is_finite()
            && equity > 0.0,
        "current entry account snapshot has a foreign identity or invalid balance/equity"
    );
    Ok(())
}

/// Convert a broker response into engine-owned opening facts only after the
/// private wire proof and the submitting API's volume scope agree. This is not
/// a durable lifecycle, an admission permit, or proof of account-history coverage.
pub(super) fn verified_opening_for_engine<'a>(
    outcome: &'a CTraderExecutionOutcome,
    environment: &str,
    account_id: i64,
    symbol_id: i64,
    symbol: &str,
    side: OrderSide,
) -> Result<(
    &'a crate::app_services::ctrader_execution::CTraderOpeningFillEvidenceV1,
    &'a BrokerSymbolVolumeScaleEvidenceV1,
    f64,
)> {
    let opening = outcome
        .opening_fill_evidence
        .as_ref()
        .context("entry response has no complete single-opening broker proof")?;
    let scale = outcome
        .volume_scale_evidence
        .as_ref()
        .context("entry response has no exact broker volume-scale scope")?;
    let expected_side = match side {
        OrderSide::Buy => "BUY",
        OrderSide::Sell => "SELL",
    };
    anyhow::ensure!(
        outcome.status == CTraderExecutionStatus::Filled
            && outcome.deal_closes_position == Some(false)
            && opening.account_id() == account_id
            && outcome.account_id == account_id
            && opening.symbol_id() == symbol_id
            && outcome.symbol_id == Some(symbol_id)
            && opening.trade_side() == expected_side
            && outcome.trade_side.as_deref() == Some(expected_side)
            && outcome.order_id == Some(opening.order_id())
            && outcome.position_id == Some(opening.position_id())
            && outcome.deal_id == Some(opening.deal_id())
            && outcome.filled_volume_raw_centi_units
                == Some(opening.filled_volume_raw_centi_units())
            && outcome.timestamp_ms == Some(opening.execution_timestamp_ms())
            && outcome.execution_price.map(f64::to_bits) == Some(opening.entry_price().to_bits())
            && scale.environment() == environment
            && scale.account_id() == account_id
            && scale.symbol_id() == symbol_id
            && scale.symbol_name() == symbol,
        "entry outcome, opening proof and admitted broker identity disagree"
    );
    let actual_lots = crate::app_services::broker_deal_economics::broker_lots_from_wire_volume_v1(
        opening.filled_volume_raw_centi_units(),
        scale.lot_size_raw_centi_units(),
    )?;
    Ok((opening, scale, actual_lots))
}

/// Does this kill-switch tier justify the PERSISTED 24 h halt, or only a
/// refusal of the order in hand?
///
/// W3 (2026-08-09). The ledger's instruction was "call `record_kill_switch_trip`
/// on the `Err` branch". Doing that for every tier would start a 24 h
/// account-wide halt because one order arrived with a malformed bracket, and a
/// safety control the operator learns to distrust is worse than no control.
///
/// - **Account-level** (halt): `PerDay`, `PerWeek`, `PerStage`, `PerMonth` say
///   the bankroll itself is in trouble. Hardware/account disconnects reach the
///   same durable cooldown through `margin_call`, outside this enum.
///
/// **What can actually fire, as of 2026-08-09** — stated because the first
/// version of this wiring advertised five halting tiers and three of them were
/// structurally unreachable:
/// - `PerDay` — live. Realized loss this UTC day (account-wide, see the journal
///   ledger at the entry site) reached `daily_loss_cap_fraction × bankroll`.
/// - `PerWeek` — live. Realized loss this ISO week reached
///   `weekly_drawdown_cap_fraction × bankroll`; a day rollover cannot clear it.
/// - `PerStage` — live as of the high-water fix in
///   `neoethos_core::domain::risky_mode`. Was unreachable before it.
/// - `PerMonth` — live, but INERT at the shipped `monthly_loss_cap_fraction`
///   (0.99): the day cap always binds first. Lower the fraction to arm it.
/// - `PreSendSanity` — live, the most frequently seen refusal.
/// - `PerTrade` — reachable only via an explicit zero/absent bracket; this loop
///   always resolves an SL and a TP, so in practice it does not fire here.
/// - **Order-level** (refuse only): `PerTrade` (missing/invalid SL or TP) and
///   `PreSendSanity` (this order's implied risk exceeded the ceiling) describe
///   THIS order. Both are still refused, and both are logged at `error`.
pub(crate) fn tier_halts_for_24h(tier: neoethos_core::domain::risky_mode::KillSwitchTier) -> bool {
    use neoethos_core::domain::risky_mode::KillSwitchTier as T;
    match tier {
        T::PerDay | T::PerWeek | T::PerStage | T::PerMonth => true,
        T::PerTrade | T::PreSendSanity => false,
    }
}

/// Start-of-UTC-day / start-of-ISO-week / start-of-calendar-month, in epoch ms,
/// for the instant `now_ms`.
fn period_starts_ms(now_ms: i64) -> Option<(i64, i64, i64)> {
    use chrono::{Datelike, NaiveDate};
    let now = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(now_ms)?;
    let d = now.date_naive();
    let to_ms = |x: NaiveDate| {
        x.and_hms_opt(0, 0, 0)
            .map(|t| t.and_utc().timestamp_millis())
    };
    let day = to_ms(d)?;
    // Monday-anchored, matching `iso_week()` used for the weekly accumulator.
    let week = to_ms(d - chrono::Duration::days(d.weekday().num_days_from_monday() as i64))?;
    let month = to_ms(NaiveDate::from_ymd_opt(d.year(), d.month(), 1)?)?;
    Some((day, week, month))
}

fn utc_day_id(now: chrono::DateTime<chrono::Utc>) -> u32 {
    use chrono::Datelike;

    let date = now.date_naive();
    (date.year().max(0) as u32) * 10_000 + date.month() * 100 + date.day()
}

/// The ACCOUNT's realized losses for the UTC day / ISO week / calendar month
/// containing `now_ms`, as POSITIVE numbers in the account currency.
///
/// **Why this exists (2026-08-09).** `RiskyModeManager`'s loss accumulators are
/// per-manager, and there is one manager per ENGINE — but `POST
/// /autonomous/start` spawns an engine per portfolio and they all trade the
/// SAME cTrader account. A per-engine ledger silently turns the day cap into
/// `N × cap`. The accumulators also live in process memory, so a restart erased
/// the day's losses while the halt they produce is persisted.
///
/// The trade journal is the account-wide, durable record of exactly this, and
/// it is written for EVERY closed deal on the account — this engine's, the
/// other engines', and the operator's manual orders. Feeding it to
/// `raise_period_losses` (a monotonic max, never an assignment) closes both
/// gaps without double-counting.
pub(crate) fn account_period_losses(
    trades: &[crate::app_services::journal_store::ClosedTrade],
    account_id: Option<&str>,
    now_ms: i64,
) -> (f64, f64, f64) {
    let Some((day, week, month)) = period_starts_ms(now_ms) else {
        return (0.0, 0.0, 0.0);
    };
    let mut out = (0.0f64, 0.0f64, 0.0f64);
    for t in trades {
        // Scope to the account being traded. `None` on the row is legacy,
        // unattributable history and is never counted; `None` for the active
        // account (no broker account configured) cannot happen on a path that
        // just fetched a balance, but is treated as "count everything" rather
        // than "count nothing" — the fail-closed direction for a LOSS ledger.
        if let Some(active) = account_id
            && t.account_id.as_deref() != Some(active)
        {
            continue;
        }
        if !t.net_profit.is_finite() || t.net_profit >= 0.0 {
            continue;
        }
        let loss = -t.net_profit;
        let ts = t.effective_ts_ms();
        if ts >= day {
            out.0 += loss;
        }
        if ts >= week {
            out.1 += loss;
        }
        if ts >= month {
            out.2 += loss;
        }
    }
    out
}

/// Weekend kill-zone windows — EXACT replica of the backtest's session gate
/// (`eval.rs`, kill_zones_enabled): returns `(force_close, block_entry)` for a
/// bar timestamp. Force-close: Friday ≥ 20:00 UTC. Entries blocked: that same
/// window plus Monday 00:00–00:30 UTC. Same integer math as the kernel so the
/// two sides can never disagree on a boundary bar.
fn weekend_kill_zone(ts_ms: i64) -> (bool, bool) {
    if ts_ms <= 0 {
        return (false, false);
    }
    let sec_in_day = (ts_ms / 1000) % 86400;
    let hour = sec_in_day / 3600;
    let min = (sec_in_day % 3600) / 60;
    let days_since_epoch = ts_ms / 86_400_000;
    let weekday = (days_since_epoch + 4) % 7; // 0=Sun, 1=Mon, 5=Fri
    let friday_kill = weekday == 5 && hour >= 20;
    let monday_kill = weekday == 1 && hour == 0 && min < 30;
    (friday_kill, friday_kill || monday_kill)
}

/// Commit the locally tracked trail only after the broker confirms the exact
/// protection operation for this position. On every error the previous value
/// stays untouched, so the intended stop remains eligible for retry.
fn commit_broker_confirmed_trail(
    confirmed_stop_price: &mut f64,
    intended_stop_price: f64,
    position_id: i64,
    outcome: &CTraderExecutionOutcome,
) -> Result<()> {
    anyhow::ensure!(
        intended_stop_price.is_finite() && intended_stop_price > 0.0,
        "refusing to commit non-finite/non-positive trailing stop {intended_stop_price}"
    );
    anyhow::ensure!(
        outcome.status == CTraderExecutionStatus::Replaced,
        "broker returned {:?}, not ORDER_REPLACED, for position protection amend",
        outcome.status
    );
    anyhow::ensure!(
        outcome.position_id == Some(position_id),
        "broker confirmed protection for position {:?}, expected {position_id}",
        outcome.position_id
    );
    *confirmed_stop_price = intended_stop_price;
    Ok(())
}

/// A partial close changes the next close request's exact wire volume. Absence
/// from a snapshot alone does not settle the trade: the existing close-money
/// reconciliation must also verify all fills before clearing local ownership.
fn refresh_tracked_position_volume(
    tracked: &mut (i64, i64),
    expected_symbol_id: i64,
    broker_positions: impl IntoIterator<Item = (i64, i64, i64)>,
) -> Result<()> {
    let mut matching = broker_positions
        .into_iter()
        .filter(|(position_id, _, _)| *position_id == tracked.0);
    let Some((position_id, symbol_id, volume)) = matching.next() else {
        return Ok(());
    };
    anyhow::ensure!(
        matching.next().is_none() && symbol_id == expected_symbol_id && volume > 0,
        "invalid or ambiguous remaining broker volume for tracked position {position_id}"
    );
    tracked.1 = volume;
    Ok(())
}

pub(crate) fn bars_to_ohlcv(bars: &[crate::app_services::ctrader_data::HistoricalBar]) -> Ohlcv {
    Ohlcv {
        timestamp: Some(bars.iter().map(|b| b.timestamp_ms).collect()),
        open: bars.iter().map(|b| b.open).collect(),
        high: bars.iter().map(|b| b.high).collect(),
        low: bars.iter().map(|b| b.low).collect(),
        close: bars.iter().map(|b| b.close).collect(),
        volume: Some(bars.iter().map(|b| b.volume.unwrap_or(0) as f64).collect()),
    }
}

// Keep the existing age budget and 2.5x spread threshold. This cache check is
// entry-only and is not an execution permit or account-specific pip contract.
const LIVE_ENTRY_SPOT_MAX_AGE_MS: i64 = 120_000;

#[derive(Debug, PartialEq)]
enum LiveEntrySpreadRefusal {
    Quote(SpotQuoteRefusal),
    InvalidPipSize,
    InvalidExpectedSpread,
    InvalidPrices,
    InvalidSpread,
    ExceedsLimit { spread_pips: f64, limit_pips: f64 },
}

fn evaluate_live_entry_spread(
    quote: std::result::Result<(f64, f64), SpotQuoteRefusal>,
    pip_size: f64,
    expected_spread_pips: f64,
) -> std::result::Result<f64, LiveEntrySpreadRefusal> {
    let (bid, ask) = quote.map_err(LiveEntrySpreadRefusal::Quote)?;
    if !pip_size.is_finite() || pip_size <= 0.0 {
        return Err(LiveEntrySpreadRefusal::InvalidPipSize);
    }
    let limit_pips = expected_spread_pips * 2.5;
    if !expected_spread_pips.is_finite() || expected_spread_pips < 0.0 || !limit_pips.is_finite() {
        return Err(LiveEntrySpreadRefusal::InvalidExpectedSpread);
    }
    if !bid.is_finite() || !ask.is_finite() || bid <= 0.0 || ask <= 0.0 || bid > ask {
        return Err(LiveEntrySpreadRefusal::InvalidPrices);
    }
    let spread_pips = (ask - bid) / pip_size;
    if !spread_pips.is_finite() {
        return Err(LiveEntrySpreadRefusal::InvalidSpread);
    }
    if spread_pips > limit_pips {
        return Err(LiveEntrySpreadRefusal::ExceedsLimit {
            spread_pips,
            limit_pips,
        });
    }
    Ok(spread_pips)
}

fn require_live_entry_spread(
    account_id: i64,
    environment: CTraderEnvironment,
    session: SpotSessionId,
    symbol_id: i64,
    pip_size: f64,
    expected_spread_pips: f64,
    now_ms: i64,
) -> std::result::Result<f64, LiveEntrySpreadRefusal> {
    let quote = live_spots::get_fresh_tick(
        account_id,
        environment,
        session,
        symbol_id,
        now_ms,
        LIVE_ENTRY_SPOT_MAX_AGE_MS,
    )
    .map(|quote| (quote.bid, quote.ask));
    evaluate_live_entry_spread(quote, pip_size, expected_spread_pips)
}

// ── Risk-based position sizing ──────────────────────────────────────────────────

/// Position size (lots) for one entry, from the account's risk budget and the
/// strategy's OWN stop distance: `lots = balance × risk% / (sl_pips ×
/// pip_value_per_lot_in_account)`, rounded DOWN after all upper bounds apply.
/// A budget below the broker's minimum is an error, never permission to round
/// up. Missing financial inputs also refuse the entry in every trading mode.
/// This bounds the price loss at the requested stop, not gap/slippage or fees.
#[allow(clippy::too_many_arguments)]
fn risk_based_lots(
    balance: f64,
    risk_fraction: f64,
    sl_pips: f64,
    meta: Option<&neoethos_core::symbol_metadata::SymbolMetadata>,
    account_ccy: &str,
    fx_quote_to_account: Option<f64>,
    live_price: Option<f64>,
    max_lot_cap: f64,
) -> Result<f64> {
    anyhow::ensure!(
        balance.is_finite()
            && balance > 0.0
            && risk_fraction.is_finite()
            && risk_fraction > 0.0
            && risk_fraction <= 1.0,
        "invalid account balance or risk fraction for position sizing"
    );
    let meta = meta.context("missing broker symbol metadata for position sizing")?;
    for (name, value) in [
        ("stop distance", sl_pips),
        ("lot step", meta.lot_step),
        ("minimum lot", meta.min_lot),
        ("maximum lot", meta.max_lot),
        ("operator lot cap", max_lot_cap),
        ("pip size", meta.pip_size),
    ] {
        anyhow::ensure!(value.is_finite() && value > 0.0, "invalid {name}: {value}");
    }
    let price = live_price
        .filter(|p| p.is_finite() && *p > 0.0)
        .context("missing current price for position sizing")?;
    let pip_val = meta.pip_value_in_account(account_ccy, fx_quote_to_account, live_price);
    let risk_budget = balance * risk_fraction;
    let raw = meta
        .risk_money_to_lots(
            risk_budget,
            sl_pips,
            account_ccy,
            fx_quote_to_account,
            Some(price),
        )
        .context("cannot price stop risk in the account currency; check the FX conversion")?;

    // Preserve the existing 30x notional ceiling; it is not a broker-margin
    // estimate. Use the SAME account conversion as pip risk, including when
    // the account currency is the pair's base. Never assume a missing FX = 1.
    const MAX_NOTIONAL_MULTIPLE: f64 = 30.0;
    let notional_per_lot = (pip_val / meta.pip_size) * price;
    anyhow::ensure!(
        notional_per_lot.is_finite() && notional_per_lot > 0.0,
        "cannot value position notional in the account currency"
    );
    let notional_cap = (balance / notional_per_lot) * MAX_NOTIONAL_MULTIPLE;
    let capped = raw.min(meta.max_lot).min(max_lot_cap).min(notional_cap);
    let lots = (capped / meta.lot_step).floor() * meta.lot_step;
    anyhow::ensure!(
        lots.is_finite() && lots >= meta.min_lot,
        "budget permits {lots:.8} lots, below broker minimum {}; entry skipped, not rounded up",
        meta.min_lot
    );
    anyhow::ensure!(
        lots * sl_pips * pip_val <= risk_budget * (1.0 + 1e-12),
        "rounded lot size exceeds the requested stop-risk budget"
    );
    Ok(lots)
}

/// The operator's `models.blend_gate_floor`.
///
/// **WIRED 2026-08-10 (audit #232).** The recipient field now exists
/// (`neoethos-core/src/config.rs`, `ModelsConfig::blend_gate_floor`, default
/// 0.34), so this stops returning `None` and returns HIS number. The value goes
/// through [`neoethos_trader::BlendConfig::from_config_values`], which REFUSES
/// a non-finite value, one outside `[0,1]`, or an inverted pair back to the
/// shipped defaults and logs both numbers — so there is no path by which a bad
/// YAML value silently changes a live position size.
///
/// The shipped default is numerically identical to the old hardcoded
/// `DEFAULT_BLEND_GATE_FLOOR`, so an operator who sets nothing sees no change:
/// what moved is that the literal in the live sizing path became a knob he can
/// actually reach.
fn operator_blend_gate_floor(settings: Option<&neoethos_core::Settings>) -> Option<f64> {
    // Written as a `match` with an `ident: Type` binding rather than
    // `.map(|s| ...)` on purpose: `config_has_recipient`'s scanner resolves a
    // field access by its RECEIVER's type, and a closure parameter resolves to
    // nothing — so the `.map` spelling would leave this knob looking like an
    // orphan in the very ledger that exists to find orphans.
    match settings {
        Some(s) => {
            let s: &neoethos_core::Settings = s;
            Some(s.models.blend_gate_floor)
        }
        None => None,
    }
}

/// The operator's `models.blend_veto_below`.
/// See [`operator_blend_gate_floor`] — same wiring, same validating
/// constructor, same shipped default (`DEFAULT_BLEND_VETO_BELOW`, 0.15).
fn operator_blend_veto_below(settings: Option<&neoethos_core::Settings>) -> Option<f64> {
    match settings {
        Some(s) => {
            let s: &neoethos_core::Settings = s;
            Some(s.models.blend_veto_below)
        }
        None => None,
    }
}

fn budgeted_role_decision_for_last_row(
    ensemble: &neoethos_models::ensemble_inference::SoftVotingEnsemble,
    dataset: &neoethos_data::SymbolDataset,
) -> Result<neoethos_models::ensemble_inference::EnsembleDecision> {
    let installed = neoethos_core::execution_budget::installed_process_budget()
        .context("live ensemble inference requires the immutable process CPU budget")?;
    let width = installed.resolved().effective_worker_limit;
    let lease = installed
        .broker()
        .try_acquire(neoethos_core::execution_budget::CpuPermitRequest::local(
            width,
        ))
        .context("query live ensemble CPU admission")?
        .context("process CPU budget is busy; live ensemble abstains on this bar")?;
    // The Search cube has already been released. Model features use only the
    // model-owned persisted recipe/fit, under this same admitted CPU lease.
    let features = Arc::new(lease.scope(|| ensemble.prepare_model_features(dataset))?);
    ensemble.bind_model_features(&features)?.last_row(
        features.n_samples(),
        neoethos_models::ensemble_inference::bootstrap::LIVE_DECISION_TAIL_ROWS,
        &lease,
    )
}

/// A requested model decision may shrink or veto a NEW entry, never become
/// genes-only sizing because inference failed. Called after existing-position
/// reconciliation/protection and before reserving an entry or sending an order.
fn checked_live_ml_entry(
    direction: Direction,
    decision: Result<neoethos_models::ensemble_inference::EnsembleDecision>,
    blend: &neoethos_trader::BlendConfig,
) -> Result<(neoethos_models::ensemble_inference::EnsembleDecision, f64)> {
    let decision = decision.context("required live model inference is unavailable")?;
    anyhow::ensure!(
        decision.validity.is_valid(),
        "required live model row is ineligible: {:?}",
        decision.validity
    );
    let ml = neoethos_trader::MlDecision {
        dir_probs: decision.dir_probs,
        regime_gate: decision.regime_gate,
        anomaly_scale: decision.anomaly_scale,
    };
    let (blended_direction, multiplier) = neoethos_trader::blend_decision(direction, &ml, blend);
    anyhow::ensure!(
        direction != Direction::Flat
            && blended_direction == direction
            && multiplier.is_finite()
            && multiplier > 0.0
            && multiplier <= 1.0,
        "live ML entry veto: no valid positive size multiplier"
    );
    Ok((decision, multiplier))
}

/// Preserve the mode's risk budget; ML can scale it once, while the confidence
/// gate receives the independent value measured by the archived gene policy.
fn live_entry_sizing_inputs(
    mode_risk: f64,
    gene_confidence: f64,
    ml_multiplier: Option<f64>,
) -> Result<(f64, Option<f64>)> {
    let scale = ml_multiplier.unwrap_or(1.0);
    anyhow::ensure!(
        mode_risk.is_finite()
            && mode_risk >= 0.0
            && gene_confidence.is_finite()
            && (0.0..=1.0).contains(&gene_confidence)
            && scale.is_finite()
            && (0.0..=1.0).contains(&scale),
        "live risk, gene confidence or ML multiplier is invalid"
    );
    Ok((mode_risk * scale, Some(gene_confidence)))
}

/// Refresh the one durable account-risk authority from a broker snapshot. If a
/// firm-local day has just rolled, the helper first proves that the snapshot's
/// balance is still the reset-boundary balance; it never manufactures an
/// anchor from a mid-day value.
async fn prepare_account_risk_period(
    authority: &SharedAccountRiskAuthority,
    identity: &AccountRiskIdentity,
    period: PropFirmPeriod,
    snapshot: AccountRiskSnapshot,
) -> Result<AccountRiskSummary> {
    let needs_anchor = authority
        .lock()
        .map_err(|_| anyhow!("account-risk authority lock is poisoned"))?
        .needs_period(period);
    let anchor_evidence = if needs_anchor {
        let evidence_identity = identity.clone();
        let to_utc_ms = chrono::Utc::now().timestamp_millis();
        Some(
            tokio::task::spawn_blocking(move || {
                fetch_anchor_evidence_blocking(&evidence_identity, period, to_utc_ms)
            })
            .await
            .map_err(|error| anyhow!("daily-anchor evidence task failed: {error}"))??,
        )
    } else {
        None
    };

    let mut guard = authority
        .lock()
        .map_err(|_| anyhow!("account-risk authority lock is poisoned"))?;
    guard
        .prepare_period(period, snapshot, anchor_evidence.as_ref())
        .context("prepare durable account-wide prop-firm risk period")?;
    Ok(guard.summary())
}

// ── Main loop ─────────────────────────────────────────────────────────────────

async fn run(
    req: StartRequest,
    stop: Arc<AtomicBool>,
    status: Arc<std::sync::Mutex<LiveTradingStatus>>,
    // The broker environment (`true` = Live/real money) that `start`'s demo
    // forward-test gate was evaluated against. Re-checked every bar; a change
    // stops the engine (see the check inside the loop).
    gated_env_is_live: bool,
    account_risk_registry: Arc<AccountRiskRegistry>,
) -> Result<()> {
    // Load portfolio artifact (same as replay_portfolio_from_dir)
    let artifact = neoethos_search::load_live_portfolio_json(&req.portfolio_path)
        .with_context(|| format!("load live portfolio {}", req.portfolio_path))?;

    if artifact.genes.is_empty() {
        anyhow::bail!("portfolio '{}' has no genes", req.portfolio_path);
    }
    let portfolio_oos_half_kelly = artifact
        .portfolio_half_kelly_risk_fraction()
        .context("resolve held-out half-Kelly sizing from live portfolio")?;

    // A valid receipt is necessary but not sufficient for real money: the
    // connected broker session must be the same environment/account/symbol id
    // that produced discovery's direct generations.
    let broker_symbols = fetch_broker_symbols_blocking()
        .context("resolve active cTrader identity for strict v3 portfolio")?;
    let broker_environment = match broker_symbols.environment {
        value if value.eq_ignore_ascii_case("demo") => neoethos_data::CTraderEnvironment::Demo,
        value if value.eq_ignore_ascii_case("live") => neoethos_data::CTraderEnvironment::Live,
        value => anyhow::bail!("active cTrader environment `{value}` is not canonical"),
    };
    let matching_symbols = broker_symbols
        .symbols
        .iter()
        .filter(|candidate| candidate.symbol_name == artifact.symbol)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        matching_symbols.len() == 1,
        "active cTrader account exposes {} exact `{}` symbols; expected one",
        matching_symbols.len(),
        artifact.symbol
    );
    let gated_account_id = broker_symbols.account_id;
    let gated_symbol_id = matching_symbols[0].symbol_id;
    artifact.validate_ctrader_runtime_binding(
        broker_environment,
        gated_account_id,
        gated_symbol_id,
        &matching_symbols[0].symbol_name,
    )?;

    let symbol = artifact.symbol.clone();
    let base_tf = artifact.base_tf.clone();
    let higher_tfs = artifact.higher_tfs.clone();
    let genes = artifact.genes.clone();
    let live_trading_policy = artifact.live_trading_policy.clone();

    if let Ok(mut s) = status.lock() {
        s.symbol = Some(symbol.clone());
        s.base_tf = Some(base_tf.clone());
        s.genes = genes.len();
        s.protection_policy_identity = Some(live_trading_policy.identity_hash.clone());
        s.trailing_enabled = Some(live_trading_policy.trailing_enabled);
        s.clear_position_telemetry(live_trading_policy.trailing_enabled);
    }

    let bar_ms = tf_duration_ms(&base_tf)?;
    let warmup = req.warmup_bars;
    let mut last_bar_ts: i64 = 0;
    // Track open position: (position_id, broker_volume_in_units)
    let mut open_position: Option<(i64, i64)> = None;
    let mut bars_evaluated: u64 = 0;

    // ── Auto-cull: retire the strategy after N consecutive losing trades ───────
    // Realized results are read from the broker's closing deals for positions
    // THIS engine opened (catches SL/TP exits too, not just engine flips).
    let cull_threshold = req.cull_after_consecutive_losses;
    // Rolling-window win-rate criterion (operator 2026-07-02): a chronic 40%-WR
    // strategy alternating wins/losses never streaks to the consecutive limit
    // but still bleeds the account — the window floor catches it.
    let cull_min_wr = req.cull_min_win_rate_pct.clamp(0.0, 100.0);
    let cull_window = req.cull_window_trades.clamp(4, 100);
    let portfolio_path = req.portfolio_path.clone();
    let mut opened_ids: HashSet<i64> = HashSet::new();
    let mut has_unresolved_broker_entry = false;
    // Retain the original intent correlation ID while this process is blocked.
    // A durable restart/recovery registry is not implied by this local state.
    let mut unresolved_entry_client_order_id: Option<String> = None;
    let mut opened_entry_filled_volumes: HashMap<i64, i64> = HashMap::new();
    let mut opened_volume_scales: HashMap<i64, BrokerSymbolVolumeScaleEvidenceV1> = HashMap::new();
    let mut close_money_accumulators: HashMap<i64, BrokerPositionMoneyAccumulatorV1> =
        HashMap::new();
    let mut unverified_close_positions: HashSet<i64> = HashSet::new();
    let mut consecutive_losses: u32 = 0;
    let mut pending_retirement_reason: Option<String> = None;
    let mut net_pnl_running: f64 = 0.0;
    // Live-learning foundation (operator 2026-07-02): remember the EXACT
    // feature row each entry acted on; pair it with the realized outcome at
    // close and append to the experience store. Pure data collection — the
    // online/RL experts train OFFLINE from this (never silently live).
    let mut pending_experience: HashMap<
        i64,
        crate::app_services::experience_store::LiveExperience,
    > = HashMap::new();
    // Rolling outcome window: true = win (net > 0). BE counts as a loss —
    // a break-even trade doesn't pay for its costs' risk.
    let mut recent_results: std::collections::VecDeque<bool> =
        std::collections::VecDeque::with_capacity(cull_window + 1);

    // ── Trailing-stop parity state (per open position).
    //
    // CORRECTED 2026-08-09 (#208). This used to say "discovery hardcodes
    // break-even + trailing ALWAYS ON ... live MUST replicate it". Discovery
    // hardcodes nothing any more: the geometry comes from
    // `models.exit_policy` and the shipped default is OFF. The parity mandate
    // is unchanged in principle and inverted in fact — live must replicate
    // WHATEVER the policy says, which today means not trailing at all. These
    // five variables are seeded on every entry regardless, so flipping the
    // policy on mid-run needs no restart to have correct state.
    let mut pos_entry_px: f64 = 0.0;
    let mut pos_sl_pips: f64 = 0.0;
    let mut pos_is_long: bool = false;
    let mut pos_extreme: f64 = 0.0;
    let mut pos_trail_px: f64 = 0.0;

    tracing::info!(
        target: "neoethos_app::live_trading",
        %symbol, %base_tf,
        genes = genes.len(),
        higher_tfs = ?higher_tfs,
        "live trading loop started"
    );

    // ── Risk-based position sizing context (resolved once at start) ────────────
    // Size each entry by % of the LIVE account balance in the broker's REAL
    // deposit currency — not a fixed lot. Any piece we can't resolve makes that
    // entry fall back to req.lot_size (never a wrong size).
    let live_config_path = crate::server::state::current_config_path();
    let sizing = Some(
        neoethos_core::Settings::from_yaml(&live_config_path).with_context(|| {
            format!(
                "load live trading settings from {}; refusing to start without one coherent risk/data policy",
                live_config_path.display()
            )
        })?,
    );
    let risk_fraction = sizing
        .as_ref()
        .map(|s| s.risk.risk_per_trade)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    // Risky Mode sizing comes from this artifact's untouched OOS edge through
    // the same bounded half-Kelly formula used by Search. Bankroll changes the
    // dollar amount at risk, not the evidence-derived fraction. PropFirm mode
    // keeps its separately configured sizing path.
    let trading_mode_risky = sizing
        .as_ref()
        .map(|s| s.system.trading_mode.eq_ignore_ascii_case("risky"))
        .unwrap_or(false);
    let risky_start_balance = sizing
        .as_ref()
        .map(|s| s.system.risky_start_balance_usd)
        .unwrap_or(0.0);
    let risky_target_balance = sizing
        .as_ref()
        .map(|s| s.system.risky_target_balance_usd)
        .unwrap_or(0.0);
    // LIVE ML gate (models.live_ml_gate, default OFF): the 32-voter soft
    // ensemble scales per-trade risk by agreement × regime × anomaly. Genes
    // ALWAYS pick the direction (Stage-3 invariant); ML only shrinks or, on
    // a hard regime/anomaly collapse, skips the bar.
    let live_ml_gate = sizing
        .as_ref()
        .map(|s| s.models.live_ml_gate)
        .unwrap_or(false);
    // The two multipliers that gate SCALES EVERY ENTRY'S SIZE by. Built ONCE
    // here, through `BlendConfig::from_config_values` — the single validating
    // constructor — instead of `..Default::default()` at the per-entry call
    // site. Two reasons for building it here and not in the loop: a refusal is
    // then logged ONCE at engine start (a per-bar warn is a warn nobody reads),
    // and the operator sees the effective numbers before the first order.
    let live_blend_cfg = neoethos_trader::BlendConfig::from_config_values(
        neoethos_trader::BlendMode::MlScale,
        operator_blend_gate_floor(sizing.as_ref()),
        operator_blend_veto_below(sizing.as_ref()),
    );
    if live_ml_gate {
        tracing::info!(
            target: "neoethos_app::live_trading",
            %symbol,
            gate_floor = live_blend_cfg.gate_floor,
            veto_below = live_blend_cfg.veto_below,
            mode = ?live_blend_cfg.mode,
            "LIVE ML blend multipliers resolved — these two numbers scale every \
             entry's risk; any refused operator value was logged above with both \
             the configured and the used number"
        );
    }
    // Journal location + the account this engine trades — the inputs to the
    // ACCOUNT-WIDE risky-mode loss ledger consulted at the pre-send check.
    // Resolved once: `data_dir` does not move at runtime, and the account id is
    // the one `broker_api::resolve_creds` routes to (the engine already stops
    // itself if the environment, and therefore the account, changes).
    let journal_data_dir: Option<std::path::PathBuf> =
        sizing.as_ref().map(|s| s.system.data_dir.clone());
    let journal_account: Option<String> = crate::app_services::journal_store::active_account_id();
    let max_lot_cap = sizing
        .as_ref()
        .map(|s| s.risk.max_lot_size)
        .filter(|v| *v > 0.0)
        .unwrap_or(f64::INFINITY);
    // Portfolio-level concurrent-risk cap: each entry budgets against
    // `cap − open_positions × risk_per_trade` using the broker's LIVE position
    // count, so many engines can't stack unbounded concurrent risk.
    //
    // THE 0.0 SENTINEL IS GONE (2026-08-10, audit #211). This used to read
    // "0 = disabled", which meant that on a knob named `max_` the loosest
    // possible setting and the never-touched field were spelled the same way —
    // and `RiskConfig::default()` shipped 0.0, so every install ran with no
    // portfolio ceiling and nothing said so. `Settings`' preset seal now
    // re-seeds any non-positive value from the preset + trading mode
    // (`config.rs:3897`), so a cap of 0 cannot arrive here through a loaded
    // config at all. If one ever does — a hand-built `Settings`, a future
    // caller that skips the seal — it is read LITERALLY: at most 0.0 of the
    // account may be at risk concurrently, i.e. no entries. Loud and tight, not
    // silently unlimited. The way to run without a ceiling is 1.0.
    let portfolio_risk_cap = sizing
        .as_ref()
        .map(|s| s.risk.max_portfolio_risk)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    if portfolio_risk_cap <= 0.0 {
        tracing::error!(
            target: "neoethos_app::live_trading",
            max_portfolio_risk = portfolio_risk_cap,
            "risk.max_portfolio_risk is not positive — the config seal should have \
             re-seeded it. Reading it literally: NO concurrent risk is permitted, so \
             every entry will be refused. Set it to 1.0 if you mean 'no ceiling'."
        );
    }
    // Weekend kill zones — force-close before the weekend, block Fri-late /
    // Mon-open entries.
    //
    // PARITY, AND IT IS REAL SINCE 2026-08-10 (audit #75/#217).
    //
    // History, so nobody re-opens this: from 2026-08-04 to 2026-08-10 the flag
    // was ONE-SIDED. `discovery_backtest_settings` hardcoded
    // `kill_zones_enabled: true`, so every backtest that ever validated a
    // strategy ran WITH kill zones unconditionally, while only this live path
    // consulted `risk.kill_zones_enabled`. Setting it to `false` could
    // therefore only move live AWAY from what was validated — holding through
    // weekend gaps no backtest in the artifact history had ever held through —
    // and never toward it. The defaults agree (both `true`), which is why it
    // went unseen.
    //
    // Both sides now read the value sealed in the portfolio's immutable search
    // authority. Editing Settings after discovery cannot silently change the
    // weekend or spread policy of a strategy that was already validated.
    let kill_zones_enabled = live_trading_policy.kill_zones_enabled;
    // ── Exit geometry — THE EXACT VALUES THE SEARCH PRICED (#208/#74) ─────────
    //
    // 2026-08-09. `models.exit_policy` (`neoethos-core/src/config.rs:1642`) is
    // the single recipient for the break-even/trailing geometry. Discovery reads
    // it through `EvaluationConfig::for_symbol` (`strategy_gene.rs:867`) and its
    // default flipped to `trailing_enabled: false` this morning. This loop did
    // NOT flip: it ran the trail unconditionally off
    // `DEFAULT_TRAILING_MIN_LOCK_PIPS` with the +1R trigger and the 1×SL trail
    // distance written inline. That is a backtest/live divergence in the exact
    // mechanism measured as capping realised payoff at 1.08 against a floor of
    // 2.0 — every strategy validated from today was scored with the take-profit
    // reachable and then traded with the stop pulled to break-even at +1R.
    //
    // The v4 live artifact cannot load without this policy and its identity
    // hash. Keep the explicit Option type because the config-recipient audit
    // follows the wrapped `ExitPolicyConfig`; unlike the old path, `Some` here
    // is guaranteed by artifact validation rather than today's Settings file.
    let exit_policy: Option<neoethos_core::config::ExitPolicyConfig> =
        Some(live_trading_policy.exit_policy());
    match exit_policy {
        Some(p) if p.trailing_enabled => tracing::warn!(
            target: "neoethos_app::live_trading",
            %symbol,
            policy_identity = %live_trading_policy.identity_hash,
            trailing_enabled = true,
            be_trigger_r = p.trailing_be_trigger_r,
            stop_multiplier = p.trailing_stop_multiplier,
            min_lock_pips = p.trailing_min_lock_pips,
            "LIVE TRAILING ARMED from the portfolio's sealed discovery policy"
        ),
        Some(_) => tracing::info!(
            target: "neoethos_app::live_trading",
            %symbol,
            policy_identity = %live_trading_policy.identity_hash,
            trailing_enabled = false,
            "live trailing DISABLED by the portfolio's sealed discovery policy"
        ),
        None => unreachable!("validated live portfolio always carries an exit policy"),
    }
    // Risky-mode operator input is a ceiling, never a substitute for measured
    // edge. A malformed ceiling is ignored loudly; schema-v5 OOS half-Kelly and
    // the Search-v5 25 % hard cap still bind.
    let risky_configured_ceiling: Option<f64> = match sizing
        .as_ref()
        .and_then(|s| s.risk.risky_max_risk_per_trade)
    {
        Some(v) if v.is_finite() && v > 0.0 => Some(v),
        Some(v) => {
            tracing::error!(
                target: "neoethos_app::live_trading",
                %symbol, configured = v,
                "risk.risky_max_risk_per_trade is set to a value that cannot \
                 bound anything (non-finite or <= 0) — IGNORING IT. The \
                 portfolio's held-out half-Kelly remains authoritative. Set a \
                 positive fraction to add a lower operator ceiling"
            );
            None
        }
        None => None,
    };
    let risky_effective_fraction = if trading_mode_risky {
        let effective = risky_configured_ceiling
            .map(|ceiling| portfolio_oos_half_kelly.min(ceiling))
            .unwrap_or(portfolio_oos_half_kelly);
        anyhow::ensure!(
            effective.is_finite() && effective > 0.0,
            "Risky Mode portfolio has no positive held-out sizing edge"
        );
        tracing::warn!(
            target: "neoethos_app::live_trading",
            %symbol,
            oos_half_kelly = portfolio_oos_half_kelly,
            configured_ceiling = ?risky_configured_ceiling,
            effective_risk_fraction = effective,
            "RISKY SIZING ARMED FROM HELD-OUT EDGE — fixed 30-50% ladder is not used"
        );
        Some(effective)
    } else {
        None
    };
    // Match the selected portfolio to its own installed candidate and captured
    // inference policy. This does not supply financial/deployment permission.
    let live_ensemble: Option<
        std::sync::Arc<neoethos_models::ensemble_inference::soft_voting::SoftVotingEnsemble>,
    > = if live_ml_gate {
        let model_settings = sizing
            .as_ref()
            .context("required live ML has no captured settings")?
            .clone();
        let portfolio_identity =
            neoethos_search::canonical_locked_portfolio_identity_sha256_v1(&artifact)?;
        let loaded = tokio::task::spawn_blocking(move || {
            candidate_models::load(
                &model_settings,
                &std::path::Path::new("models").join("candidates"),
                &portfolio_identity,
            )
        })
        .await
        .context("required live ensemble loader task failed; refusing to start")?
        .context("models.live_ml_gate is ON but its ensemble cannot load; refusing to start")?;
        let ensemble = loaded.ensemble;
        let outcome =
            neoethos_models::ensemble_inference::EnsemblePredictor::load_outcome(&ensemble);
        // #166. `loaded` is NOT the number of voters: an expert whose
        // output kind is not Classification3, or one on the operator's
        // exclusion list, is held in the outcome and never votes. Log
        // both, and name the non-voters — "31 loaded" next to "2 voting"
        // is the difference between a working ensemble and a banner.
        let unused = {
            let mut v = ensemble.experts_unused_for_voting();
            v.sort_unstable();
            v.join(",")
        };
        tracing::info!(
            target: "neoethos_app::live_trading",
            %symbol, %base_tf,
            loaded = outcome.loaded_count(),
            training_handoff = %loaded.training_handoff,
            candidate_tree = %loaded.candidate_tree,
            missing = outcome.missing_count(),
            degraded = outcome.degraded_count(),
            voting = ensemble.voting_expert_count(),
            unused_for_voting = %unused,
            "LIVE ML gate armed — ensemble voters loaded (genes still pick direction; ML only scales size)"
        );
        Some(std::sync::Arc::new(ensemble))
    } else {
        // Explicit genes-only mode does not load or consult trained models.
        // Report the resolved policy, not stale claims about old artifact
        // losses, historical defaults, or which models another job trained.
        tracing::warn!(
            target: "neoethos_app::live_trading",
            %symbol, %base_tf,
            "models.live_ml_gate is OFF — explicit genes-only execution; trained models are not used for entry decisions"
        );
        None
    };
    let entry_environment = if gated_env_is_live {
        CTraderEnvironment::Live
    } else {
        CTraderEnvironment::Demo
    };
    let startup_symbol = symbol.clone();
    let startup_entry_context: crate::app_services::broker_api::LiveEntryContext =
        tokio::task::spawn_blocking(move || {
            fetch_live_entry_context_blocking(
                &startup_symbol,
                entry_environment,
                gated_account_id,
                gated_symbol_id,
            )
        })
        .await
        .context("startup broker entry-context task failed")??;
    let startup_symbol_contract = startup_entry_context.contract().clone();
    let sym_meta = Some(startup_entry_context.metadata().clone());
    let exact_pip_size = startup_entry_context.metadata().pip_size;
    anyhow::ensure!(
        live_trading_policy.sealed_evaluation_config()?.pip_value == exact_pip_size,
        "archived strategy pip size differs from the current account-bound broker symbol"
    );
    let account_balance = startup_entry_context.margin().balance;
    let account_ccy = startup_entry_context.account_currency().to_owned();
    let trader_money_digits = startup_entry_context.trader_money_digits();
    tracing::info!(
        target: "neoethos_app::live_trading",
        %symbol, balance = account_balance, account_ccy = %account_ccy,
        trader_money_digits, risk_fraction,
        "account-bound broker sizing contract resolved; price and FX are read afresh per entry"
    );
    drop(startup_entry_context);
    let resolved_settings = sizing
        .as_ref()
        .context("live settings disappeared after successful startup resolution")?;
    let expected_environment = if gated_env_is_live { "live" } else { "demo" };
    let account_identity =
        AccountRiskIdentity::new(expected_environment, gated_account_id, &account_ccy)?;
    let account_entry_authority = account_risk_registry
        .acquire_entry(account_identity.clone(), &resolved_settings.system.data_dir)?;
    {
        let entry_state = account_entry_authority
            .lock()
            .map_err(|_| anyhow!("account-entry authority lock is poisoned"))?;
        if let Some(client_order_id) = entry_state.unresolved_client_order_id() {
            has_unresolved_broker_entry = true;
            unresolved_entry_client_order_id = Some(client_order_id.to_owned());
            tracing::error!(target: "neoethos_app::live_trading",
                client_order_id, account_id = gated_account_id,
                "restored unresolved entry; new entries blocked until exact broker reconciliation, no automatic retry");
        }
    }

    // ── Risky Mode kill switch (W3, 2026-08-09) ───────────────────────────────
    // `RiskyModeManager` implements the kill-switch tiers, a
    // pre-send sanity ceiling and daily/weekly/monthly loss accumulators. Until
    // now its ONLY construction in the workspace was inside the scenarios API.
    // This live manager carries the artifact-derived half-Kelly fraction and
    // enforces the same kill switches against the actual broker bankroll.
    //
    // Scope: the manager exists ONLY when `system.trading_mode == "risky"`. The
    // default `prop_firm` path constructs nothing and its behaviour is
    // byte-identical to before this change.
    //
    // Currency note, stated rather than hidden: the manager's fields are named
    // `*_usd`, but the bankroll fed to it is the broker's balance in the
    // account's REAL deposit currency (GBP for this operator), and
    // `system.risky_start_balance_usd` / `risky_target_balance_usd` are read in
    // that same currency. This is the identical convention the pre-existing
    // ladder call at the entry site already used — one consistent unit, not two.
    let mut risky_manager: Option<neoethos_core::domain::risky_mode::RiskyModeManager> = None;
    if trading_mode_risky {
        use neoethos_core::domain::risky_mode as rm;
        let bankroll = if account_balance.is_finite() && account_balance > 0.0 {
            account_balance
        } else {
            risky_start_balance
        };
        let cfg = rm::RiskyModeConfig {
            starting_capital_usd: risky_start_balance,
            target_capital_usd: risky_target_balance,
            stage_doubling_factor: rm::DEFAULT_DOUBLING_FACTOR,
            stages: rm::build_logarithmic_stages_at_risk(
                risky_start_balance,
                risky_target_balance,
                rm::DEFAULT_DOUBLING_FACTOR,
                risky_effective_fraction
                    .context("Risky Mode is missing its OOS sizing fraction")?,
            ),
            ..rm::RiskyModeConfig::default()
        };
        // FAIL CLOSED. If the evidence-sized ladder cannot be validated, no
        // Risky order may run without its kill switches.
        let manager = rm::RiskyModeManager::new(cfg, bankroll).with_context(|| {
            format!(
                "Risky Mode is ON (system.trading_mode = \"risky\") but its kill switch could \
                 not be built from system.risky_start_balance_usd = {risky_start_balance} / \
                 system.risky_target_balance_usd = {risky_target_balance} with a live balance \
                 of {bankroll}. REFUSING TO START: Risky Mode will not run without its \
                 held-out sizing evidence, daily/weekly/stage/monthly loss caps, and its \
                 pre-send ceiling. Fix those two settings, or set \
                 system.trading_mode = \"prop_firm\""
            )
        })?;
        let stage = manager.current_stage();
        tracing::warn!(
            target: "neoethos_app::live_trading",
            %symbol,
            bankroll,
            stage_idx = stage.stage_idx,
            stage_risk_per_trade = stage.risk_per_trade_fraction,
            stage_daily_loss_cap = stage.daily_loss_cap_fraction,
            stage_weekly_drawdown_cap = stage.weekly_drawdown_cap_fraction,
            presend_ceiling = manager.config().presend_sanity_ceiling_fraction,
            monthly_loss_cap = manager.config().monthly_loss_cap_fraction,
            high_water_stage_idx = manager.high_water_stage_idx(),
             "RISKY MODE KILL SWITCH ARMED — every entry is checked before the \
              order is sent. Tiers that can actually fire: PreSendSanity (this \
              order's risk >= 55% of bankroll), PerDay (the ACCOUNT's realized \
              loss this UTC day reached the stage cap), PerWeek (the ACCOUNT's \
              realized loss this ISO week reached the stage cap), PerStage (the bankroll \
             retreated below the rung under the highest stage reached), \
             PerMonth (inert at the shipped 0.99 cap — the day cap binds \
             first). Broker margin-call/account-disconnect events start the \
             same durable 24 h cooldown through app_services::margin_call."
        );
        risky_manager = Some(manager);
    }
    // Accumulator period cursors for the manager's daily / weekly / monthly
    // ledgers. Nothing ever reset them because nothing ever fed them; they are
    // rolled here, at entry time, from the same UTC clock the daily entry cap
    // and the drawdown breakers use.
    //
    // SEEDED, not `None` (2026-08-09). With `None` the first entry attempt saw
    // "every period changed" and wiped all three accumulators. That was benign
    // only by accident — nothing could have accumulated before the first entry
    // — and it stops being benign the moment the ledger is seeded from the
    // journal at the entry site, which is exactly what now happens: an
    // account-wide loss already booked today would have been erased by the
    // first entry attempt after every restart.
    let mut risky_period: Option<(u32, u32, u32)> = if trading_mode_risky {
        use chrono::Datelike;
        let d = chrono::Utc::now().date_naive();
        let iso = d.iso_week();
        Some((
            (d.year().max(0) as u32) * 10_000 + d.month() * 100 + d.day(),
            (iso.year().max(0) as u32) * 100 + iso.week(),
            (d.year().max(0) as u32) * 100 + d.month(),
        ))
    } else {
        None
    };

    // ── Account-wide prop-firm risk authority ────────────────────────────────
    // Exactly one authority exists per broker account and is shared by every
    // portfolio engine through AppApiState. Its checkpoint is durable; a
    // second engine or app restart cannot reset the day anchor, peak,
    // circuit-breaker latch, targets, or revenge window. The entry count is
    // owned separately by `account_entry_authority` for both live modes.
    let mut prop_firm_authority: Option<SharedAccountRiskAuthority> = None;
    let mut prop_firm_identity: Option<AccountRiskIdentity> = None;
    let mut active_prop_firm_period: Option<PropFirmPeriod> = None;
    if !trading_mode_risky {
        let settings = resolved_settings;
        let initial_status = tokio::task::spawn_blocking(
            crate::app_services::broker_api::fetch_margin_status_blocking,
        )
        .await
        .map_err(|error| anyhow::anyhow!("initial account-equity task failed: {error}"))??;
        anyhow::ensure!(
            initial_status
                .environment_label
                .eq_ignore_ascii_case(expected_environment)
                && initial_status.account_id == gated_account_id,
            "initial prop-firm equity snapshot belongs to {}/account {}, expected {}/account {}",
            initial_status.environment_label,
            initial_status.account_id,
            expected_environment,
            gated_account_id
        );
        let identity = account_identity.clone();
        let initial_snapshot = AccountRiskSnapshot {
            balance: initial_status.balance,
            equity: initial_status.equity,
        };
        let authority = account_risk_registry.acquire(
            identity.clone(),
            settings,
            initial_snapshot,
            &settings.system.data_dir,
        )?;
        let initial_period = prop_firm_period(settings.risk.preset, chrono::Utc::now())?;
        let summary =
            prepare_account_risk_period(&authority, &identity, initial_period, initial_snapshot)
                .await?;
        tracing::warn!(
            target: "neoethos_app::live_trading",
            %symbol,
            preset = summary.preset.as_str(),
            challenge_phase = %summary.challenge_phase,
            challenge_mode = summary.challenge_mode,
            recovery_mode_enabled = summary.recovery_mode_enabled,
            reset_zone = initial_period.reset_zone,
            balance = initial_status.balance,
            equity = initial_status.equity,
            day_start_balance = summary.day_start_balance,
            daily_drawdown_limit = summary.daily_drawdown_limit,
            total_drawdown_limit = summary.total_drawdown_limit,
            max_risk_per_trade = summary.max_risk_per_trade,
            phase_advisory_max_risk_per_trade = summary.phase_advisory_max_risk_per_trade,
            min_confidence_threshold = summary.min_confidence_threshold,
            "ACCOUNT-WIDE PROP-FIRM RISK AUTHORITY ARMED — shared by every portfolio and \
             atomically persisted before order send"
        );
        if summary.preset != neoethos_core::domain::prop_firm::PropFirmPreset::Ftmo
            && summary.preset != neoethos_core::domain::prop_firm::PropFirmPreset::None
        {
            tracing::error!(
                target: "neoethos_app::live_trading",
                preset = summary.preset.as_str(),
                reset_zone = initial_period.reset_zone,
                "this preset has no authoritative reset timezone encoded; its daily boundary \
                 remains UTC until the firm's current contract is evidenced"
            );
        }
        active_prop_firm_period = Some(initial_period);
        prop_firm_identity = Some(identity);
        prop_firm_authority = Some(authority);
    }
    let initial_entry_day = active_prop_firm_period
        .map(|period| period.day_id)
        .unwrap_or_else(|| utc_day_id(chrono::Utc::now()));
    let initial_entries = account_entry_authority
        .lock()
        .map_err(|_| anyhow!("account-entry authority lock is poisoned at startup"))?
        .prepare_day(initial_entry_day)?;
    tracing::warn!(
        target: "neoethos_app::live_trading",
        %symbol,
        accounting_day = initial_entry_day,
        entries_today = initial_entries,
        mode = if trading_mode_risky { "risky_utc" } else { "prop_firm_local" },
        "DURABLE ACCOUNT ENTRY AUTHORITY ARMED — shared by every portfolio and preserved across app restarts"
    );
    // Account-wide daily entry cap (2026-08-08): `risk.max_trades_per_day`
    // sat in the operator config with NOTHING on the entry path reading it —
    // his engines took 20-47 entries/day each against a configured 8. Armed
    // only by `risk.max_trades_per_day_enabled` (default false ⇒ `None` here
    // ⇒ behaviour unchanged); `max_trades_per_day: 0` disables like the other
    // caps. Both modes now use the same durable account authority; only the
    // caller-proved reset calendar differs (firm-local versus UTC).
    let daily_entry_cap: Option<u32> = sizing
        .as_ref()
        .filter(|s| s.risk.max_trades_per_day_enabled)
        .map(|s| s.risk.max_trades_per_day)
        .filter(|&cap| cap > 0)
        .map(|cap| u32::try_from(cap).unwrap_or(u32::MAX));
    if let Some(cap) = daily_entry_cap {
        if let Some(period) = active_prop_firm_period {
            tracing::warn!(
                target: "neoethos_app::live_trading",
                %symbol,
                cap,
                firm_day = period.day_id,
                reset_zone = period.reset_zone,
                "DAILY ENTRY CAP ARMED — durable account-wide count shared by every portfolio; resets only at the firm's verified day boundary"
            );
        } else {
            tracing::warn!(
                target: "neoethos_app::live_trading",
                %symbol,
                cap,
                utc_day = initial_entry_day,
                "DAILY ENTRY CAP ARMED — Risky Mode uses the durable account-wide count and resets only at UTC midnight, never on app restart"
            );
        }
    }
    loop {
        if stop.load(Ordering::Relaxed) {
            tracing::info!(target: "neoethos_app::live_trading", "stop requested");
            break;
        }

        // Sleep until just after the next bar boundary — but INTERRUPTIBLY.
        // A single long sleep made Stop appear dead: on H1 the loop wouldn't
        // re-check the stop flag for up to an hour. Poll it every 500ms so Stop
        // (and Stop-all) takes effect within ~½s on any timeframe.
        let now_ms = chrono::Utc::now().timestamp_millis();
        let next_boundary = (now_ms / bar_ms + 1) * bar_ms;
        let wait_ms = (next_boundary - now_ms + 3_000).max(5_000) as u64;
        tracing::debug!(
            target: "neoethos_app::live_trading",
            wait_secs = wait_ms / 1000,
            "waiting for next bar"
        );
        let mut waited: u64 = 0;
        let mut stop_requested = false;
        while waited < wait_ms {
            if stop.load(Ordering::Relaxed) {
                stop_requested = true;
                break;
            }
            let chunk = (wait_ms - waited).min(500);
            tokio::time::sleep(Duration::from_millis(chunk)).await;
            waited += chunk;
        }
        if stop_requested || stop.load(Ordering::Relaxed) {
            break;
        }

        // ── Broker-environment re-check (W2, 2026-08-09) ─────────────────────
        // The demo forward-test gate is an ADMISSION decision: it answers "has
        // this strategy earned the right to trade THIS account with real
        // money", once, at start. That is the right shape — auto-cull and the
        // drawdown breakers handle the ongoing protection, and re-running the
        // whole gate every bar would turn a metric the audit already flags as
        // imprecise (`max_drawdown_pct` is the ACCOUNT equity curve, shared
        // with manual trades and every other running engine) into a hair
        // trigger that halts a live engine holding a position.
        //
        // What was actually broken is that the admission was granted against
        // one environment and the orders went to another: `prepare_new_order`
        // re-reads `ctrader.environment` from disk on EVERY order
        // (broker_api.rs:218/:270), so flipping Demo → Live in Settings routed
        // a running engine's next order to real money through a gate evaluated
        // against a demo account.
        //
        // So: capture at start (see `start`), compare here, and REFUSE TO
        // CONTINUE on any change. This is placed immediately after the sleep
        // and before the first broker call of the iteration, so nothing —
        // neither an entry, nor a force-close, nor a trailing amend — is sent
        // to an account this engine was never admitted to. A change in EITHER
        // direction stops the engine: the account id itself changes with the
        // environment, so `open_position`, `opened_ids`, `pending_experience`
        // and the day-start balance all refer to the previous account and
        // carrying them across is unsound regardless of which way it went.
        let env_now_is_live = crate::app_services::live_gate::active_env_is_live();
        if env_now_is_live != gated_env_is_live {
            tracing::error!(
                target: "neoethos_app::live_trading",
                %symbol, %base_tf,
                gated_env_is_live,
                env_now_is_live,
                open_position_id = ?open_position.map(|(id, _)| id),
                "STOPPING: the cTrader broker environment changed while this \
                 engine was running (Demo <-> Live). The demo forward-test gate \
                 admitted this strategy against the OTHER environment and was \
                 never re-consulted, so no further order will be sent. Any \
                 position opened under the previous environment belongs to the \
                 previous account and is NOT closed by this engine — check the \
                 broker. Restart the engine to re-run the gate against the \
                 environment now selected."
            );
            if let Ok(mut s) = status.lock() {
                s.last_signal = Some(format!(
                    "STOPPED: broker environment changed ({} -> {}) — restart to re-gate",
                    if gated_env_is_live { "Live" } else { "Demo" },
                    if env_now_is_live { "Live" } else { "Demo" },
                ));
            }
            break;
        }

        // ── Fetch base-TF bars (with configurable retry) ─────────────────────
        let max_tries = crate::app_services::env_overrides::ctrader_stream_max_attempts();
        let mut base_snapshot_opt = None;
        for attempt in 0..max_tries {
            let sym = symbol.clone();
            let tf = base_tf.clone();
            match tokio::task::spawn_blocking(move || {
                fetch_recent_broker_trendbar_snapshot_blocking(&sym, &tf, warmup)
            })
            .await?
            {
                Ok(snapshot) => {
                    base_snapshot_opt = Some(snapshot);
                    break;
                }
                Err(e) => {
                    let last = attempt + 1 == max_tries;
                    tracing::warn!(
                        target: "neoethos_app::live_trading",
                        error = %e, attempt, max_tries, last,
                        "fetch base-TF bars failed"
                    );
                    if !last {
                        let backoff_ms =
                            crate::app_services::env_overrides::ctrader_stream_backoff_base_ms()
                                * (1u64 << attempt.min(4));
                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    }
                }
            }
        }
        let base_snapshot = match base_snapshot_opt {
            Some(snapshot) => snapshot,
            None => continue,
        };

        // Check if there really is a new bar
        let latest_ts = base_snapshot
            .bars()
            .last()
            .map(|bar| bar.timestamp_ms)
            .unwrap_or(0);
        if latest_ts <= last_bar_ts {
            tracing::debug!(
                target: "neoethos_app::live_trading",
                last_bar_ts, latest_ts,
                "no new bar yet"
            );
            continue;
        }
        last_bar_ts = latest_ts;

        let mut reconciled_open_position = false;
        // ── Broker reconcile: account THIS engine's closed trades ─────────────
        // Reads the broker's signed closing-deal components for positions we
        // opened — catches SL/TP exits, not just engine-initiated closes. Every
        // partial close is deduplicated by deal id, but no money/risk/learning
        // side effect occurs until the same broker snapshot proves the whole
        // position is flat. On N consecutive losses the strategy is
        // permanently retired (blacklisted) and the engine stops.
        //
        // 2026-07-18 deep-audit fix: this MUST run whenever we track open ids,
        // not only when culling is configured — under hold-to-bracket parity
        // (below) it is the ONLY place a broker-side SL/TP exit clears
        // `open_position`; gating it on cull settings would leave the engine
        // holding a phantom position forever and never re-entering.
        if !opened_ids.is_empty() {
            if let Ok(Ok(runtime)) = tokio::task::spawn_blocking(
                crate::app_services::broker_api::fetch_account_runtime_blocking,
            )
            .await
            {
                let runtime_is_live = matches!(
                    runtime.environment,
                    crate::app_services::ctrader_live_auth::CTraderEnvironment::Live
                );
                let runtime_identity_matches = runtime_is_live == gated_env_is_live
                    && runtime.trader.account_id == gated_account_id
                    && runtime.reconcile.account_id == gated_account_id
                    && runtime
                        .recent_deals
                        .iter()
                        .all(|deal| deal.account_id == gated_account_id)
                    && runtime.deposit_asset_name == account_ccy;
                if !runtime_identity_matches {
                    tracing::error!(
                        target: "neoethos_app::live_trading",
                        %symbol,
                        runtime_environment = ?runtime.environment,
                        runtime_account_id = runtime.trader.account_id,
                        reconcile_account_id = runtime.reconcile.account_id,
                        admitted_account_id = gated_account_id,
                        runtime_currency = %runtime.deposit_asset_name,
                        admitted_currency = %account_ccy,
                        "broker close-money snapshot identity mismatch; refusing all monetary \
                         side effects and keeping every tracked position pending"
                    );
                    continue;
                }

                let canonical_runtime_environment = if runtime_is_live { "live" } else { "demo" };
                if let Some(tracked) = open_position.as_mut() {
                    if let Err(error) = refresh_tracked_position_volume(
                        tracked,
                        gated_symbol_id,
                        runtime.reconcile.positions.iter().map(|position| {
                            (
                                position.position_id,
                                position.symbol_id,
                                position.volume_raw_centi_units,
                            )
                        }),
                    ) {
                        tracing::error!(
                            target: "neoethos_app::live_trading",
                            %error,
                            "broker remaining-position identity/volume refused; retaining position ownership"
                        );
                        continue;
                    }
                }
                let broker_open_position_ids: HashSet<i64> = runtime
                    .reconcile
                    .positions
                    .iter()
                    .map(|position| position.position_id)
                    .collect();
                reconciled_open_position = open_position.is_some_and(|(position_id, _)| {
                    broker_open_position_ids.contains(&position_id)
                });

                for deal in &runtime.recent_deals {
                    if !opened_ids.contains(&deal.position_id) {
                        continue;
                    }
                    // Opening deals have no `closePositionDetail`. A real
                    // closing detail always has an explicit conversion-fee
                    // state, including `NotApplied` when the wire omitted it.
                    let Some(pnl_conversion_fee) = deal.pnl_conversion_fee_state else {
                        continue;
                    };
                    let Some(volume_scale) = opened_volume_scales.get(&deal.position_id) else {
                        unverified_close_positions.insert(deal.position_id);
                        if let Some(accumulator) =
                            close_money_accumulators.get_mut(&deal.position_id)
                        {
                            accumulator.refuse_unverified_fill();
                        }
                        tracing::error!(
                            target: "neoethos_app::live_trading",
                            position_id = deal.position_id,
                            deal_id = deal.deal_id,
                            "closing deal has no exact broker lotSize evidence; refusing its money \
                             and keeping the position pending"
                        );
                        continue;
                    };
                    let wire = BrokerDealWireSnapshotV1 {
                        environment: canonical_runtime_environment.to_string(),
                        account_id: deal.account_id,
                        deal_id: deal.deal_id,
                        order_id: deal.order_id,
                        position_id: deal.position_id,
                        symbol_id: deal.symbol_id,
                        symbol_name: symbol.clone(),
                        deal_status: deal.deal_status.clone(),
                        trade_side: deal.trade_side.clone(),
                        filled_volume_raw_centi_units: deal.filled_volume_raw_centi_units,
                        execution_timestamp_ms: deal.execution_timestamp_ms,
                        execution_price: deal.execution_price,
                        entry_price: deal.entry_price,
                        money_digits: deal.money_digits,
                        gross_profit_raw_scaled: deal.gross_profit_raw_scaled,
                        commission_raw_scaled_signed: deal.commission_raw_scaled_signed,
                        swap_raw_scaled_signed: deal.swap_raw_scaled_signed,
                        pnl_conversion_fee,
                    };
                    match build_broker_deal_money_evidence_v1(
                        &wire,
                        volume_scale,
                        &runtime.deposit_asset_name,
                    ) {
                        Ok(evidence) => {
                            let accumulator = close_money_accumulators
                                .entry(deal.position_id)
                                .or_insert_with(|| {
                                    BrokerPositionMoneyAccumulatorV1::new(&evidence)
                                });
                            if unverified_close_positions.contains(&deal.position_id) {
                                accumulator.refuse_unverified_fill();
                            }
                            if let Err(error) = accumulator.observe_fill(&evidence) {
                                accumulator.refuse_unverified_fill();
                                unverified_close_positions.insert(deal.position_id);
                                tracing::error!(
                                    target: "neoethos_app::live_trading",
                                    position_id = deal.position_id,
                                    deal_id = deal.deal_id,
                                    error = %error,
                                    "closing-deal identity/dedup refusal; no monetary side effect"
                                );
                            }
                        }
                        Err(error) => {
                            unverified_close_positions.insert(deal.position_id);
                            if let Some(accumulator) =
                                close_money_accumulators.get_mut(&deal.position_id)
                            {
                                accumulator.refuse_unverified_fill();
                            }
                            tracing::error!(
                                target: "neoethos_app::live_trading",
                                position_id = deal.position_id,
                                deal_id = deal.deal_id,
                                error = %error,
                                "broker closing-deal money refused; keeping position pending"
                            );
                        }
                    }
                }

                let mut finalized_positions = Vec::new();
                for position_id in opened_ids.iter().copied().collect::<Vec<_>>() {
                    let position_still_open = broker_open_position_ids.contains(&position_id);
                    let close_timestamp_ms = runtime
                        .recent_deals
                        .iter()
                        .filter(|deal| {
                            deal.position_id == position_id
                                && deal.pnl_conversion_fee_state.is_some()
                        })
                        .map(|deal| deal.execution_timestamp_ms)
                        .max();
                    let Some(accumulator) = close_money_accumulators.get_mut(&position_id) else {
                        continue;
                    };
                    if !position_still_open {
                        let expected_entry_filled_volume =
                            opened_entry_filled_volumes.get(&position_id).copied();
                        let complete_volume =
                            expected_entry_filled_volume.is_some_and(|expected| {
                                accumulator.verify_complete_filled_volume(expected).is_ok()
                            });
                        if close_timestamp_ms.is_none() || !complete_volume {
                            accumulator.refuse_unverified_fill();
                            unverified_close_positions.insert(position_id);
                            tracing::error!(
                                target: "neoethos_app::live_trading",
                                position_id,
                                expected_entry_filled_volume_raw_centi_units =
                                    ?expected_entry_filled_volume,
                                observed_close_filled_volume_raw_centi_units =
                                    accumulator.filled_volume_raw_centi_units(),
                                close_timestamp_ms = ?close_timestamp_ms,
                                "flat broker position has incomplete close-fill identity/volume; \
                                 refusing all monetary side effects"
                            );
                        }
                    }
                    match accumulator.finalize_if_position_closed(position_still_open) {
                        Ok(Some(closed)) => {
                            if let Some(close_timestamp_ms) = close_timestamp_ms {
                                finalized_positions.push((closed, close_timestamp_ms));
                            }
                        }
                        Ok(None) => {}
                        Err(error) => tracing::error!(
                            target: "neoethos_app::live_trading",
                            position_id,
                            error = %error,
                            "flat broker position lacks complete verified close money; \
                             refusing all monetary side effects and keeping it pending"
                        ),
                    }
                }

                for (closed, close_timestamp_ms) in finalized_positions {
                    let position_id = closed.position_id();
                    let net = closed.component_sum_account_currency().amount();
                    opened_ids.remove(&position_id);
                    opened_entry_filled_volumes.remove(&position_id);
                    opened_volume_scales.remove(&position_id);
                    close_money_accumulators.remove(&position_id);
                    unverified_close_positions.remove(&position_id);
                    net_pnl_running += net;
                    // W3 (2026-08-09): feed the Risky Mode kill switch its
                    // ONLY input. `net` is now the checked sum of exact signed
                    // broker components, typed in its deposit currency.
                    if let Some(m) = risky_manager.as_mut() {
                        m.record_trade_outcome(net);
                        tracing::info!(
                            target: "neoethos_app::live_trading",
                            position_id,
                            component_sum_account_currency = net,
                            bankroll = m.current_bankroll_usd(),
                            stage_idx = m.current_stage().stage_idx,
                            daily_loss = m.daily_loss_accumulated_usd(),
                            monthly_loss = m.monthly_loss_accumulated_usd(),
                            "risky-mode kill switch: verified closed-position money recorded"
                        );
                    }
                    // Feed the account-wide prop-firm authority only after the broker has
                    // proved the whole position flat and the signed monetary
                    // components complete. The entry snapshot supplies the
                    // behavioral fields used by the revenge-pattern gate;
                    // missing fields are left absent rather than invented.
                    if let Some(authority) = prop_firm_authority.as_ref() {
                        let snapshot = pending_experience.get(&position_id);
                        let trade = neoethos_core::domain::risk::ClosedTrade {
                            entry_time_sec: snapshot
                                .map(|experience| (experience.entry_ts_ms.max(0) / 1000) as u64)
                                .unwrap_or((close_timestamp_ms.max(0) / 1000) as u64),
                            exit_time_sec: (close_timestamp_ms.max(0) / 1000) as u64,
                            pnl: net,
                            size: snapshot
                                .map(|experience| experience.lots)
                                .unwrap_or_else(|| closed.actual_filled_lots()),
                            direction: snapshot.map(|experience| i32::from(experience.direction)),
                        };
                        match authority.lock() {
                            Ok(mut guard) => match guard.record_closed_trade(trade) {
                                Ok(revenge_window) => tracing::info!(
                                    target: "neoethos_app::live_trading",
                                    position_id,
                                    component_sum_account_currency = net,
                                    had_entry_snapshot = snapshot.is_some(),
                                    revenge_window,
                                    "account-wide prop-firm authority: verified closed trade persisted"
                                ),
                                Err(error) => tracing::error!(
                                    target: "neoethos_app::live_trading",
                                    position_id,
                                    error = %error,
                                    "closed trade reached account risk in memory but its durable checkpoint failed; future entries fail closed while persistence remains unavailable"
                                ),
                            },
                            Err(_) => tracing::error!(
                                target: "neoethos_app::live_trading",
                                position_id,
                                "account-risk authority lock is poisoned; no new prop-firm entry can be authorised"
                            ),
                        }
                    }
                    if net < 0.0 {
                        consecutive_losses += 1;
                    } else {
                        consecutive_losses = 0;
                    }
                    // Complete + persist the experience pair (entry features
                    // → verified realized outcome) for offline live-learning.
                    if let Some(mut exp) = pending_experience.remove(&position_id) {
                        exp.close_ts_ms = Some(close_timestamp_ms);
                        exp.net_profit = Some(net);
                        crate::app_services::experience_store::record(&exp);
                    }
                    recent_results.push_back(net > 0.0);
                    while recent_results.len() > cull_window {
                        recent_results.pop_front();
                    }
                    // If the broker closed OUR tracked position (SL/TP), drop it
                    // so trailing doesn't try to amend a dead position.
                    if open_position.map(|(id, _)| id) == Some(position_id) {
                        open_position = None;
                    }
                    if opened_ids.is_empty() {
                        if let Ok(mut s) = status.lock() {
                            s.open_position_id = None;
                            s.clear_position_telemetry(live_trading_policy.trailing_enabled);
                            s.last_exit_reason = Some(
                                "broker_reported_close_unclassified_sl_tp_or_manual".to_string(),
                            );
                        }
                    }
                    tracing::info!(
                        target: "neoethos_app::live_trading",
                        position_id,
                        component_sum_account_currency = net,
                        close_deal_count = closed.deal_count(),
                        closed_lots = closed.actual_filled_lots(),
                        consecutive_losses,
                        "auto-cull: verified flat position accounted"
                    );
                }
                let wins = recent_results.iter().filter(|w| **w).count();
                let window_wr_pct = if recent_results.is_empty() {
                    None
                } else {
                    Some(wins as f64 / recent_results.len() as f64 * 100.0)
                };
                if let Ok(mut s) = status.lock() {
                    s.consecutive_losses = consecutive_losses;
                    s.window_win_rate_pct = window_wr_pct;
                    s.window_trades = recent_results.len() as u32;
                }

                // Either criterion retires: a losing STREAK, or a FULL window
                // whose win rate sits under the profitability floor.
                let mut cull_reason = pending_retirement_reason.clone();
                if cull_reason.is_none()
                    && cull_threshold > 0
                    && consecutive_losses >= cull_threshold
                {
                    cull_reason = Some(format!(
                        "{consecutive_losses} consecutive losing trades (demo/live auto-cull)"
                    ));
                } else if cull_reason.is_none()
                    && cull_min_wr > 0.0
                    && recent_results.len() >= cull_window
                {
                    if let Some(wr) = window_wr_pct {
                        if wr < cull_min_wr {
                            cull_reason = Some(format!(
                                "win rate {wr:.0}% over the last {} trades is below the {cull_min_wr:.0}% floor (demo/live auto-cull)",
                                recent_results.len()
                            ));
                        }
                    }
                }
                if let Some(reason) = cull_reason {
                    // Retirement is an exit-only decision, not a temporary
                    // rolling statistic. A profitable final close must not
                    // reset the loss streak and silently resume entries.
                    pending_retirement_reason = Some(reason.clone());
                    tracing::warn!(
                        target: "neoethos_app::live_trading",
                        %symbol, portfolio_path = %portfolio_path,
                        %reason, net_pnl = net_pnl_running,
                        "AUTO-CULL: retiring strategy (blacklist)"
                    );
                    // A close response, including a partial fill, does not
                    // prove the position is flat. Keep the exit-only loop
                    // alive until the broker/close-money reconciliation above
                    // clears ownership, then finalise retirement.
                    if let Some((pos_id, vol)) = open_position {
                        let close_report = if reconciled_open_position {
                            let close_attempt = tokio::task::spawn_blocking(move || {
                                close_position_blocking(
                                    pos_id,
                                    vol,
                                    Some(gated_env_is_live),
                                    Some(gated_account_id),
                                )
                            })
                            .await;
                            match close_attempt {
                                Ok(Ok(outcome)) => format!(
                                    "broker returned {:?}; awaiting complete-close reconciliation",
                                    outcome.status
                                ),
                                Ok(Err(error)) => format!("close failed: {error}"),
                                Err(error) => format!("close task failed: {error}"),
                            }
                        } else {
                            "broker position absent; awaiting verified closing-fill accounting"
                                .to_string()
                        };
                        tracing::warn!(
                            target: "neoethos_app::live_trading",
                            %symbol, position_id = pos_id, %close_report,
                            "AUTO-CULL RETIREMENT PENDING — position remains tracked until broker reconciliation proves it flat"
                        );
                        if let Ok(mut s) = status.lock() {
                            s.retired = true;
                            s.running = true;
                            s.open_position_id = Some(pos_id);
                            s.last_signal = Some(format!(
                                "retirement pending for position {pos_id}: {close_report}; still supervised"
                            ));
                        }
                        continue;
                    }
                    if let Some(fp) =
                        crate::app_services::strategy_blacklist::fingerprint_file(&portfolio_path)
                    {
                        crate::app_services::strategy_blacklist::retire(
                            crate::app_services::strategy_blacklist::BlacklistEntry {
                                fingerprint: fp,
                                portfolio_path: portfolio_path.clone(),
                                symbol: Some(symbol.clone()),
                                reason,
                                consecutive_losses,
                                net_pnl: net_pnl_running,
                                retired_at_unix_ms: chrono::Utc::now().timestamp_millis(),
                            },
                        );
                    }
                    if let Ok(mut s) = status.lock() {
                        s.retired = true;
                        s.running = false;
                        s.open_position_id = None;
                        s.clear_position_telemetry(live_trading_policy.trailing_enabled);
                        s.last_exit_reason = Some("auto_cull_retirement".to_string());
                    }
                    // Close the loop: the retirement left a coverage gap on this
                    // (symbol, base_tf) — queue a fresh Discovery to refill it.
                    //
                    // The retired strategy cannot come back (#218/#219): it
                    // cannot be SELECTED (`is_blacklisted`, matched on the GENE
                    // rather than the file bytes) and, since 2026-08-10, it
                    // cannot be PROMOTED either — `neoethos_search::
                    // live_portfolio` drops any retired RULE from the artifact
                    // the trader consumes, reading this same blacklist file.
                    // The GA can still spend time re-deriving it inside the run
                    // queued on the next line; when it does, it says so.
                    crate::app_services::rediscovery::request(symbol.clone(), base_tf.clone());
                    break;
                }
            }
        }

        // Reconcile account identity and exact remaining volume BEFORE sending
        // a weekend close. A successful response still cannot replace the next
        // broker/close-money reconciliation's proof of complete closure.
        if kill_zones_enabled && reconciled_open_position && weekend_kill_zone(latest_ts).0 {
            if let Some((pos_id, vol)) = open_position {
                let result = tokio::task::spawn_blocking(move || {
                    close_position_blocking(
                        pos_id,
                        vol,
                        Some(gated_env_is_live),
                        Some(gated_account_id),
                    )
                })
                .await;
                let close_report = match result {
                    Ok(Ok(outcome)) => format!(
                        "broker returned {:?}; awaiting verified complete-close reconciliation",
                        outcome.status
                    ),
                    Ok(Err(error)) => format!("close failed: {error}"),
                    Err(error) => format!("close task failed: {error}"),
                };
                tracing::info!(
                    target: "neoethos_app::live_trading",
                    %symbol, position_id = pos_id, %close_report,
                    "weekend close remains supervised until broker reconciliation proves it flat"
                );
                if let Ok(mut s) = status.lock() {
                    s.open_position_id = Some(pos_id);
                    s.last_signal = Some(format!("weekend close: {close_report}"));
                }
                // Do not amend protection or enter again using a snapshot
                // taken before this close request.
                continue;
            }
        }

        // Keep a plain base frame for exit geometry and broker order pricing.
        // Feature construction below uses only the canonical snapshot reopened
        // from the exact broker-bound Vortex publications.
        let base_ohlcv = bars_to_ohlcv(base_snapshot.bars());
        let bar_high = base_ohlcv.high.last().copied().unwrap_or(pos_entry_px);
        let bar_low = base_ohlcv.low.last().copied().unwrap_or(pos_entry_px);

        // Observe maximum favourable excursion even when the validated policy
        // deliberately has trailing disabled. This is supervision evidence: it
        // makes a large unrealised win that later closes at SL visible instead
        // of disappearing from the engine state.
        if open_position.is_some() && pos_entry_px > 0.0 && pos_sl_pips > 0.0 {
            if pos_is_long {
                pos_extreme = pos_extreme.max(bar_high);
            } else {
                pos_extreme = if pos_extreme > 0.0 {
                    pos_extreme.min(bar_low)
                } else {
                    bar_low
                };
            }
            let favorable_move_r = favorable_excursion_r(
                pos_entry_px,
                pos_extreme,
                pos_sl_pips,
                exact_pip_size,
                pos_is_long,
            );
            let protection_state = match exit_policy {
                Some(policy) if !policy.trailing_enabled => "disabled_by_search",
                Some(policy)
                    if favorable_move_r
                        .is_some_and(|move_r| move_r >= policy.trailing_be_trigger_r) =>
                {
                    if pos_trail_px > 0.0 {
                        "broker_confirmed"
                    } else {
                        "trigger_reached_pending_broker"
                    }
                }
                Some(_) => "armed_waiting_trigger",
                None => "missing_policy",
            };
            if let Ok(mut s) = status.lock() {
                s.position_entry_price = Some(pos_entry_px);
                s.initial_stop_pips = Some(pos_sl_pips);
                s.favorable_extreme_price = Some(pos_extreme);
                s.favorable_move_r = favorable_move_r;
                s.protection_state = Some(protection_state.to_string());
            }
        }

        // ── Trailing stop — PARITY with the discovery backtest ────────────────
        //
        // CORRECTED 2026-08-09 (#208 / #74). This comment used to read:
        //
        //   "eval.rs hardcodes break-even + trailing ALWAYS ON: once the
        //    favorable move reaches +1R (= sl_pips) the stop trails 1×SL behind
        //    the running extreme ... Without this, trades the backtest saved at
        //    break-even become full losses."
        //
        // Every clause of that was true at breakfast and false by lunchtime.
        // `eval.rs` no longer hardcodes anything: it reads
        // `settings.trailing_enabled` / `trailing_be_trigger_r` /
        // `trailing_atr_multiplier` / `trailing_min_lock_pips`
        // (`neoethos-search/src/eval.rs:1045-1058`, `:1078-1090`), fed from
        // `models.exit_policy` via `strategy_gene.rs:867`, and the exact
        // resolved values now travel inside the live-portfolio artifact. So
        // "ALWAYS ON" is wrong,
        // "+1R" and "1×SL" are configured rather than fixed, and the last
        // sentence has its sign backwards for the shipped policy: with the
        // policy OFF, the trades the backtest scores at the take-profit were
        // being converted live into break-even scratches.
        //
        // The geometry below is the same shape `eval.rs` computes — the trigger
        // is `be_trigger_r × sl_pips`, the trail sits `stop_multiplier × sl_pips`
        // behind the running extreme, and the locked-profit floor is
        // `min_lock_pips`. Ratchet-only; no intra-bar look-ahead (we act on the
        // just-closed bar, the broker enforces it next). Using the running
        // extreme rather than eval's per-bar high is equivalent: both ratchet
        // monotonically and the trail distance is constant, so
        // `max_i(hi_i) - d == max_i(hi_i - d)`.
        //
        // When the sealed policy is OFF, NOTHING here runs and no stop is moved.
        let trailing = exit_policy.filter(|p| p.trailing_enabled);
        if let (Some(policy), Some((pos_id, _))) = (trailing, open_position) {
            if pos_sl_pips > 0.0 && pos_entry_px > 0.0 {
                let pip = exact_pip_size;
                // Guard the three configured numbers the same way the rest of
                // this loop guards config: a corrupt field must not silently
                // become "trail at zero distance", which would close every
                // winning position at its own high.
                let trigger_r = policy.trailing_be_trigger_r;
                let stop_mult = policy.trailing_stop_multiplier;
                let lock_pips = policy.trailing_min_lock_pips;
                if !(trigger_r.is_finite()
                    && trigger_r > 0.0
                    && stop_mult.is_finite()
                    && stop_mult > 0.0
                    && lock_pips.is_finite()
                    && lock_pips >= 0.0)
                {
                    tracing::error!(
                        target: "neoethos_app::live_trading",
                        %symbol,
                        be_trigger_r = trigger_r,
                        stop_multiplier = stop_mult,
                        min_lock_pips = lock_pips,
                        "sealed live-portfolio policy has trailing ENABLED but its geometry \
                         is unusable — REFUSING TO MOVE THE STOP this bar"
                    );
                } else {
                    let r_dist = pos_sl_pips * pip; // 1R in price units
                    let trigger_dist = trigger_r * r_dist;
                    let trail_dist = stop_mult * r_dist;
                    let mut new_trail: Option<f64> = None;
                    // Same floor the backtest applies (`eval.rs:1052`): once the
                    // trail engages it never sits closer to entry than the locked
                    // profit. Without it the live stop protects a different amount
                    // than the strategy was scored on.
                    let locked = lock_pips * pip;
                    if pos_is_long {
                        if pos_extreme - pos_entry_px >= trigger_dist {
                            let candidate = (pos_extreme - trail_dist).max(pos_entry_px + locked);
                            if pos_trail_px == 0.0 || candidate > pos_trail_px {
                                new_trail = Some(candidate);
                            }
                        }
                    } else {
                        if pos_entry_px - pos_extreme >= trigger_dist {
                            let candidate = (pos_extreme + trail_dist).min(pos_entry_px - locked);
                            if pos_trail_px == 0.0 || candidate < pos_trail_px {
                                new_trail = Some(candidate);
                            }
                        }
                    }
                    if let Some(raw) = new_trail {
                        let sl_price = (raw / pip).round() * pip; // snap to a broker-valid pip grid
                        // #199 CLOSED. This was the last order-path call still
                        // resolving credentials WITHOUT the environment it was
                        // admitted against, so a Demo->Live flip mid-iteration
                        // could send a stop-modification to an environment this
                        // engine was never admitted to. The binding function was
                        // written in the same wave and then never called — the
                        // hand-off was routed to the build phase, and the build
                        // phase edited no files, so it fell on the floor. Its own
                        // doc-comment at broker_api.rs:1323 already said the
                        // autopilot "must call amend_position_sltp_expecting".
                        // Now it does.
                        //
                        // `gated_env_is_live` is this run's admission (parameter
                        // at :512), the same value close_position_blocking is
                        // already given at :1222 and :1371. Some(_) REFUSES the
                        // amend outright when broker_credentials.toml now names
                        // the other environment; no request leaves the process.
                        let expected_env = gated_env_is_live;
                        let amend = tokio::task::spawn_blocking(move || {
                            amend_position_sltp_expecting_account(
                                pos_id,
                                Some(sl_price),
                                None,
                                None,
                                Some(expected_env),
                                Some(gated_account_id),
                            )
                        })
                        .await;
                        // The Result was dropped twice over before: once by the
                        // missing check, once by `let _`. The callee logs its own
                        // failure at error; this names which position and which
                        // stop the operator did NOT get, and skips the
                        // "advanced" line below so the log cannot claim a move
                        // that never happened.
                        let advanced = match amend {
                            Ok(Ok(outcome)) => match commit_broker_confirmed_trail(
                                &mut pos_trail_px,
                                sl_price,
                                pos_id,
                                &outcome,
                            ) {
                                Ok(()) => true,
                                Err(error) => {
                                    if let Ok(mut s) = status.lock() {
                                        s.protection_state =
                                            Some("trigger_reached_pending_broker".to_string());
                                        s.last_protection_error = Some(error.to_string());
                                    }
                                    tracing::error!(
                                        target: "neoethos_app::live_trading",
                                        position_id = pos_id, intended_sl = sl_price,
                                        %error,
                                        "TRAILING STOP NOT CONFIRMED — local ratchet remains at the last broker-confirmed level and this stop remains retryable"
                                    );
                                    false
                                }
                            },
                            Ok(Err(error)) => {
                                if let Ok(mut s) = status.lock() {
                                    s.protection_state =
                                        Some("trigger_reached_pending_broker".to_string());
                                    s.last_protection_error = Some(error.to_string());
                                }
                                tracing::error!(
                                    target: "neoethos_app::live_trading",
                                    position_id = pos_id, intended_sl = sl_price,
                                    %error,
                                    "TRAILING STOP NOT MOVED — the broker refused the amend.                                      The position is still protected by its previous stop,                                      not by the one this iteration computed."
                                );
                                false
                            }
                            Err(join) => {
                                if let Ok(mut s) = status.lock() {
                                    s.protection_state =
                                        Some("trigger_reached_pending_broker".to_string());
                                    s.last_protection_error = Some(join.to_string());
                                }
                                tracing::error!(
                                    target: "neoethos_app::live_trading",
                                    position_id = pos_id, intended_sl = sl_price,
                                    error = %join,
                                    "TRAILING STOP NOT MOVED — the amend task failed to run."
                                );
                                false
                            }
                        };
                        if advanced {
                            if let Ok(mut s) = status.lock() {
                                s.protection_state = Some("broker_confirmed".to_string());
                                s.confirmed_stop_price = Some(pos_trail_px);
                                s.last_protection_error = None;
                            }
                            tracing::info!(
                                target: "neoethos_app::live_trading",
                                position_id = pos_id, new_sl = sl_price, extreme = pos_extreme,
                                be_trigger_r = trigger_r,
                                stop_multiplier = stop_mult,
                                min_lock_pips = lock_pips,
                                "trailing stop advanced (sealed discovery geometry)"
                            );
                        }
                    }
                }
            }
        }

        let mut direct_snapshots = vec![base_snapshot];
        let mut complete_direct_series = true;
        for htf in &higher_tfs {
            let sym = symbol.clone();
            let tf = htf.clone();
            match tokio::task::spawn_blocking(move || {
                fetch_recent_broker_trendbar_snapshot_blocking(&sym, &tf, warmup)
            })
            .await?
            {
                Ok(snapshot) => direct_snapshots.push(snapshot),
                Err(e) => {
                    tracing::warn!(
                        target: "neoethos_app::live_trading",
                        tf = %htf, error = %e,
                        "failed to fetch required higher-TF bars; rejecting the complete live bar"
                    );
                    complete_direct_series = false;
                    break;
                }
            }
        }
        if !complete_direct_series {
            continue;
        }

        let canonical_snapshot = match crate::app_services::live_feature_snapshot::LiveCanonicalFeatureSnapshot::publish_for_artifact(
            &artifact,
            direct_snapshots,
        ) {
            Ok(snapshot) => snapshot,
            Err(e) => {
                tracing::warn!(
                    target: "neoethos_app::live_trading",
                    error = %e,
                    "canonical live broker publication failed; rejecting the complete live bar"
                );
                continue;
            }
        };

        // ── Feature computation ───────────────────────────────────────────────
        let live_features = match artifact.prepare_live_features(canonical_snapshot.dataset()) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(
                    target: "neoethos_app::live_trading",
                    error = %e,
                    "feature computation failed, skipping bar"
                );
                continue;
            }
        };

        let aligned = match artifact.project_live_features(&live_features) {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(
                    target: "neoethos_app::live_trading",
                    error = %e,
                    "artifact-bound live feature projection/normalization failed, skipping bar"
                );
                continue;
            }
        };

        if aligned.n_samples() == 0 {
            tracing::warn!(
                target: "neoethos_app::live_trading",
                "empty aligned feature frame, skipping bar"
            );
            continue;
        }

        // ── Gene signal + the strategy's OWN brackets (last bar) ──────────────
        // Replay the saved SMC/confidence/adaptive recipe, not ambient defaults.
        let netted = neoethos_trader::combine_gene_signals_with_archived_policy(
            &genes,
            &aligned,
            &base_ohlcv,
            &live_trading_policy,
        )
        .with_context(|| format!("synthesize live gene signals for {symbol} {base_tf}"))?;
        // Keep the exact source snapshot alive for model input, but retain no
        // whole Search cube while preparing a separately fitted model cube.
        let direction = netted.directions.last().copied().unwrap_or(Direction::Flat);
        let gene_confidence = netted.confidences.last().copied().unwrap_or(0.0);
        // Gene-derived SL/TP (pips) for THIS bar: we place the STRATEGY'S own
        // brackets, never an imposed stop. 0.0 ⇒ a signal-exit-only strategy, so
        // the live order stays bracket-free (exactly what the backtest does).
        let gene_sl = netted.sl_pips.last().copied().unwrap_or(0.0);
        let gene_tp = netted.tp_pips.last().copied().unwrap_or(0.0);
        let experience_feature_row = if direction != Direction::Flat {
            Some(
                aligned
                    .dense_window(aligned.n_samples() - 1, aligned.n_samples())
                    .map(|window| window.values.row(0).iter().copied().collect::<Vec<f64>>()),
            )
        } else {
            None
        };
        drop(aligned);
        drop(live_features);

        bars_evaluated += 1;
        let signal_label = format!("{direction:?}");
        tracing::info!(
            target: "neoethos_app::live_trading",
            %symbol, %base_tf,
            signal = %signal_label,
            bar_ts = latest_ts,
            bars_evaluated,
            open_position_id = ?open_position.map(|(id, _)| id),
            "bar signal evaluated"
        );

        if let Ok(mut s) = status.lock() {
            s.last_signal = Some(signal_label);
            s.bars_evaluated = bars_evaluated;
            s.open_position_id = open_position
                .map(|(id, _)| id)
                .or_else(|| opened_ids.iter().copied().next());
        }

        // ── Execution ─────────────────────────────────────────────────────────
        // PARITY (2026-07-18 deep audit): the discovery kernel (eval.rs)
        // consults the signal ONLY while FLAT. While a position is open the
        // signal is ignored entirely — exits happen exclusively via SL/TP
        // (broker-enforced live), the trail when `models.exit_policy` arms it,
        // plus the weekend force-close; the
        // production EvaluationConfig ships max_hold_bars = 0. The previous
        // live code closed + reopened on EVERY non-flat bar (paying the
        // spread per bar and resetting the trailing state) and closed on a
        // Flat signal — a trade profile no validated backtest ever had.
        match direction {
            Direction::Long | Direction::Short => {
                if open_position.is_some() || !opened_ids.is_empty() || has_unresolved_broker_entry
                {
                    if let Some(intent) = unresolved_entry_client_order_id.as_deref() {
                        if let Ok(mut s) = status.lock() {
                            s.last_signal = Some(format!(
                                "blocked: unresolved broker entry {intent}; reconciliation required"
                            ));
                        }
                    }
                    // Hold to bracket — the trailing block above keeps the
                    // broker-side stop in sync WHEN the exit policy arms it
                    // (otherwise the original SL/TP stands); nothing to
                    // execute this bar. `opened_ids` also blocks a duplicate
                    // entry when a broker fill omitted exact volume/contract
                    // evidence and therefore remains deliberately pending.
                    continue;
                }

                // News gate (block_on_news): block NEW entries inside the
                // blackout window of a high-impact event for this symbol's
                // currencies. Exits (weekend force-close, auto-cull flatten,
                // broker-side brackets) are never gated — closing reduces
                // risk. Fail-soft: a calendar outage never blocks (see
                // news_calendar.rs).
                let gate_sym = symbol.clone();
                let now_ms = chrono::Utc::now().timestamp_millis();
                if let Ok(Some(event)) = tokio::task::spawn_blocking(move || {
                    crate::app_services::news_calendar::entry_blackout_for(&gate_sym, now_ms)
                })
                .await
                {
                    tracing::warn!(
                        target: "neoethos_app::live_trading",
                        %symbol, event = %event,
                        "entry blocked by news gate (block_on_news) — skipping this bar"
                    );
                    if let Ok(mut s) = status.lock() {
                        s.last_signal = Some(format!("blocked by news: {event}"));
                    }
                    continue;
                }

                // Weekend kill zone — PARITY entry block (Fri ≥20:00 / Mon <00:30
                // UTC): the backtest never entered in these windows.
                if kill_zones_enabled {
                    let (_, block_entry) = weekend_kill_zone(latest_ts);
                    if block_entry {
                        tracing::info!(
                            target: "neoethos_app::live_trading",
                            %symbol, "entry blocked — weekend kill zone (parity with backtest)"
                        );
                        if let Ok(mut s) = status.lock() {
                            s.last_signal = Some("blocked: weekend kill zone".to_string());
                        }
                        continue;
                    }
                }

                // Capture one connection for this entry decision. Both sides must
                // be causal and fresh for the pinned account/environment/symbol.
                // A reconnect invalidates this decision, not silently refreshes it.
                let quote_environment = if gated_env_is_live {
                    CTraderEnvironment::Live
                } else {
                    CTraderEnvironment::Demo
                };
                let entry_quote_session = match live_spots::current_session(
                    gated_account_id,
                    quote_environment,
                ) {
                    Some(session) => session,
                    None => {
                        tracing::warn!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            reason = ?SpotQuoteRefusal::NoActiveSession,
                            "entry blocked — no current quote session for the pinned account/environment"
                        );
                        if let Ok(mut s) = status.lock() {
                            s.last_signal = Some("blocked: quote NoActiveSession".to_string());
                        }
                        continue;
                    }
                };
                let quote_now_ms = chrono::Utc::now().timestamp_millis();
                if let Err(reason) = require_live_entry_spread(
                    gated_account_id,
                    quote_environment,
                    entry_quote_session,
                    gated_symbol_id,
                    exact_pip_size,
                    live_trading_policy.expected_spread_pips_at(quote_now_ms),
                    quote_now_ms,
                ) {
                    tracing::warn!(
                        target: "neoethos_app::live_trading",
                        %symbol, ?reason,
                        "entry blocked — current quote or spread check refused"
                    );
                    if let Ok(mut s) = status.lock() {
                        s.last_signal = Some(format!("blocked: quote/spread {reason:?}"));
                    }
                    continue;
                }

                // Open new position
                let side = if direction == Direction::Long {
                    OrderSide::Buy
                } else {
                    OrderSide::Sell
                };
                let now_utc = chrono::Utc::now();
                let today = utc_day_id(now_utc);

                // Fresh account state at ENTRY time. Both modes MUST use
                // broker-measured equity, including unrealised P/L; a missing or
                // foreign snapshot refuses this bar without latching the breaker.
                // No stale startup balance or missing-position-count fallback.
                let (entry_balance, entry_equity, open_positions_now, entry_context) = {
                    let context_symbol = symbol.clone();
                    let context = match tokio::task::spawn_blocking(move || {
                        fetch_live_entry_context_blocking(
                            &context_symbol,
                            entry_environment,
                            gated_account_id,
                            gated_symbol_id,
                        )
                    })
                    .await
                    {
                        Ok(Ok(context)) => context,
                        Ok(Err(error)) => {
                            tracing::error!(
                                target: "neoethos_app::live_trading",
                                %symbol,
                                rule = "risk.account_snapshot_unresolvable",
                                error = %error,
                                "entry refused — broker equity snapshot unavailable; next bar retries"
                            );
                            if let Ok(mut s) = status.lock() {
                                s.last_signal =
                                    Some("blocked: broker equity unavailable".to_string());
                            }
                            continue;
                        }
                        Err(error) => {
                            tracing::error!(
                                target: "neoethos_app::live_trading",
                                %symbol,
                                rule = "risk.account_snapshot_task",
                                error = %error,
                                "entry refused — broker equity task failed; next bar retries"
                            );
                            if let Ok(mut s) = status.lock() {
                                s.last_signal =
                                    Some("blocked: broker equity task failed".to_string());
                            }
                            continue;
                        }
                    };
                    let snapshot = context.margin();
                    let expected_environment = if gated_env_is_live { "live" } else { "demo" };
                    if context.account_currency() != account_ccy
                        || context.contract() != &startup_symbol_contract
                        || validate_entry_account_values(
                            snapshot.environment_label,
                            snapshot.account_id,
                            snapshot.balance,
                            snapshot.equity,
                            expected_environment,
                            gated_account_id,
                        )
                        .is_err()
                    {
                        tracing::error!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            rule = "risk.account_snapshot_identity",
                            snapshot_environment = snapshot.environment_label,
                            snapshot_account_id = snapshot.account_id,
                            expected_environment,
                            expected_account_id = gated_account_id,
                            balance = snapshot.balance,
                            equity = snapshot.equity,
                            "entry refused — broker equity snapshot identity/value is not authoritative"
                        );
                        if let Ok(mut s) = status.lock() {
                            s.last_signal =
                                Some("blocked: broker equity identity mismatch".to_string());
                        }
                        continue;
                    }
                    (
                        snapshot.balance,
                        snapshot.equity,
                        snapshot.open_position_count,
                        context,
                    )
                };

                if let Some(authority) = prop_firm_authority.as_ref() {
                    let Some(identity) = prop_firm_identity.as_ref() else {
                        tracing::error!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            rule = "risk.account_authority_identity",
                            "entry refused — the account-risk authority has no broker identity"
                        );
                        continue;
                    };
                    let Some(settings) = sizing.as_ref() else {
                        tracing::error!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            rule = "risk.account_authority_settings",
                            "entry refused — the account-risk authority has no resolved settings"
                        );
                        continue;
                    };
                    let current_period = match prop_firm_period(settings.risk.preset, now_utc) {
                        Ok(period) => period,
                        Err(error) => {
                            tracing::error!(
                                target: "neoethos_app::live_trading",
                                %symbol,
                                rule = "risk.firm_period",
                                error = %error,
                                "entry refused — firm-local risk period could not be resolved"
                            );
                            continue;
                        }
                    };
                    let period_changed = active_prop_firm_period != Some(current_period);
                    let snapshot = AccountRiskSnapshot {
                        balance: entry_balance,
                        equity: entry_equity,
                    };
                    match prepare_account_risk_period(authority, identity, current_period, snapshot)
                        .await
                    {
                        Ok(summary) => {
                            active_prop_firm_period = Some(current_period);
                            if period_changed {
                                tracing::warn!(
                                    target: "neoethos_app::live_trading",
                                    %symbol,
                                    firm_day = current_period.day_id,
                                    firm_month = current_period.month_id,
                                    reset_zone = current_period.reset_zone,
                                    day_start_balance = summary.day_start_balance,
                                    "account-wide prop-firm authority rolled to a broker-evidenced period"
                                );
                            }
                        }
                        Err(error) => {
                            tracing::error!(
                                target: "neoethos_app::live_trading",
                                %symbol,
                                rule = "risk.daily_anchor",
                                error = %error,
                                firm_day = current_period.day_id,
                                reset_zone = current_period.reset_zone,
                                "entry refused — durable account-risk state could not be refreshed; exits and trailing continue"
                            );
                            if let Ok(mut s) = status.lock() {
                                s.last_signal = Some(format!(
                                    "REFUSED: account-risk anchor/state unavailable for {}",
                                    current_period.day_id
                                ));
                            }
                            continue;
                        }
                    }
                }
                let entry_day_id = if trading_mode_risky {
                    today
                } else if let Some(period) = active_prop_firm_period {
                    period.day_id
                } else {
                    tracing::error!(
                        target: "neoethos_app::live_trading",
                        %symbol,
                        rule = "risk.daily_entry_period_unprepared",
                        "entry refused — prop-firm mode has no active firm-local entry day"
                    );
                    continue;
                };
                match account_entry_authority.lock() {
                    Ok(mut guard) => {
                        let previous_day = guard.day_id();
                        match guard.prepare_day(entry_day_id) {
                            Ok(count) => {
                                if previous_day != Some(entry_day_id) {
                                    tracing::warn!(
                                        target: "neoethos_app::live_trading",
                                        %symbol,
                                        previous_accounting_day = ?previous_day,
                                        accounting_day = entry_day_id,
                                        entries_today = count,
                                        "durable account entry authority rolled to the caller-proved day"
                                    );
                                }
                            }
                            Err(error) => {
                                tracing::error!(
                                    target: "neoethos_app::live_trading",
                                    %symbol,
                                    rule = "risk.state_persistence",
                                    accounting_day = entry_day_id,
                                    error = %error,
                                    "entry refused — daily entry state could not be durably prepared"
                                );
                                continue;
                            }
                        }
                    }
                    Err(_) => {
                        tracing::error!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            rule = "risk.account_entry_authority_lock",
                            "entry refused — account-entry authority lock is poisoned"
                        );
                        continue;
                    }
                };
                // ── Risky Mode kill switch: period rollover + persisted halt ──
                // (W3, 2026-08-09). Two things happen here, both before a slot
                // is reserved so a refusal costs nothing.
                //
                // 1. Roll the manager's daily / weekly / monthly accumulators.
                //    `reset_*_accumulator` had zero callers, so without this the
                //    ledgers would only ever grow and the day cap would trip
                //    once and stay tripped for the life of the process.
                // 2. Consult the PERSISTED kill-switch cooldown. The in-process
                //    manager loses its state when the app restarts; the
                //    `last_killed_at_utc_ms` timestamp does not. This is what
                //    makes a tripped kill switch mean "stop for 24 h" instead
                //    of "stop until someone restarts the app", and it is the
                //    clock `bridge.rs` clears after the cooldown expires and the
                //    Risk screen renders.
                if let Some(m) = risky_manager.as_mut() {
                    use chrono::Datelike;
                    let d = chrono::Utc::now().date_naive();
                    let iso = d.iso_week();
                    let period = (
                        today,
                        (iso.year().max(0) as u32) * 100 + iso.week(),
                        (d.year().max(0) as u32) * 100 + d.month(),
                    );
                    match risky_period {
                        Some(prev) if prev == period => {}
                        prev => {
                            if prev.map(|p| p.0) != Some(period.0) {
                                m.reset_daily_accumulator();
                            }
                            if prev.map(|p| p.1) != Some(period.1) {
                                m.reset_weekly_accumulator();
                            }
                            if prev.map(|p| p.2) != Some(period.2) {
                                m.reset_monthly_accumulator();
                            }
                            risky_period = Some(period);
                        }
                    }
                }
                if trading_mode_risky
                    && let Some(remaining) =
                        crate::app_services::risky_mode_persistence::kill_switch_cooldown_remaining_secs()
                {
                    tracing::error!(
                        target: "neoethos_app::live_trading",
                        %symbol,
                        rule = "risky_mode.kill_switch_cooldown",
                        cooldown_remaining_secs = remaining,
                        cooldown_remaining_hours = remaining / 3600,
                         "entry refused — the Risky Mode kill switch is TRIPPED. A \
                          previous entry hit a per-day / per-week / per-stage / per-month loss \
                         tier and started the 24h halt. No Risky-Mode entry will be \
                         sent until it elapses (the bridge clears the expired halt; the Risk \
                         screen shows the remaining time). Exits and trailing \
                         continue normally."
                    );
                    if let Ok(mut s) = status.lock() {
                        s.last_signal = Some(format!(
                            "HALTED: risky-mode kill switch, cooldown remaining {}h {}m",
                            remaining / 3600,
                            (remaining % 3600) / 60
                        ));
                    }
                    continue;
                }

                // Base per-trade risk. Risky Mode uses the immutable OOS
                // half-Kelly fraction resolved once at startup; current balance
                // determines the money amount only. PropFirm mode retains its
                // configured fraction.
                let base_risk = if trading_mode_risky {
                    let frac = risky_effective_fraction
                        .expect("Risky startup validates an OOS sizing fraction");
                    tracing::info!(
                        target: "neoethos_app::live_trading",
                        %symbol, bankroll = entry_balance, risk_pct = frac,
                        oos_half_kelly = portfolio_oos_half_kelly,
                        "risky-mode sizing from held-out Search edge"
                    );
                    frac
                } else {
                    risk_fraction
                };

                // LIVE ML gate: the genes chose the direction above; the
                // ensemble may only SHRINK the size (agreement × regime ×
                // anomaly, MlScale mode) or skip the bar on a hard collapse.
                // Inference errors or invalid rows skip only this new entry;
                // existing-position reconciliation and protection ran above.
                // The ML multiplier is not the strategies' measured confidence.
                let ml_multiplier = if let Some(ens) = live_ensemble.as_deref() {
                    match checked_live_ml_entry(
                        direction,
                        budgeted_role_decision_for_last_row(ens, canonical_snapshot.dataset()),
                        &live_blend_cfg,
                    ) {
                        Ok((d, conf)) => {
                            tracing::info!(
                                target: "neoethos_app::live_trading",
                                %symbol, conf,
                                p_buy = d.dir_probs[1], p_sell = d.dir_probs[2],
                                regime_gate = d.regime_gate, anomaly = d.anomaly_scale,
                                "ML gate scaled entry risk (genes kept the direction)"
                            );
                            if let Ok(mut s) = status.lock() {
                                s.last_signal = Some(format!("{direction:?} · ML×{conf:.2}"));
                            }
                            Some(conf)
                        }
                        Err(err) => {
                            tracing::warn!(
                                target: "neoethos_app::live_trading",
                                %symbol, error = %err,
                                "entry skipped — required ML decision unavailable or vetoed; no genes-only fallback, exits and protection continue"
                            );
                            if let Ok(mut s) = status.lock() {
                                s.last_signal = Some(format!("skipped: required ML — {err}"));
                            }
                            continue;
                        }
                    }
                } else {
                    None
                };
                let (base_risk, measured_confidence) =
                    live_entry_sizing_inputs(base_risk, gene_confidence, ml_multiplier)?;

                // Reserve as late as possible, after feature/ML abstentions but
                // before the risk decision and broker send. Every mode uses this
                // same durable counter, atomically shared across portfolio
                // engines; only `entry_day_id`'s proved calendar differs.
                let reservation = match account_entry_authority.lock() {
                    Ok(mut guard) => guard.try_reserve_entry(entry_day_id, daily_entry_cap),
                    Err(_) => {
                        tracing::error!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            rule = "risk.account_entry_authority_lock",
                            "entry refused — account-entry authority lock is poisoned"
                        );
                        continue;
                    }
                };
                let entry_reservation = match reservation {
                    Ok(entries_before) => EntryReservation {
                        authority: account_entry_authority.clone(),
                        day_id: entry_day_id,
                        entries_before,
                    },
                    Err(refusal) => {
                        tracing::warn!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            rule = refusal.rule,
                            detail = %refusal.detail,
                            accounting_day = entry_day_id,
                            mode = if trading_mode_risky { "risky_utc" } else { "prop_firm_local" },
                            cap = ?daily_entry_cap,
                            "entry refused by the durable account-wide reservation gate; exits continue"
                        );
                        if let Ok(mut s) = status.lock() {
                            s.last_signal =
                                Some(format!("REFUSED by {}: {}", refusal.rule, refusal.detail));
                        }
                        continue;
                    }
                };

                // The production recipient for risk.challenge_mode,
                // risk.challenge_phase and risk.recovery_mode_enabled. The gate
                // sees exact broker equity and the clamp only shrinks the
                // already-ML-scaled risk fraction.
                let base_risk = if let (Some(authority), Some(period)) =
                    (prop_firm_authority.as_ref(), active_prop_firm_period)
                {
                    let gate = neoethos_core::domain::risk::TradeGateInput {
                        balance: entry_balance,
                        equity: entry_equity,
                        confidence: measured_confidence,
                        current_time_sec: now_utc.timestamp().max(0) as u64,
                        current_hour: {
                            use chrono::Timelike;
                            now_utc.hour()
                        },
                        // `gate_and_size` replaces this with the exact durable
                        // pre-reservation count carried by `entry_reservation`.
                        entries_today: 0,
                        open_positions: open_positions_now,
                    };
                    let (decision, summary) = match authority.lock() {
                        Ok(mut guard) => {
                            let decision = guard.gate_and_size(
                                period,
                                gate,
                                base_risk,
                                entry_reservation.entries_before(),
                            );
                            (decision, guard.summary())
                        }
                        Err(_) => {
                            tracing::error!(
                                target: "neoethos_app::live_trading",
                                %symbol,
                                rule = "risk.account_authority_lock",
                                "entry refused — account-risk authority lock is poisoned"
                            );
                            entry_reservation.release();
                            continue;
                        }
                    };
                    let clamped = match decision {
                        Ok(clamped) => clamped,
                        Err(refusal) => {
                            tracing::warn!(
                                target: "neoethos_app::live_trading",
                                %symbol,
                                rule = refusal.rule,
                                detail = %refusal.detail,
                                balance = entry_balance,
                                equity = entry_equity,
                                entries_today = entry_reservation.entries_before(),
                                open_positions = gate.open_positions,
                                recovery_mode = summary.recovery_mode,
                                circuit_breaker_latched = summary.circuit_breaker_latched,
                                "ENTRY REFUSED BY THE ACCOUNT-WIDE PROP-FIRM RISK AUTHORITY (exits and trailing continue)"
                            );
                            if let Ok(mut s) = status.lock() {
                                s.last_signal = Some(format!(
                                    "REFUSED by {}: {}",
                                    refusal.rule, refusal.detail
                                ));
                            }
                            entry_reservation.release();
                            continue;
                        }
                    };
                    if clamped <= f64::EPSILON {
                        tracing::warn!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            intended_risk_pct = base_risk,
                            allowed_risk_pct = clamped,
                            "entry refused — prop-firm size clamp resolved to zero"
                        );
                        if let Ok(mut s) = status.lock() {
                            s.last_signal =
                                Some("REFUSED: prop-firm risk size is zero".to_string());
                        }
                        entry_reservation.release();
                        continue;
                    }
                    if clamped < base_risk {
                        tracing::warn!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            intended_risk_pct = base_risk,
                            effective_risk_pct = clamped,
                            per_trade_ceiling = summary.max_risk_per_trade,
                            recovery_mode = summary.recovery_mode,
                            "prop-firm entry risk clamped"
                        );
                    }
                    clamped
                } else {
                    base_risk
                };

                // Portfolio-level concurrent-risk budget (max_portfolio_risk):
                // remaining = cap − open_positions × base_risk. Skip the
                // entry when the budget is spent; size down when only part fits.
                //
                // The `if portfolio_risk_cap > 0.0` guard that used to wrap this
                // is DELETED (2026-08-10, audit #211): it made 0.0 mean "no
                // ceiling" on a knob named `max_`, which is the same disguise
                // `RiskConfig::default()` shipped. The cap is now always applied
                // literally — a cap of 0 permits no concurrent risk and every
                // entry is refused, which the ERROR at engine start names. "No
                // ceiling" is spelled 1.0.
                let open_n = open_positions_now as f64;
                let remaining = portfolio_risk_cap - open_n * base_risk;
                if remaining <= f64::EPSILON {
                    tracing::warn!(
                        target: "neoethos_app::live_trading",
                        %symbol, open_positions = open_n,
                        cap = portfolio_risk_cap,
                        "entry skipped — portfolio risk budget spent \
                         (max_portfolio_risk reached across open positions)"
                    );
                    if let Ok(mut s) = status.lock() {
                        s.last_signal = Some("blocked: portfolio risk budget spent".to_string());
                    }
                    // No entry happened — release the daily entry slot.
                    entry_reservation.release();
                    continue;
                }
                let effective_risk = base_risk.min(remaining);

                // Size by the account's risk %, using the EFFECTIVE stop
                // distance actually placed on the order (gene SL / override /
                // default). Missing inputs or a budget below the broker's
                // minimum skip this entry instead of using a fixed fallback.
                // Default to the strategy's OWN bracket; `req.*` is only an
                // explicit operator override (Autopilot sends none, so the
                // gene's discovered SL/TP is what actually gets placed).
                // PARITY: the discovery kernel NEVER runs bracket-free — a
                // gene without its own SL/TP is evaluated with the 20/40-pip
                // defaults (discovery.rs backtest-settings builder). Under
                // hold-to-bracket execution a naked position would never
                // close, so live mirrors the same defaults.
                let sl = req
                    .stop_loss_pips
                    .or((gene_sl > 0.0).then_some(gene_sl))
                    .or(Some(20.0));
                let tp = req
                    .take_profit_pips
                    .or((gene_tp > 0.0).then_some(gene_tp))
                    .or(Some(40.0));
                let entry_valuation = match entry_context.valuation(
                    entry_quote_session,
                    chrono::Utc::now().timestamp_millis(),
                    LIVE_ENTRY_SPOT_MAX_AGE_MS,
                ) {
                    Ok(valuation) => valuation,
                    Err(error) => {
                        tracing::warn!(target: "neoethos_app::live_trading", %symbol, %error,
                            "entry skipped: current quote/conversion unavailable");
                        if let Ok(mut s) = status.lock() {
                            s.last_signal = Some(format!("blocked: live conversion — {error}"));
                        }
                        entry_reservation.release();
                        continue;
                    }
                };
                let fx_quote_to_account = Some(entry_valuation.quote_to_account_rate);
                let last_price = Some(entry_valuation.price_for_risk);
                let lot = match risk_based_lots(
                    entry_balance,
                    effective_risk,
                    sl.unwrap_or(0.0),
                    Some(entry_context.metadata()),
                    &account_ccy,
                    fx_quote_to_account,
                    last_price,
                    max_lot_cap,
                ) {
                    Ok(lots) => lots,
                    Err(error) => {
                        tracing::warn!(target: "neoethos_app::live_trading", %symbol,
                            %error, effective_risk, entry_balance,
                            "entry skipped: cannot satisfy position-sizing budget");
                        if let Ok(mut s) = status.lock() {
                            s.last_signal = Some(format!("blocked: position sizing — {error}"));
                        }
                        entry_reservation.release();
                        continue;
                    }
                };

                // ── Risky Mode kill switch: THE PRE-SEND CHECK (W3) ──────────
                // The last thing before the order leaves the process. Checks
                // per-trade bracket validity, the pre-send sanity ceiling
                // (55% of bankroll), the per-day and per-week loss caps, the
                // per-stage retreat trigger and the per-month cap.
                //
                // `size_usd` must be what the manager expects — the money at
                // risk if the STOP fires — computed from the lot ACTUALLY being
                // sent, not from `effective_risk`. Those two diverge whenever
                // `max_lot_size`, the broker's min/max lot, the lot step or the
                // 30x affordability guard bind, and the pre-send ceiling exists
                // precisely to catch a size that came out wrong.
                if let Some(m) = risky_manager.as_mut() {
                    // Measure against the money that actually exists. The
                    // manager's own cursor only sees THIS engine's closed
                    // trades; the account also moves from manual orders, the
                    // other running engines, deposits and swap. Without this
                    // line an account that grew elsewhere would size at 50% of
                    // the NEW balance and be judged against a ceiling computed
                    // from the OLD one — a PreSendSanity refusal on a correctly
                    // sized order.
                    m.sync_bankroll(entry_balance);
                    // ACCOUNT-WIDE, RESTART-DURABLE LOSS LEDGER (2026-08-09).
                    // Raise this manager's day/week/month accumulators to the
                    // account's realized losses from the journal, so the caps
                    // bind on the account rather than on this one engine, and
                    // survive an app restart. See `account_period_losses`.
                    if let Some(dir) = journal_data_dir.as_ref() {
                        let trades = crate::app_services::journal_store::query_closed_trades(
                            dir, None, None,
                        );
                        let now_ms = crate::app_services::journal_store::now_unix_ms();
                        let (d_loss, w_loss, m_loss) =
                            account_period_losses(&trades, journal_account.as_deref(), now_ms);
                        let before = m.daily_loss_accumulated_usd();
                        m.raise_period_losses(d_loss, w_loss, m_loss);
                        if m.daily_loss_accumulated_usd() > before {
                            tracing::warn!(
                                target: "neoethos_app::live_trading",
                                %symbol,
                                rule = "risky_mode.account_wide_loss_ledger",
                                engine_daily_loss = before,
                                account_daily_loss = m.daily_loss_accumulated_usd(),
                                account_monthly_loss = m.monthly_loss_accumulated_usd(),
                                "this engine's own ledger under-counted the ACCOUNT's \
                                 realized loss (other engines / manual orders / a \
                                 restart); the day cap is now measured on the account"
                            );
                        }
                    }
                    let sl_pips = sl.unwrap_or(0.0);
                    let tp_pips = tp.unwrap_or(0.0);
                    let pip_val = sym_meta
                        .as_ref()
                        .map(|meta| {
                            meta.pip_value_in_account(&account_ccy, fx_quote_to_account, last_price)
                        })
                        .filter(|v| v.is_finite() && *v > 0.0);
                    let Some(pip_val) = pip_val else {
                        // Defence in depth: sizing already requires a priced
                        // stop. The independent pre-send check must not assume
                        // that a missing pip value means zero monetary risk.
                        tracing::error!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            rule = "risky_mode.presend_sanity",
                            account_ccy = %account_ccy,
                            has_symbol_metadata = sym_meta.is_some(),
                            fx_quote_to_account = ?fx_quote_to_account,
                            last_price = ?last_price,
                            "entry refused — Risky Mode cannot price this symbol's \
                             pip value in the account currency, so neither the \
                             held-out half-Kelly size nor the pre-send ceiling can be \
                             computed. Add the symbol to symbol_metadata.json or \
                             fix the quote->account FX rate."
                        );
                        if let Ok(mut s) = status.lock() {
                            s.last_signal = Some(
                                "blocked: risky-mode cannot price pip value for this symbol"
                                    .to_string(),
                            );
                        }
                        entry_reservation.release();
                        continue;
                    };
                    let size_at_risk = lot * sl_pips * pip_val;
                    if let Err(tier) = m.check_trade_allowed(size_at_risk, sl_pips, tp_pips) {
                        // Account-level tiers are a HALT: they say the bankroll
                        // itself is in trouble, so they start the persisted 24h
                        // cooldown and stop every Risky-Mode entry on this
                        // machine until it elapses. Order-level tiers refuse
                        // only THIS order — a malformed bracket or a mis-sized
                        // lot is not a reason to stop trading for a day, and
                        // pretending otherwise would train the operator to
                        // distrust the halt.
                        let halts_for_24h = tier_halts_for_24h(tier);
                        // A PreSendSanity refusal has two very different
                        // causes and the operator must be able to tell them
                        // apart. If the lot is already at the broker's MINIMUM
                        // and still breaches the ceiling, the account is simply
                        // too small to trade this symbol inside the ceiling —
                        // that refusal repeats on every bar, permanently, and
                        // reads as a silent engine unless it is named.
                        let broker_min_lot = sym_meta
                            .as_ref()
                            .map(|meta| meta.min_lot)
                            .filter(|v| v.is_finite() && *v > 0.0);
                        let at_broker_min_lot =
                            broker_min_lot.is_some_and(|min| lot <= min * 1.000_001);
                        if tier == neoethos_core::domain::risky_mode::KillSwitchTier::PreSendSanity
                            && at_broker_min_lot
                        {
                            tracing::error!(
                                target: "neoethos_app::live_trading",
                                %symbol,
                                rule = "risky_mode.presend_sanity.account_too_small",
                                broker_min_lot = ?broker_min_lot,
                                lot,
                                sl_pips,
                                size_at_risk,
                                bankroll = m.current_bankroll_usd(),
                                ceiling = m.current_bankroll_usd()
                                    * m.config().presend_sanity_ceiling_fraction,
                                "this account is TOO SMALL to trade {symbol} inside the \
                                 pre-send ceiling: the broker's minimum lot already puts \
                                 more at risk than the ceiling allows. This refusal will \
                                 repeat on EVERY bar for this symbol until the balance \
                                 grows or the stop distance shrinks — it is not transient."
                            );
                        }
                        tracing::error!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            rule = "risky_mode.check_trade_allowed",
                            tier = ?tier,
                            halts_for_24h,
                            size_at_risk,
                            lot,
                            sl_pips,
                            tp_pips,
                            bankroll = m.current_bankroll_usd(),
                            presend_ceiling = m.current_bankroll_usd()
                                * m.config().presend_sanity_ceiling_fraction,
                            daily_loss = m.daily_loss_accumulated_usd(),
                            daily_cap = m.current_stage().daily_loss_cap_fraction
                                * m.current_bankroll_usd(),
                            weekly_loss = m.weekly_loss_accumulated_usd(),
                            weekly_cap = m.current_stage().weekly_drawdown_cap_fraction
                                * m.current_bankroll_usd(),
                            monthly_loss = m.monthly_loss_accumulated_usd(),
                            stage_idx = m.current_stage().stage_idx,
                            "ENTRY REFUSED BY THE RISKY MODE KILL SWITCH"
                        );
                        if halts_for_24h {
                            // Persist it, so the halt survives a restart and the
                            // Risk screen's cooldown row finally means something.
                            if let Err(e) =
                                crate::app_services::risky_mode_persistence::record_kill_switch_trip(
                                )
                            {
                                // The refusal already happened above; this only
                                // failed to make it durable. Say so loudly —
                                // the operator must know the 24h halt will not
                                // survive a restart.
                                tracing::error!(
                                    target: "neoethos_app::live_trading",
                                    %symbol, error = %e,
                                    "kill switch tripped but the 24h cooldown could NOT be \
                                     persisted — this entry is still refused, but the halt \
                                     will not survive an app restart"
                                );
                            }
                        }
                        if let Ok(mut s) = status.lock() {
                            s.last_signal = Some(format!(
                                "REFUSED by risky-mode kill switch: {tier:?}{}",
                                if halts_for_24h { " (24h halt)" } else { "" }
                            ));
                        }
                        entry_reservation.release();
                        continue;
                    }
                }

                // A reservation taken just before midnight must not authorize
                // an order in the next accounting day. Re-resolve the boundary
                // immediately before broker submission; on rollover, release
                // the old slot and let the next bar rebuild/gate new-day state.
                let current_entry_day = if trading_mode_risky {
                    Some(utc_day_id(chrono::Utc::now()))
                } else {
                    prop_firm_period(resolved_settings.risk.preset, chrono::Utc::now())
                        .ok()
                        .map(|period| period.day_id)
                };
                let reservation_period_is_current =
                    current_entry_day == Some(entry_reservation.day_id);
                if !reservation_period_is_current {
                    tracing::warn!(
                        target: "neoethos_app::live_trading",
                        %symbol,
                        rule = "risk.entry_period_rolled_before_send",
                        reserved_accounting_day = entry_reservation.day_id,
                        current_accounting_day = ?current_entry_day,
                        "entry skipped — the accounting day rolled after reservation and before broker submission"
                    );
                    entry_reservation.release();
                    continue;
                }

                let entry_money_digits = entry_context.trader_money_digits();
                let entry_client_order_id = entry_context.client_order_id().to_owned();
                let submission_spread_policy = live_trading_policy.clone();
                let submission_account_currency = account_ccy.clone();
                let submission_may_have_started = Arc::new(AtomicBool::new(false));
                let submission_marker = submission_may_have_started.clone();
                let submission_entry_authority = account_entry_authority.clone();
                let submission_day_id = entry_reservation.day_id;
                let result = match tokio::task::spawn_blocking(move || {
                    entry_context.submit_market_order_blocking(
                        entry_quote_session, side, lot, sl, tp,
                        Some("NeoEthos-Auto".to_owned()), LIVE_ENTRY_SPOT_MAX_AGE_MS,
                        &submission_marker,
                        |request| {
                            submission_entry_authority.lock()
                                .map_err(|_| anyhow!("account-entry authority lock is poisoned"))?
                                .begin_submission(submission_day_id, request, &submission_marker)
                        },
                        |metadata, valuation| {
                            let now = chrono::Utc::now();
                            anyhow::ensure!(
                                utc_day_id(now) == today,
                                "UTC day changed during entry preparation; reserve a new decision"
                            );
                            evaluate_live_entry_spread(
                                Ok((valuation.bid, valuation.ask)), metadata.pip_size,
                                submission_spread_policy.expected_spread_pips_at(now.timestamp_millis()),
                            ).map_err(|reason| anyhow!("entry quote/spread refused before submission: {reason:?}"))?;
                            let current_max_lots = risk_based_lots(
                                entry_balance, effective_risk, sl.unwrap_or(0.0),
                                Some(metadata), &submission_account_currency,
                                Some(valuation.quote_to_account_rate), Some(valuation.price_for_risk),
                                max_lot_cap,
                            )?;
                            anyhow::ensure!(
                                lot <= current_max_lots,
                                "live quote/conversion changed the permitted size; refuse rather than increase the budget"
                            );
                            Ok(())
                        },
                    )
                })
                .await
                {
                    Ok(r) => r,
                    Err(join_err) => {
                        if submission_may_have_started.load(Ordering::Acquire) {
                            has_unresolved_broker_entry = true;
                            unresolved_entry_client_order_id = Some(entry_client_order_id.clone());
                            tracing::error!(target: "neoethos_app::live_trading", %symbol,
                                error = %join_err, client_order_id = %entry_client_order_id,
                                "entry submission outcome unknown; keeping reserved slot and blocking new entries until broker reconciliation");
                            if let Ok(mut s) = status.lock() {
                                s.last_signal = Some("blocked: unresolved broker entry after submit task failure".to_owned());
                            }
                            continue;
                        }
                        // The completed task never reached the executor; release only its exact local intent.
                        entry_reservation.release_unsent(&entry_client_order_id, &submission_may_have_started);
                        return Err(join_err.into());
                    }
                };

                match result {
                    Ok(outcome) => {
                        let (opening, volume_scale, actual_filled_lots) =
                            match verified_opening_for_engine(
                                &outcome,
                                if gated_env_is_live { "live" } else { "demo" },
                                gated_account_id,
                                gated_symbol_id,
                                &symbol,
                                side,
                            ) {
                                Ok(verified) => verified,
                                Err(error) => {
                                    // Accepted, partial, reducing or incomplete responses may
                                    // already have changed broker exposure. Retain the reserved
                                    // slot and original intent; do not invent an owned position.
                                    has_unresolved_broker_entry = true;
                                    unresolved_entry_client_order_id =
                                        Some(entry_client_order_id.clone());
                                    let recorded = account_entry_authority.lock()
                                        .map_err(|_| anyhow!("account-entry authority lock is poisoned"))
                                        .and_then(|mut state| state.record_unresolved_outcome(
                                            &entry_client_order_id,
                                            if gated_env_is_live {
                                                crate::app_services::ctrader_live_auth::CTraderEnvironment::Live
                                            } else {
                                                crate::app_services::ctrader_live_auth::CTraderEnvironment::Demo
                                            },
                                            &outcome,
                                        ));
                                    if let Err(record_error) = recorded {
                                        tracing::error!(target: "neoethos_app::live_trading",
                                            client_order_id = %entry_client_order_id, %record_error,
                                            "known order reference could not be persisted; original entry intent remains unresolved");
                                    }
                                    tracing::error!(
                                        target: "neoethos_app::live_trading",
                                        %symbol, %error,
                                        client_order_id = %entry_client_order_id,
                                        execution_status = ?outcome.status,
                                        order_id = ?outcome.order_id,
                                        position_id = ?outcome.position_id,
                                        deal_id = ?outcome.deal_id,
                                        "entry response lacks exact opening proof; retaining reservation and unresolved intent, no automatic retry"
                                    );
                                    if let Ok(mut s) = status.lock() {
                                        s.last_signal = Some(format!(
                                            "blocked: unresolved broker entry {entry_client_order_id}; opening reconciliation required"
                                        ));
                                        s.protection_state = Some("entry_unresolved".to_owned());
                                        s.last_protection_error = Some(error.to_string());
                                    }
                                    continue;
                                }
                            };
                        let pos_id = opening.position_id();
                        let broker_volume = opening.filled_volume_raw_centi_units();
                        opened_ids.insert(pos_id);
                        open_position = Some((pos_id, broker_volume));
                        opened_entry_filled_volumes.insert(pos_id, broker_volume);
                        opened_volume_scales.insert(pos_id, volume_scale.clone());
                        // Preserve the exact reference for later recovery, but only
                        // this unchanged opening proof can clear durable uncertainty.
                        let confirmed = account_entry_authority
                            .lock()
                            .map_err(|_| anyhow!("account-entry authority lock is poisoned"))
                            .and_then(|mut state| {
                                state.confirm_verified_opening(&entry_client_order_id, opening)
                            });
                        if let Err(error) = confirmed {
                            has_unresolved_broker_entry = true;
                            unresolved_entry_client_order_id = Some(entry_client_order_id.clone());
                            tracing::error!(target: "neoethos_app::live_trading",
                                client_order_id = %entry_client_order_id, %error,
                                "opening is tracked but durable intent could not be resolved; further entries remain blocked");
                        }

                        // Seed the archived protection using actual opening execution,
                        // never a historical/current quote substituted for broker fill.
                        pos_entry_px = opening.entry_price();
                        pos_sl_pips = sl.unwrap_or(0.0);
                        pos_is_long = opening.trade_side() == "BUY";
                        pos_extreme = pos_entry_px;
                        pos_trail_px = 0.0;
                        let feature_row = experience_feature_row
                            .context("entry has no retained Search experience row")
                            .and_then(|row| row);
                        match feature_row {
                            Ok(features) => {
                                pending_experience.insert(
                                    pos_id,
                                    crate::app_services::experience_store::LiveExperience {
                                        schema_version: 1,
                                        position_id: pos_id,
                                        symbol: symbol.clone(),
                                        base_tf: base_tf.clone(),
                                        portfolio_path: portfolio_path.clone(),
                                        direction: if pos_is_long { 1 } else { -1 },
                                        sl_pips: sl.unwrap_or(0.0),
                                        tp_pips: tp.unwrap_or(0.0),
                                        lots: actual_filled_lots,
                                        entry_ts_ms: opening.execution_timestamp_ms(),
                                        entry_price: Some(opening.entry_price()),
                                        features,
                                        close_ts_ms: None,
                                        net_profit: None,
                                    },
                                );
                            }
                            Err(error) => tracing::warn!(
                                target: "neoethos_app::live_trading",
                                %symbol, position_id = pos_id, error = %error,
                                "opened position but refused to persist a non-exact feature experience row"
                            ),
                        }

                        if let Ok(mut s) = status.lock() {
                            s.open_position_id = Some(pos_id);
                            s.position_entry_price = Some(pos_entry_px);
                            s.initial_stop_pips = (pos_sl_pips > 0.0).then_some(pos_sl_pips);
                            s.favorable_extreme_price = Some(pos_extreme);
                            s.favorable_move_r = Some(0.0);
                            s.confirmed_stop_price = None;
                            s.last_protection_error = None;
                            s.last_exit_reason = None;
                            s.protection_state = Some(
                                if live_trading_policy.trailing_enabled {
                                    "armed_waiting_trigger"
                                } else {
                                    "disabled_by_search"
                                }
                                .to_owned(),
                            );
                        }

                        tracing::info!(
                            target: "neoethos_app::live_trading",
                            side = ?side, position_id = pos_id,
                            deal_id = opening.deal_id(),
                            client_order_id = %entry_client_order_id,
                            fill_price = opening.entry_price(),
                            fill_timestamp_ms = opening.execution_timestamp_ms(),
                            trader_money_digits = entry_money_digits,
                            "exact broker opening fill is tracked; durable lifecycle integration remains pending"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "neoethos_app::live_trading",
                            error = %e,
                            side = ?side,
                            "order submission returned an error"
                        );
                        if submission_may_have_started.load(Ordering::Acquire) {
                            has_unresolved_broker_entry = true;
                            unresolved_entry_client_order_id = Some(entry_client_order_id.clone());
                            let recorded = account_entry_authority
                                .lock()
                                .map_err(|_| anyhow!("account-entry authority lock is poisoned"))
                                .and_then(|mut state| {
                                    state.record_unresolved_error(&entry_client_order_id, &e)
                                });
                            if let Err(error) = recorded {
                                tracing::error!(target: "neoethos_app::live_trading",
                                    client_order_id = %entry_client_order_id, %error,
                                    "accepted order context could not be persisted; original durable intent remains blocked");
                            }
                            tracing::error!(target: "neoethos_app::live_trading", %symbol,
                                error = %e, client_order_id = %entry_client_order_id,
                                "entry submission outcome unknown; retaining reserved slot, no automatic retry");
                            if let Ok(mut s) = status.lock() {
                                s.last_signal = Some(
                                    "blocked: unresolved broker entry requires reconciliation"
                                        .to_owned(),
                                );
                            }
                        } else {
                            // Local preparation/freshness/budget refusal: completed attempt is definitely unsent.
                            entry_reservation.release_unsent(
                                &entry_client_order_id,
                                &submission_may_have_started,
                            );
                        }
                    }
                }
            }

            Direction::Flat => {
                // PARITY: the backtest does NOT exit on a flat signal — an
                // open position runs to its SL/TP/trailing bracket (or the
                // weekend force-close). Nothing to execute.
            }
        }
    }

    // Mark stopped
    if let Ok(mut s) = status.lock() {
        s.running = false;
        // Stopping this engine does not close a broker position. Preserve that
        // fact so the UI cannot display "flat" while a server-side bracket is
        // still active and needs operator supervision.
        s.open_position_id = open_position
            .map(|(id, _)| id)
            .or_else(|| opened_ids.iter().copied().next());
    }

    tracing::info!(target: "neoethos_app::live_trading", "live trading loop exited");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn gene_confidence_is_not_replaced_by_ml_or_applied_twice_to_mode_risk() {
        for mode_risk in [0.01, 0.035] {
            assert_eq!(
                super::live_entry_sizing_inputs(mode_risk, 0.8, None).unwrap(),
                (mode_risk, Some(0.8))
            );
            assert_eq!(
                super::live_entry_sizing_inputs(mode_risk, 0.8, Some(0.25)).unwrap(),
                (mode_risk * 0.25, Some(0.8))
            );
        }
        assert!(super::live_entry_sizing_inputs(0.01, f64::NAN, Some(0.5)).is_err());
        assert!(super::live_entry_sizing_inputs(0.01, 0.8, Some(1.1)).is_err());
    }

    use super::*;
    use crate::app_services::journal_store::ClosedTrade;
    use neoethos_core::domain::risky_mode as rm;

    #[test]
    fn live_entry_spread_preserves_every_quote_refusal_without_a_fallback() {
        use SpotQuoteRefusal::*;
        for reason in [
            CacheUnavailable,
            InvalidTimeBudget,
            NoActiveSession,
            SessionMismatch,
            MissingQuote,
            MissingBid,
            MissingAsk,
            MissingBidTimestamp,
            MissingAskTimestamp,
            InvalidBidTimestamp,
            InvalidAskTimestamp,
            StaleBid,
            StaleAsk,
            FutureBid,
            FutureAsk,
            InvalidPrices,
        ] {
            assert_eq!(
                evaluate_live_entry_spread(Err(reason), 0.0001, 1.5),
                Err(LiveEntrySpreadRefusal::Quote(reason))
            );
        }
        assert_eq!(LIVE_ENTRY_SPOT_MAX_AGE_MS, 120_000);
    }

    #[test]
    fn live_entry_spread_preserves_exact_threshold_equality_and_zero_spread() {
        assert_eq!(
            evaluate_live_entry_spread(Ok((100.0, 105.0)), 1.0, 2.0),
            Ok(5.0)
        );
        assert_eq!(
            evaluate_live_entry_spread(Ok((100.0, 106.0)), 1.0, 2.0),
            Err(LiveEntrySpreadRefusal::ExceedsLimit {
                spread_pips: 6.0,
                limit_pips: 5.0,
            })
        );
        assert_eq!(
            evaluate_live_entry_spread(Ok((100.0, 100.0)), 1.0, 0.0),
            Ok(0.0)
        );
    }

    #[test]
    fn live_entry_spread_refuses_invalid_prices_units_policy_and_overflow() {
        for pip in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                evaluate_live_entry_spread(Ok((1.0, 2.0)), pip, 1.0),
                Err(LiveEntrySpreadRefusal::InvalidPipSize)
            );
        }
        for expected in [-1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            assert_eq!(
                evaluate_live_entry_spread(Ok((1.0, 2.0)), 1.0, expected),
                Err(LiveEntrySpreadRefusal::InvalidExpectedSpread)
            );
        }
        for quote in [
            (0.0, 1.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (2.0, 1.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::INFINITY, f64::INFINITY),
        ] {
            assert_eq!(
                evaluate_live_entry_spread(Ok(quote), 1.0, 1.0),
                Err(LiveEntrySpreadRefusal::InvalidPrices)
            );
        }
        assert_eq!(
            evaluate_live_entry_spread(Ok((1.0, f64::MAX)), f64::MIN_POSITIVE, 1.0),
            Err(LiveEntrySpreadRefusal::InvalidSpread)
        );
    }

    fn sizing_metadata() -> neoethos_core::symbol_metadata::SymbolMetadata {
        serde_json::from_value(serde_json::json!({
            "symbol": "EURUSD", "base": "EUR", "quote": "USD",
            "pip_size": 0.0001, "contract_size": 100000.0, "pip_value_quote": 10.0,
            "digits": 5, "min_lot": 0.01, "max_lot": 100.0, "lot_step": 0.01
        }))
        .unwrap()
    }

    #[test]
    fn required_live_ml_failure_does_not_return_a_gene_only_multiplier() {
        let config = neoethos_trader::BlendConfig::from_config_values(
            neoethos_trader::BlendMode::MlScale,
            None,
            None,
        );
        let error = checked_live_ml_entry(
            Direction::Long,
            Err(anyhow!("CPU permits unavailable")),
            &config,
        )
        .expect_err("a busy or failed inference cannot authorize full-size genes-only entry");
        assert!(format!("{error:#}").contains("CPU permits unavailable"));
    }

    #[test]
    fn entry_sizing_requires_current_account_identity_and_finite_equity_in_both_environments() {
        for environment in ["demo", "live"] {
            assert!(
                validate_entry_account_values(environment, 42, 1000.0, 900.0, environment, 42)
                    .is_ok()
            );
            assert!(
                validate_entry_account_values(environment, 99, 1000.0, 900.0, environment, 42)
                    .is_err()
            );
            let foreign = if environment == "demo" {
                "live"
            } else {
                "demo"
            };
            assert!(
                validate_entry_account_values(foreign, 42, 1000.0, 900.0, environment, 42).is_err()
            );
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
                assert!(
                    validate_entry_account_values(environment, 42, invalid, 900.0, environment, 42)
                        .is_err()
                );
                assert!(
                    validate_entry_account_values(
                        environment,
                        42,
                        1000.0,
                        invalid,
                        environment,
                        42
                    )
                    .is_err()
                );
            }
        }
    }

    fn live_test_model_decision() -> neoethos_models::ensemble_inference::EnsembleDecision {
        neoethos_models::ensemble_inference::EnsembleDecision {
            dir_probs: [0.05, 0.9, 0.05],
            regime_gate: 1.0,
            anomaly_scale: 1.0,
            validity: neoethos_data::FeatureCellValidity::Valid,
        }
    }

    #[test]
    fn valid_live_ml_keeps_the_gene_direction_and_scales_once() {
        let config = neoethos_trader::BlendConfig::from_config_values(
            neoethos_trader::BlendMode::MlScale,
            None,
            None,
        );
        let decision = live_test_model_decision();
        let (observed, multiplier) =
            checked_live_ml_entry(Direction::Long, Ok(decision), &config).unwrap();
        assert_eq!(observed, decision);
        assert!((multiplier - 0.9).abs() < 1e-12);
        assert!((0.03 * multiplier - 0.027).abs() < 1e-12);
        assert!(checked_live_ml_entry(Direction::Flat, Ok(decision), &config).is_err());
    }

    #[test]
    fn invalid_live_ml_rows_are_refused_even_with_finite_payloads() {
        let config = neoethos_trader::BlendConfig::from_config_values(
            neoethos_trader::BlendMode::MlScale,
            None,
            None,
        );
        for validity in [
            neoethos_data::FeatureCellValidity::Warmup,
            neoethos_data::FeatureCellValidity::AlignmentMissing,
        ] {
            let mut decision = live_test_model_decision();
            decision.validity = validity;
            let error = checked_live_ml_entry(Direction::Long, Ok(decision), &config)
                .expect_err("typed invalidity is not an ordinary neutral vote");
            assert!(error.to_string().contains("ineligible"));
        }
    }

    #[test]
    fn nonfinite_or_zero_live_ml_cannot_authorize_an_entry() {
        let config = neoethos_trader::BlendConfig::from_config_values(
            neoethos_trader::BlendMode::MlScale,
            Some(0.0),
            Some(0.0),
        );
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for field in 0..5 {
                let mut decision = live_test_model_decision();
                match field {
                    0..=2 => decision.dir_probs[field] = value,
                    3 => decision.regime_gate = value,
                    _ => decision.anomaly_scale = value,
                }
                assert!(checked_live_ml_entry(Direction::Long, Ok(decision), &config).is_err());
            }
        }
        let mut veto = live_test_model_decision();
        veto.anomaly_scale = 0.0;
        assert!(checked_live_ml_entry(Direction::Long, Ok(veto), &config).is_err());
    }

    #[test]
    fn risk_sizing_never_rounds_a_small_budget_up_to_broker_minimum() {
        let meta = sizing_metadata();
        // $0.10 stop budget; the minimum 0.01 lot would lose $2 at 20 pips.
        let error = risk_based_lots(10.0, 0.01, 20.0, Some(&meta), "USD", None, Some(1.1), 100.0)
            .unwrap_err();
        assert!(error.to_string().contains("below broker minimum"));
    }

    #[test]
    fn risk_sizing_rounds_down_and_respects_non_step_aligned_caps() {
        let meta = sizing_metadata();
        let lots = risk_based_lots(
            10_000.0,
            0.01,
            20.0,
            Some(&meta),
            "USD",
            None,
            Some(1.1),
            0.255,
        )
        .unwrap();
        assert!((lots - 0.25).abs() < 1e-12);
        assert!(lots * 20.0 * 10.0 <= 100.0);
        assert!(
            risk_based_lots(
                10_000.0,
                0.01,
                20.0,
                Some(&meta),
                "USD",
                None,
                Some(1.1),
                0.005
            )
            .is_err(),
            "an operator cap below minimum is not raised"
        );
    }

    #[test]
    fn risk_sizing_uses_account_currency_and_refuses_missing_conversion() {
        let meta = sizing_metadata();
        let lots = risk_based_lots(
            10_000.0,
            0.01,
            20.0,
            Some(&meta),
            "GBP",
            Some(0.8),
            Some(1.1),
            100.0,
        )
        .unwrap();
        assert!((lots - 0.62).abs() < 1e-12, "100 / (20 * 8) rounded down");
        assert!(
            risk_based_lots(
                10_000.0,
                0.01,
                20.0,
                Some(&meta),
                "GBP",
                None,
                Some(1.1),
                100.0
            )
            .is_err()
        );
    }

    #[test]
    fn base_account_notional_uses_inverse_price_not_an_assumed_fx_one() {
        let mut meta = sizing_metadata();
        meta.symbol = "USDJPY".into();
        meta.base = "USD".into();
        meta.quote = "JPY".into();
        meta.pip_size = 0.01;
        meta.pip_value_quote = 1000.0;
        let lots = risk_based_lots(
            10_000.0,
            0.25,
            5.0,
            Some(&meta),
            "USD",
            None,
            Some(150.0),
            100.0,
        )
        .unwrap();
        // 30 * $10,000 / $100,000 per lot = 3, not 0.02 (the old JPY-as-USD bug).
        assert!((lots - 3.0).abs() < 1e-12);
    }

    #[test]
    fn risk_sizing_rejects_non_finite_inputs_instead_of_a_fixed_lot_fallback() {
        let mut meta = sizing_metadata();
        for invalid in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            assert!(
                risk_based_lots(
                    invalid,
                    0.01,
                    20.0,
                    Some(&meta),
                    "USD",
                    None,
                    Some(1.1),
                    100.0
                )
                .is_err()
            );
            assert!(
                risk_based_lots(
                    10_000.0,
                    invalid,
                    20.0,
                    Some(&meta),
                    "USD",
                    None,
                    Some(1.1),
                    100.0
                )
                .is_err()
            );
        }
        meta.lot_step = 0.0;
        assert!(
            risk_based_lots(
                10_000.0,
                0.01,
                20.0,
                Some(&meta),
                "USD",
                None,
                Some(1.1),
                100.0
            )
            .is_err()
        );
        assert!(
            risk_based_lots(10_000.0, 0.01, 20.0, None, "USD", None, Some(1.1), 100.0).is_err()
        );
    }

    #[test]
    fn favorable_excursion_is_measured_in_the_positions_own_initial_risk() {
        let long = favorable_excursion_r(1.1000, 1.1060, 20.0, 0.0001, true)
            .expect("valid long excursion");
        let short = favorable_excursion_r(1.1000, 1.0940, 20.0, 0.0001, false)
            .expect("valid short excursion");
        assert!(
            (long - 3.0).abs() < 1e-12,
            "long MFE should be 3R, got {long}"
        );
        assert!(
            (short - 3.0).abs() < 1e-12,
            "short MFE should be 3R, got {short}"
        );
        assert_eq!(
            favorable_excursion_r(1.1000, 1.1060, 0.0, 0.0001, true),
            None,
            "a position with no proved initial risk has no meaningful R multiple"
        );
    }

    #[test]
    fn live_status_serializes_profit_protection_evidence() {
        let mut status = LiveTradingStatus::default();
        status.protection_policy_identity = Some("fnv64:0123456789abcdef".to_string());
        status.trailing_enabled = Some(true);
        status.protection_state = Some("broker_confirmed".to_string());
        status.position_entry_price = Some(1.1);
        status.initial_stop_pips = Some(20.0);
        status.favorable_extreme_price = Some(1.106);
        status.favorable_move_r = Some(3.0);
        status.confirmed_stop_price = Some(1.104);
        status.last_exit_reason = Some("broker_reported_close_unclassified".to_string());

        let value = serde_json::to_value(status).expect("serialize live status");
        assert_eq!(value["protectionState"], "broker_confirmed");
        assert_eq!(value["favorableMoveR"], 3.0);
        assert_eq!(value["confirmedStopPrice"], 1.104);
        assert_eq!(value["protectionPolicyIdentity"], "fnv64:0123456789abcdef");
    }

    fn protection_outcome(
        status: CTraderExecutionStatus,
        position_id: i64,
    ) -> CTraderExecutionOutcome {
        CTraderExecutionOutcome {
            status,
            account_id: 7,
            symbol_id: Some(14),
            order_id: None,
            position_id: Some(position_id),
            deal_id: None,
            trade_side: Some("BUY".to_string()),
            order_type: None,
            lot_size: Some(0.01),
            requested_lot_size: Some(0.01),
            filled_lot_size: None,
            filled_volume_raw_centi_units: None,
            volume_scale_evidence: None,
            deal_closes_position: None,
            opening_fill_evidence: None,
            execution_price: Some(1.10),
            gross_profit: None,
            fee: None,
            swap: None,
            net_profit: None,
            timestamp_ms: Some(1_710_000_000_000),
            error_code: None,
            description: None,
        }
    }

    #[test]
    fn local_trail_changes_only_after_exact_broker_confirmation() {
        let mut confirmed = 1.095;

        let accepted = protection_outcome(CTraderExecutionStatus::Accepted, 42);
        assert!(commit_broker_confirmed_trail(&mut confirmed, 1.101, 42, &accepted).is_err());
        assert_eq!(
            confirmed, 1.095,
            "transport success is not broker confirmation"
        );

        let wrong_position = protection_outcome(CTraderExecutionStatus::Replaced, 99);
        assert!(commit_broker_confirmed_trail(&mut confirmed, 1.101, 42, &wrong_position).is_err());
        assert_eq!(
            confirmed, 1.095,
            "another position's amend must not advance this one"
        );

        let replaced = protection_outcome(CTraderExecutionStatus::Replaced, 42);
        commit_broker_confirmed_trail(&mut confirmed, 1.101, 42, &replaced)
            .expect("matching ORDER_REPLACED is an exact confirmation");
        assert_eq!(confirmed, 1.101);
    }

    #[test]
    fn partial_close_reconciliation_updates_exact_remaining_volume_not_flat_state() {
        let mut tracked = (42, 100_000);
        refresh_tracked_position_volume(&mut tracked, 14, [(99, 14, 20_000), (42, 14, 37_001)])
            .unwrap();
        assert_eq!(tracked, (42, 37_001));
        refresh_tracked_position_volume(&mut tracked, 14, []).unwrap();
        assert_eq!(
            tracked,
            (42, 37_001),
            "absence alone must not bypass verified closing-fill accounting"
        );
    }

    #[test]
    fn invalid_remaining_volume_or_identity_preserves_tracked_position() {
        for positions in [
            vec![(42, 15, 10_000)],
            vec![(42, 14, 0)],
            vec![(42, 14, -1)],
            vec![(42, 14, 10_000), (42, 14, 10_000)],
        ] {
            let mut tracked = (42, 100_000);
            assert!(refresh_tracked_position_volume(&mut tracked, 14, positions).is_err());
            assert_eq!(tracked, (42, 100_000));
        }
    }

    /// The operator's `models.blend_*` numbers REACH the live blend (audit
    /// #232).
    ///
    /// These two multipliers scale every entry's risk and were struct literals
    /// in this file with no config recipient. The regression this pins is the
    /// half-wired shape they were in before: readers that compiled, were
    /// called, and returned `None` — so the run used the shipped default no
    /// matter what the operator typed, and looked identical either way.
    #[test]
    fn the_operator_blend_multipliers_reach_the_live_blend_config() {
        let mut settings = neoethos_core::Settings::default();
        settings.models.blend_gate_floor = 0.61;
        settings.models.blend_veto_below = 0.11;

        assert_eq!(operator_blend_gate_floor(Some(&settings)), Some(0.61));
        assert_eq!(operator_blend_veto_below(Some(&settings)), Some(0.11));

        let cfg = neoethos_trader::BlendConfig::from_config_values(
            neoethos_trader::BlendMode::MlScale,
            operator_blend_gate_floor(Some(&settings)),
            operator_blend_veto_below(Some(&settings)),
        );
        assert!(
            (cfg.gate_floor - 0.61).abs() < 1e-9 && (cfg.veto_below - 0.11).abs() < 1e-9,
            "the configured pair must be the pair the live blend uses, got {} / {}",
            cfg.gate_floor,
            cfg.veto_below
        );

        // No settings at all is still the shipped default, not a panic.
        let shipped = neoethos_trader::BlendConfig::from_config_values(
            neoethos_trader::BlendMode::MlScale,
            operator_blend_gate_floor(None),
            operator_blend_veto_below(None),
        );
        let default = neoethos_trader::BlendConfig::default();
        assert!((shipped.gate_floor - default.gate_floor).abs() < 1e-9);
        assert!((shipped.veto_below - default.veto_below).abs() < 1e-9);

        // And an inverted pair is refused back to the shipped defaults rather
        // than vetoing every floored bar.
        settings.models.blend_gate_floor = 0.10;
        settings.models.blend_veto_below = 0.90;
        let refused = neoethos_trader::BlendConfig::from_config_values(
            neoethos_trader::BlendMode::MlScale,
            operator_blend_gate_floor(Some(&settings)),
            operator_blend_veto_below(Some(&settings)),
        );
        assert!((refused.gate_floor - default.gate_floor).abs() < 1e-9);
        assert!((refused.veto_below - default.veto_below).abs() < 1e-9);
    }

    #[test]
    fn account_level_tiers_halt_for_24h_and_order_level_tiers_do_not() {
        use rm::KillSwitchTier as T;
        for tier in [T::PerDay, T::PerWeek, T::PerStage, T::PerMonth] {
            assert!(
                tier_halts_for_24h(tier),
                "{tier:?} is a bankroll-level event and must start the persisted halt"
            );
        }
        for tier in [T::PerTrade, T::PreSendSanity] {
            assert!(
                !tier_halts_for_24h(tier),
                "{tier:?} describes one order — refuse it, do not halt the account for a day"
            );
        }
    }

    /// The exact manager `run` builds when `system.trading_mode == "risky"`,
    /// from the operator's shipped `system.risky_start_balance_usd: 100.0` /
    /// `risky_target_balance_usd: 50000.0`.
    fn operator_manager(bankroll: f64) -> rm::RiskyModeManager {
        let cfg = rm::RiskyModeConfig {
            starting_capital_usd: 100.0,
            target_capital_usd: 50_000.0,
            stage_doubling_factor: rm::DEFAULT_DOUBLING_FACTOR,
            stages: rm::build_logarithmic_stages(100.0, 50_000.0, rm::DEFAULT_DOUBLING_FACTOR),
            ..rm::RiskyModeConfig::default()
        };
        rm::RiskyModeManager::new(cfg, bankroll).expect("the shipped risky settings must build")
    }

    #[test]
    fn the_operators_shipped_risky_settings_build_a_manager() {
        // If this ever fails, `run` now REFUSES TO START in risky mode — which
        // is the intended fail-closed behaviour, but it must not happen by
        // accident on the config the operator actually ships.
        let m = operator_manager(100.0);
        assert_eq!(m.current_stage().stage_idx, 0);
        assert!(
            (m.current_stage().risk_per_trade_fraction
                - rm::RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION)
                .abs()
                < 1e-9,
            "the shipped ladder starts at the default risk, not the absolute safety ceiling"
        );
    }

    #[test]
    fn the_gate_and_the_sizer_read_the_same_rung_of_the_same_ladder() {
        // This is the property that makes the wiring coherent, and it is worth
        // pinning: the entry SIZE comes from the free function
        // `stage_risk_fraction_for_bankroll(start, target, doubling, balance)`,
        // while the GATE comes from a stateful `RiskyModeManager`. They are
        // only meaningful together if both land on the same stage. They do,
        // because the manager is built from the same three inputs and
        // `sync_bankroll` relocates it by the same `locate_stage_idx`.
        //
        // NOTE (#209/#210, 2026-08-09): `run` now CLAMPS the sizer's output to
        // `risk.risky_max_risk_per_trade` when that is lower. That clamp lives
        // at the call site, not in the ladder, so this rung-identity property
        // is unchanged — the clamp can only make the entry smaller than the
        // rung the gate is judging, never larger, which is the safe direction.
        for balance in [100.0, 199.0, 200.0, 1_600.0, 25_000.0, 80_000.0] {
            let mut m = operator_manager(100.0);
            m.sync_bankroll(balance);
            let sizer = rm::stage_risk_fraction_for_bankroll(
                100.0,
                50_000.0,
                rm::DEFAULT_DOUBLING_FACTOR,
                balance,
            )
            .expect("the shipped ladder resolves");
            assert!(
                (m.current_stage().risk_per_trade_fraction - sizer).abs() < 1e-12,
                "balance {balance}: gate rung {} != sizer rung {sizer}",
                m.current_stage().risk_per_trade_fraction
            );
        }
    }

    #[test]
    fn a_normally_sized_risky_entry_is_allowed() {
        // 100 balance, 50% stage ⇒ ~50 at risk, under the 55% pre-send ceiling.
        let m = operator_manager(100.0);
        assert!(m.check_trade_allowed(50.0, 20.0, 40.0).is_ok());
    }

    #[test]
    fn a_bracketless_entry_is_refused_per_trade_and_does_not_halt_the_day() {
        let m = operator_manager(100.0);
        let tier = m
            .check_trade_allowed(10.0, 0.0, 40.0)
            .expect_err("a zero stop-loss must be refused");
        assert_eq!(tier, rm::KillSwitchTier::PerTrade);
        assert!(!tier_halts_for_24h(tier));
    }

    #[test]
    fn an_oversized_entry_trips_the_presend_ceiling() {
        // 55% of 100 = 55. Anything at or above it is refused before the order
        // leaves the process — this is the tier that catches a lot that came
        // out wrong (bad pip value, a cap that did not bind).
        let m = operator_manager(100.0);
        let tier = m
            .check_trade_allowed(60.0, 20.0, 40.0)
            .expect_err("60 at risk on a 100 bankroll exceeds the 55% ceiling");
        assert_eq!(tier, rm::KillSwitchTier::PreSendSanity);
        assert!(!tier_halts_for_24h(tier));
    }

    #[test]
    fn accumulated_daily_losses_trip_the_day_cap_and_that_one_does_halt() {
        // Stage 0 daily cap is 80% of bankroll. Feed the manager real closed
        // trades exactly as the broker-reconcile block now does.
        let mut m = operator_manager(100.0);
        assert!(m.check_trade_allowed(10.0, 20.0, 40.0).is_ok());
        m.record_trade_outcome(-45.0);
        m.record_trade_outcome(-10.0);
        // bankroll 45, daily loss 55, cap = 0.80 * 45 = 36 ⇒ tripped.
        let tier = m
            .check_trade_allowed(1.0, 20.0, 40.0)
            .expect_err("the day cap must refuse further entries");
        assert_eq!(tier, rm::KillSwitchTier::PerDay);
        assert!(
            tier_halts_for_24h(tier),
            "a blown day is exactly what the persisted 24h cooldown is for"
        );
    }

    #[test]
    fn a_balance_that_grew_elsewhere_does_not_produce_a_false_presend_refusal() {
        // The regression this pins: the manager's cursor only sees THIS
        // engine's trades. Another engine wins, the account goes 100 -> 130,
        // and the next entry sizes at 50% of 130 = 65 — against a ceiling of
        // 0.55 * 100 = 55 if the cursor is stale. Refused, wrongly.
        let mut m = operator_manager(100.0);
        assert_eq!(
            m.check_trade_allowed(65.0, 20.0, 40.0),
            Err(rm::KillSwitchTier::PreSendSanity),
            "stale cursor: this is the false refusal"
        );
        m.sync_bankroll(130.0);
        assert!(
            m.check_trade_allowed(65.0, 20.0, 40.0).is_ok(),
            "after reconciling to the real balance the same order is fine"
        );
    }

    #[test]
    fn syncing_the_bankroll_keeps_the_days_losses_on_the_ledger() {
        // sync_bankroll must not launder a bad day. The cap is a fraction of
        // the CURRENT bankroll, so a recovery loosens it — but the accumulated
        // loss itself survives.
        let mut m = operator_manager(100.0);
        m.record_trade_outcome(-70.0);
        assert!((m.daily_loss_accumulated_usd() - 70.0).abs() < 1e-9);
        m.sync_bankroll(200.0);
        assert!(
            (m.daily_loss_accumulated_usd() - 70.0).abs() < 1e-9,
            "the day's losses are still on the ledger"
        );
        assert!((m.current_bankroll_usd() - 200.0).abs() < 1e-9);
    }

    #[test]
    fn a_failed_balance_fetch_never_moves_the_cursor() {
        // 0.0 / NaN come back from `fetch_account_runtime_blocking` failures.
        // Zeroing the bankroll would instantly trip PerStage on every engine.
        let mut m = operator_manager(400.0);
        let before = m.current_bankroll_usd();
        m.sync_bankroll(0.0);
        m.sync_bankroll(f64::NAN);
        m.sync_bankroll(-1.0);
        assert!((m.current_bankroll_usd() - before).abs() < 1e-9);
    }

    #[test]
    fn resetting_the_daily_accumulator_reopens_trading() {
        // Proves the period rollover in the entry block matters: without a
        // reset call the day cap trips once and stays tripped for the life of
        // the process.
        //
        // -46 on a 100 bankroll is chosen to trip the DAY cap only: it leaves
        // bankroll 54, day cap 0.80 x 54 = 43.2 (tripped by 46) and month cap
        // 0.99 x 54 = 53.46 (not tripped). A bigger loss would trip both and
        // the daily reset alone would not reopen trading — which is correct
        // behaviour, and exactly why the test picks the day-only case.
        let mut m = operator_manager(100.0);
        m.record_trade_outcome(-46.0);
        assert_eq!(
            m.check_trade_allowed(1.0, 20.0, 40.0),
            Err(rm::KillSwitchTier::PerDay)
        );
        m.reset_daily_accumulator();
        assert!(m.check_trade_allowed(1.0, 20.0, 40.0).is_ok());
    }

    #[test]
    fn weekly_account_loss_halts_until_the_iso_week_rolls() {
        let mut m = operator_manager(100.0);
        let cap = m.current_stage().weekly_drawdown_cap_fraction * m.current_bankroll_usd();
        m.raise_period_losses(0.0, cap, 0.0);

        let tier = m
            .check_trade_allowed(1.0, 20.0, 40.0)
            .expect_err("a spent ISO week must refuse a new entry");
        assert_eq!(tier, rm::KillSwitchTier::PerWeek);
        assert!(tier_halts_for_24h(tier));

        m.reset_daily_accumulator();
        assert_eq!(
            m.check_trade_allowed(1.0, 20.0, 40.0),
            Err(rm::KillSwitchTier::PerWeek),
            "the UTC-day rollover must not clear the ISO-week cap"
        );
        m.reset_weekly_accumulator();
        assert!(m.check_trade_allowed(1.0, 20.0, 40.0).is_ok());
    }

    // ── The ACCOUNT-wide, restart-durable loss ledger (2026-08-09) ───────────

    fn closed(net: f64, account: Option<&str>, exit_ms: i64) -> ClosedTrade {
        ClosedTrade {
            schema_version: 2,
            recorded_at_unix_ms: exit_ms,
            position_id: exit_ms,
            symbol: "EURUSD".to_string(),
            side: "BUY".to_string(),
            lots: 0.01,
            account_id: account.map(|s| s.to_string()),
            environment: Some("Live".to_string()),
            entry_ts_ms: Some(exit_ms - 1),
            entry_price: Some(1.1),
            exit_ts_ms: Some(exit_ms),
            exit_price: Some(1.1),
            gross_profit: net,
            commission: 0.0,
            swap: 0.0,
            net_profit: net,
            balance_after: None,
        }
    }

    /// 2026-08-09T12:00:00Z — a Sunday. Chosen deliberately: the ISO week runs
    /// Monday..Sunday, so "start of week" is 6 days back, which catches an
    /// off-by-one that a mid-week timestamp would hide.
    const NOW_MS: i64 = 1_786_276_800_000;

    #[test]
    fn losses_are_bucketed_by_utc_day_iso_week_and_calendar_month() {
        let day = 86_400_000i64;
        let trades = vec![
            closed(-10.0, Some("A"), NOW_MS - 3_600_000), // today
            closed(-20.0, Some("A"), NOW_MS - 2 * day),   // this ISO week
            closed(-40.0, Some("A"), NOW_MS - 7 * day),   // 2 Aug: this month, BEFORE Monday 3 Aug
            closed(-80.0, Some("A"), NOW_MS - 60 * day),  // a previous month
            closed(500.0, Some("A"), NOW_MS - 3_600_000), // a WIN — never counted
        ];
        let (d, w, m) = account_period_losses(&trades, Some("A"), NOW_MS);
        assert!((d - 10.0).abs() < 1e-9, "day = {d}");
        assert!((w - 30.0).abs() < 1e-9, "week = {w}");
        assert!((m - 70.0).abs() < 1e-9, "month = {m}");
    }

    #[test]
    fn another_engines_loss_on_the_same_account_closes_this_engines_day() {
        // THE DEFECT THIS CLOSES: the ledger was per-ENGINE while the account
        // is shared, so N engines permitted ~N x the intended daily cap.
        let mut m = operator_manager(100.0);
        // This engine has traded nothing.
        assert!(m.check_trade_allowed(1.0, 20.0, 40.0).is_ok());
        // A SIBLING engine lost 46 on the same account; the journal has it.
        let trades = vec![closed(-46.0, Some("A"), NOW_MS - 60_000)];
        let (d, w, mo) = account_period_losses(&trades, Some("A"), NOW_MS);
        m.sync_bankroll(54.0);
        m.raise_period_losses(d, w, mo);
        assert_eq!(
            m.check_trade_allowed(1.0, 20.0, 40.0),
            Err(rm::KillSwitchTier::PerDay),
            "the account's day is spent — this engine must not open another"
        );
    }

    #[test]
    fn a_foreign_accounts_losses_do_not_close_this_accounts_day() {
        let trades = vec![
            closed(-46.0, Some("OTHER"), NOW_MS - 60_000),
            closed(-1.0, None, NOW_MS - 60_000), // legacy, unattributable
        ];
        let (d, w, mo) = account_period_losses(&trades, Some("A"), NOW_MS);
        assert_eq!((d, w, mo), (0.0, 0.0, 0.0));
    }

    #[test]
    fn the_ledger_survives_a_restart_because_the_journal_does() {
        // A fresh manager (what a restart produces) reads the day's realized
        // loss back out of the journal instead of starting from zero.
        let mut fresh = operator_manager(54.0);
        assert!(fresh.check_trade_allowed(1.0, 20.0, 40.0).is_ok());
        let trades = vec![closed(-46.0, Some("A"), NOW_MS - 7_200_000)];
        let (d, w, mo) = account_period_losses(&trades, Some("A"), NOW_MS);
        fresh.raise_period_losses(d, w, mo);
        assert_eq!(
            fresh.check_trade_allowed(1.0, 20.0, 40.0),
            Err(rm::KillSwitchTier::PerDay),
            "restarting the app must not hand back a spent day"
        );
    }
}
