//! Live autonomous trading service (Path A).
//!
//! Polls the broker for new closed bars, computes features, evaluates gene
//! signals, and places/closes orders via cTrader.
//!
//! PARITY, STATED HONESTLY (corrected 2026-08-09). This header used to claim
//! the loop "uses the exact same pipeline as
//! `neoethos_trader::replay_portfolio_from_dir` so live signals are
//! byte-identical to the offline backtest". Two thirds of that is true and the
//! third is not, and the difference is where money leaks:
//!
//! - **Direction: shared shape.** Live nets the portfolio's genes with
//!   `neoethos_trader::combine_gene_signals_with_brackets`; the replay nets the
//!   same genes over the same feature cube with `combine_gene_signals`. Same
//!   gene evaluation, one carrying the brackets the live order needs.
//! - **Exits: NOT shared.** The discovery backtest (`neoethos-search/eval.rs`)
//!   and this loop both take their break-even/trailing geometry from
//!   `models.exit_policy`. `neoethos-trader` has **no trailing code at all** —
//!   zero occurrences of `trail` in the crate — so the Replay screen's exits
//!   are a different simulator from both. Do not read a replay number as a
//!   prediction of this loop's exits.
//! - **Execution: NOT shared.** The replay fills at the mark through
//!   `MockExecutionAdapter` behind a `PermissiveRiskGate`. This loop pays a
//!   real broker and passes every gate in this file.
//!
//! The parity that IS load-bearing, and that this file must not break, is
//! live-vs-**discovery**: same genes, same features, same exit policy.
//!
//! Entry point: [`start`].  The returned [`Handle`] stops the loop.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use neoethos_data::{Ohlcv, SymbolDataset};
use neoethos_trader::Direction;
use serde::{Deserialize, Serialize};

use crate::app_services::broker_api::{
    OrderSide, amend_position_sltp_blocking, close_position_blocking,
    fetch_recent_chart_bars_blocking, submit_market_order_blocking,
};

/// Account-wide per-UTC-day entry counter behind `risk.max_trades_per_day`.
///
/// ONE `static`, so every engine in the process shares it — engines are
/// per-portfolio but they all trade the SAME broker account, and a per-engine
/// counter would quietly turn a cap of "8" into `8 × engines` (the known
/// weakness of the unmerged 715058fe draft, deliberately not reproduced).
/// Only refuses entries when `risk.max_trades_per_day_enabled` arms it;
/// disarmed it still counts, so logs can always say where the day stands.
static ACCOUNT_DAILY_ENTRIES: neoethos_core::domain::daily_entry_cap::AccountDailyEntryCap =
    neoethos_core::domain::daily_entry_cap::AccountDailyEntryCap::new();

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
    /// itself and permanently retires the strategy (blacklist). Default 6.
    /// 0 disables auto-cull for this engine.
    #[serde(default = "default_cull_losses")]
    pub cull_after_consecutive_losses: u32,
    /// Auto-cull, rolling-window criterion: over the last `cull_window_trades`
    /// closed trades, the win rate must stay ≥ this percent or the strategy is
    /// retired. Catches CHRONIC losers that never lose N in a row (e.g. 40% WR
    /// alternating wins/losses bleeds the account but never streaks). Default
    /// 57% — the operator's break-even-plus-margin floor. 0 disables.
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
    6
}
pub fn default_cull_min_win_rate_pct() -> f64 {
    57.0
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
            consecutive_losses: 0,
            window_win_rate_pct: None,
            window_trades: 0,
            retired: false,
        }
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
        self.status
            .lock()
            .map(|s| s.running)
            .unwrap_or(false)
    }

    pub fn snapshot(&self) -> LiveTradingStatus {
        self.status
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Spawn the live trading loop and return a [`Handle`].  Returns immediately.
///
/// SAFETY GATE: on a REAL-money (Live) broker environment the strategy must
/// first clear the demo forward-test gate (≥100 demo fills + live metrics within
/// tolerance of backtest). A Demo environment is unconditionally allowed — that
/// is exactly how the demo fills accumulate. See [`crate::app_services::live_gate`].
pub fn start(req: StartRequest) -> Result<Handle> {
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
                 Run this strategy on a DEMO account until it qualifies, then switch to Live.",
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
        if let Err(e) = run(req, stop_clone, status_clone.clone(), gated_env_is_live).await {
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

fn tf_duration_ms(tf: &str) -> i64 {
    let m: i64 = 60_000;
    match tf {
        "M1" => m,
        "M2" => 2 * m,
        "M3" => 3 * m,
        "M4" => 4 * m,
        "M5" => 5 * m,
        "M10" => 10 * m,
        "M15" => 15 * m,
        "M30" => 30 * m,
        "H1" => 60 * m,
        "H4" => 240 * m,
        "H12" => 720 * m,
        "D1" => 1440 * m,
        "W1" => 10080 * m,
        _ => 60 * m,
    }
}

/// Does this kill-switch tier justify the PERSISTED 24 h halt, or only a
/// refusal of the order in hand?
///
/// W3 (2026-08-09). The ledger's instruction was "call `record_kill_switch_trip`
/// on the `Err` branch". Doing that for every tier would start a 24 h
/// account-wide halt because one order arrived with a malformed bracket, and a
/// safety control the operator learns to distrust is worse than no control.
///
/// - **Account-level** (halt): `PerDay`, `PerStage`, `PerMonth` say the bankroll
///   itself is in trouble; `Manual` and `HardwareConnLoss` are sticky halts that
///   already require an explicit clear. All five persist and stop every
///   Risky-Mode entry until the cooldown elapses or the bridge re-arms.
///
/// **What can actually fire, as of 2026-08-09** — stated because the first
/// version of this wiring advertised five halting tiers and three of them were
/// structurally unreachable:
/// - `PerDay` — live. Realized loss this UTC day (account-wide, see the journal
///   ledger at the entry site) reached `daily_loss_cap_fraction × bankroll`.
/// - `PerStage` — live as of the high-water fix in
///   `neoethos_core::domain::risky_mode`. Was unreachable before it.
/// - `PerMonth` — live, but INERT at the shipped `monthly_loss_cap_fraction`
///   (0.99): the day cap always binds first. Lower the fraction to arm it.
/// - `PreSendSanity` — live, the most frequently seen refusal.
/// - `PerTrade` — reachable only via an explicit zero/absent bracket; this loop
///   always resolves an SL and a TP, so in practice it does not fire here.
/// - `HardwareConnLoss` — **acquired a producer on 2026-08-09.** The broker
///   margin-call / account-disconnect watcher
///   (`crate::app_services::margin_call`) routes a cTrader margin-call or
///   account-disconnect event into this tier through
///   `risky_mode_persistence::record_kill_switch_trip`, so it is now a sticky
///   24 h halt the operator actually has. This classification did not change —
///   that was the point of classifying it as halting before a producer existed.
///   **If that module is ever removed, this bullet becomes a lie: revert it.**
/// - `Manual` — **still no producer.** `trip_manual_halt` has zero callers in
///   the workspace, so this tier remains inert. It stays classified as halting
///   so wiring a producer later needs no change here — but do not describe it to
///   the operator as protection he currently has.
/// - **Order-level** (refuse only): `PerTrade` (missing/invalid SL or TP) and
///   `PreSendSanity` (this order's implied risk exceeded the ceiling) describe
///   THIS order. Both are still refused, and both are logged at `error`.
/// - `ManualOrderWhileAutonomousOnly` cannot be produced by
///   `check_trade_allowed`; classified order-level so the match stays
///   exhaustive without inventing a halt.
pub(crate) fn tier_halts_for_24h(
    tier: neoethos_core::domain::risky_mode::KillSwitchTier,
) -> bool {
    use neoethos_core::domain::risky_mode::KillSwitchTier as T;
    match tier {
        T::PerDay | T::PerStage | T::PerMonth | T::Manual | T::HardwareConnLoss => true,
        T::PerTrade | T::PreSendSanity | T::ManualOrderWhileAutonomousOnly => false,
    }
}

/// Start-of-UTC-day / start-of-ISO-week / start-of-calendar-month, in epoch ms,
/// for the instant `now_ms`.
fn period_starts_ms(now_ms: i64) -> Option<(i64, i64, i64)> {
    use chrono::{Datelike, NaiveDate};
    let now = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(now_ms)?;
    let d = now.date_naive();
    let to_ms = |x: NaiveDate| x.and_hms_opt(0, 0, 0).map(|t| t.and_utc().timestamp_millis());
    let day = to_ms(d)?;
    // Monday-anchored, matching `iso_week()` used for the weekly accumulator.
    let week = to_ms(d - chrono::Duration::days(d.weekday().num_days_from_monday() as i64))?;
    let month = to_ms(NaiveDate::from_ymd_opt(d.year(), d.month(), 1)?)?;
    Some((day, week, month))
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

pub(crate) fn bars_to_ohlcv(bars: &[crate::app_services::ctrader_data::HistoricalBar]) -> Ohlcv {
    Ohlcv {
        timestamp: Some(bars.iter().map(|b| b.timestamp_ms).collect()),
        open: bars.iter().map(|b| b.open).collect(),
        high: bars.iter().map(|b| b.high).collect(),
        low: bars.iter().map(|b| b.low).collect(),
        close: bars.iter().map(|b| b.close).collect(),
        volume: Some(
            bars.iter()
                .map(|b| b.volume.unwrap_or(0) as f64)
                .collect(),
        ),
    }
}

// ── Risk-based position sizing ──────────────────────────────────────────────────

/// Resolve the `quote → account` FX rate so cross-pair pip values can be
/// converted into the account currency (e.g. USD→GBP via GBPUSD). Blocking —
/// fetches a few recent bars of the bridging pair from the broker. Returns
/// `None` when neither orientation of the bridge pair is fetchable, so the
/// caller falls back to a fixed lot rather than mis-size.
fn resolve_quote_to_account_rate(quote: &str, account: &str, tf: &str) -> Option<f64> {
    let q = quote.trim().to_ascii_uppercase();
    let a = account.trim().to_ascii_uppercase();
    if q.is_empty() || a.is_empty() {
        return None;
    }
    if q == a {
        return Some(1.0);
    }
    let last_close = |sym: &str| -> Option<f64> {
        crate::app_services::broker_api::fetch_recent_chart_bars_blocking(sym, tf, 3)
            .ok()
            .and_then(|bars| bars.last().map(|b| b.close))
            .filter(|c| c.is_finite() && *c > 0.0)
    };
    // ACCOUNT+QUOTE (e.g. GBPUSD): price = QUOTE units per 1 ACCOUNT → quote→account = 1/price.
    if let Some(p) = last_close(&format!("{a}{q}")) {
        return Some(1.0 / p);
    }
    // QUOTE+ACCOUNT (e.g. USDGBP): price = ACCOUNT units per 1 QUOTE → quote→account = price.
    if let Some(p) = last_close(&format!("{q}{a}")) {
        return Some(p);
    }
    None
}

/// Position size (lots) for one entry, from the account's risk budget and the
/// strategy's OWN stop distance: `lots = balance × risk% / (sl_pips ×
/// pip_value_per_lot_in_account)`, snapped to the symbol's lot step and clamped
/// to `[min_lot, min(max_lot, max_lot_cap)]`. Returns `fallback` whenever a
/// correct size can't be computed (no balance / risk / stop, missing metadata,
/// or a cross pair whose pip value collapses to NaN without an FX rate) — it
/// NEVER returns a wrong size.
#[allow(clippy::too_many_arguments)]
fn risk_based_lots(
    balance: f64,
    risk_fraction: f64,
    sl_pips: f64,
    meta: Option<&neoethos_core::symbol_metadata::SymbolMetadata>,
    account_ccy: &str,
    fx_quote_to_account: Option<f64>,
    live_price: Option<f64>,
    fallback: f64,
    max_lot_cap: f64,
) -> f64 {
    if !(balance > 0.0 && risk_fraction > 0.0 && sl_pips.is_finite() && sl_pips > 0.0) {
        return fallback;
    }
    let Some(meta) = meta else {
        return fallback;
    };
    let pip_val = meta.pip_value_in_account(account_ccy, fx_quote_to_account, live_price);
    if !(pip_val.is_finite() && pip_val > 0.0) {
        return fallback;
    }
    let raw = (balance * risk_fraction) / (sl_pips * pip_val);
    if !(raw.is_finite() && raw > 0.0) {
        return fallback;
    }
    let step = if meta.lot_step > 0.0 { meta.lot_step } else { 0.01 };
    let min_lot = if meta.min_lot > 0.0 { meta.min_lot } else { step };
    let max_lot = meta.max_lot.min(max_lot_cap).max(min_lot);
    let mut lots = (raw / step).floor() * step;

    // Affordability guard: a small account must NEVER be handed a position it
    // can't hold (operator saw a 47-lot order). Cap the NOTIONAL to
    // balance × a conservative max leverage — independent of pip_value, so a
    // mis-resolved pip value (tiny denominator → huge `raw`) can't blow the lot
    // count up. Uses live price × contract size × the quote→account FX rate.
    if let Some(price) = live_price.filter(|p| p.is_finite() && *p > 0.0) {
        let fx = fx_quote_to_account.filter(|r| r.is_finite() && *r > 0.0).unwrap_or(1.0);
        let notional_per_lot = meta.contract_size * price * fx;
        if notional_per_lot > 0.0 {
            const MAX_LEVERAGE: f64 = 30.0; // conservative; under-sizes safely
            let affordable = (balance * MAX_LEVERAGE) / notional_per_lot;
            if affordable < lots {
                lots = (affordable / step).floor() * step;
            }
        }
    }
    lots.clamp(min_lot, max_lot)
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
) -> Result<()> {
    // Load portfolio artifact (same as replay_portfolio_from_dir)
    let artifact = neoethos_search::load_live_portfolio_json(&req.portfolio_path)
        .with_context(|| format!("load live portfolio {}", req.portfolio_path))?;

    if artifact.genes.is_empty() {
        anyhow::bail!("portfolio '{}' has no genes", req.portfolio_path);
    }
    if artifact.normalize_features {
        anyhow::bail!(
            "portfolio was discovered with feature normalisation ON — \
             normalization stats are not persisted, cannot reproduce live features. \
             Re-run discovery with normalisation OFF."
        );
    }

    let symbol = artifact.symbol.clone();
    let base_tf = artifact.base_tf.clone();
    let higher_tfs = artifact.higher_tfs.clone();
    let effective_names = artifact.effective_feature_names.clone();
    let genes = artifact.genes.clone();

    if let Ok(mut s) = status.lock() {
        s.symbol = Some(symbol.clone());
        s.base_tf = Some(base_tf.clone());
        s.genes = genes.len();
    }

    let bar_ms = tf_duration_ms(&base_tf);
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
    let mut opened_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut consecutive_losses: u32 = 0;
    let mut net_pnl_running: f64 = 0.0;
    // Live-learning foundation (operator 2026-07-02): remember the EXACT
    // feature row each entry acted on; pair it with the realized outcome at
    // close and append to the experience store. Pure data collection — the
    // online/RL experts train OFFLINE from this (never silently live).
    let mut pending_experience: HashMap<i64, crate::app_services::experience_store::LiveExperience> =
        HashMap::new();
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
    let sizing = neoethos_core::Settings::from_yaml(&crate::server::state::current_config_path()).ok();
    let risk_fraction = sizing
        .as_ref()
        .map(|s| s.risk.risk_per_trade)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    // Risky Mode sizing context. When `trading_mode == "risky"` the per-entry
    // risk comes from the bankroll-stage ladder (30 %→50 %, tapering as the
    // account grows) instead of the static prop-firm `risk_per_trade`. Read
    // once here; the actual stage fraction is resolved per entry off the LIVE
    // balance (the account compounds). The default "prop_firm" mode leaves the
    // sizing path byte-for-byte unchanged.
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
    // Journal location + the account this engine trades — the inputs to the
    // ACCOUNT-WIDE risky-mode loss ledger consulted at the pre-send check.
    // Resolved once: `data_dir` does not move at runtime, and the account id is
    // the one `broker_api::resolve_creds` routes to (the engine already stops
    // itself if the environment, and therefore the account, changes).
    let journal_data_dir: Option<std::path::PathBuf> =
        sizing.as_ref().map(|s| s.system.data_dir.clone());
    let journal_account: Option<String> =
        crate::app_services::journal_store::active_account_id();
    let max_lot_cap = sizing
        .as_ref()
        .map(|s| s.risk.max_lot_size)
        .filter(|v| *v > 0.0)
        .unwrap_or(f64::INFINITY);
    // Portfolio-level concurrent-risk cap (0 = disabled): each entry budgets
    // against `cap − open_positions × risk_per_trade` using the broker's LIVE
    // position count, so many engines can't stack unbounded concurrent risk.
    let portfolio_risk_cap = sizing
        .as_ref()
        .map(|s| s.risk.max_portfolio_risk)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    // Weekend kill zones — force-close before the weekend, block Fri-late /
    // Mon-open entries.
    //
    // CORRECTED 2026-08-04. This comment used to read:
    //
    //   "PARITY with the backtest (eval.rs): discovery runs with
    //    kill_zones_enabled from the SAME config flag ... Live must match or
    //    positions ride weekend gaps no validated strategy ever held through."
    //
    // Discovery does NOT read that flag. `discovery_backtest_settings`
    // (neoethos-search/src/discovery.rs:1365) hardcodes
    // `kill_zones_enabled: true`, so every backtest that ever validated a
    // strategy ran WITH kill zones, unconditionally. Only this live path
    // consults `risk.kill_zones_enabled` (neoethos-core/src/config.rs:378,
    // default true).
    //
    // So the flag is not a shared switch, it is a one-sided one, and the
    // failure it creates is exactly the one the old comment warned about:
    // setting `risk.kill_zones_enabled = false` makes live hold through
    // weekend gaps that no backtest in the artifact history ever held
    // through. Defaults agree (both true), which is why this went unseen.
    //
    // Do not "restore parity" by wiring the flag into discovery without
    // deciding which side is authoritative — that would silently re-score
    // every strategy in the library against a different simulator.
    let kill_zones_enabled = sizing
        .as_ref()
        .map(|s| s.risk.kill_zones_enabled)
        .unwrap_or(true);
    // Live spread gate reference: the spread the BACKTEST charged per trade.
    // When the live spread blows past a multiple of it (rollover, thin books),
    // entering would pay costs the validated edge never budgeted for.
    let backtest_spread_pips = sizing
        .as_ref()
        .map(|s| s.risk.backtest_spread_pips)
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.5);
    // ── Exit geometry — THE SAME RESOLVED VALUES THE SEARCH READS (#208/#74) ───
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
    // FAIL-CLOSED. `None` here means the config could not be read at all
    // (`Settings::from_yaml` failed above). We do NOT fall back to the constant:
    // an unresolvable policy means we cannot prove parity with what was scored,
    // so the trail does not arm and positions run to their real SL/TP. That
    // matches the shipped default, which is also OFF.
    let exit_policy: Option<neoethos_core::config::ExitPolicyConfig> =
        sizing.as_ref().map(|s| s.models.exit_policy);
    let exit_policy_config_path = crate::server::state::current_config_path();
    match exit_policy {
        Some(p) if p.trailing_enabled => tracing::warn!(
            target: "neoethos_app::live_trading",
            %symbol,
            trailing_enabled = true,
            be_trigger_r = p.trailing_be_trigger_r,
            stop_multiplier = p.trailing_stop_multiplier,
            min_lock_pips = p.trailing_min_lock_pips,
            "LIVE TRAILING ARMED from models.exit_policy — stops will be pulled \
             to break-even once price reaches the trigger. This must match the \
             policy discovery scored these genes under, or live gives back wins \
             the backtest was paid for"
        ),
        Some(_) => tracing::info!(
            target: "neoethos_app::live_trading",
            %symbol,
            trailing_enabled = false,
            "live trailing DISABLED by models.exit_policy.trailing_enabled — \
             open positions run to their real stop or take-profit, matching \
             what discovery scored"
        ),
        None => tracing::error!(
            target: "neoethos_app::live_trading",
            %symbol,
            config_path = %exit_policy_config_path.display(),
            "models.exit_policy is UNRESOLVABLE (Settings failed to load) — \
             REFUSING TO TRAIL. No stop on any open position will be moved by \
             this engine. Fix the config to restore configured exit behaviour"
        ),
    }
    // ── Risky-mode per-trade ceiling: config vs constant (#209/#210) ───────────
    //
    // 2026-08-09. `risk.risky_max_risk_per_trade` (`config.rs:345`, shipped
    // 0.30) had exactly one reader in the workspace and it was the SEARCH
    // (`discovery.rs:820`). The live ladder is bounded by
    // `RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION = 0.50`
    // (`domain/risky_mode.rs:136`), and the entry site below takes its base risk
    // straight from `stage_risk_fraction_for_bankroll`. Twenty percentage points
    // of the account per trade, with nothing reconciling the two numbers.
    //
    // Resolution, deliberately NOT the operator's decision on which number is
    // "right": THE LOWER ONE BINDS. A limit the operator wrote down is never
    // silently raised. The constant stays where it is (`server/risky.rs` reports
    // it as the band's ceiling); this clamp only ever shrinks the entry.
    //
    // A non-finite or non-positive configured value is NOT treated as "size
    // zero" — it is treated as unusable, logged at error, and leaves the ladder
    // unclamped, because inventing a ceiling from a corrupt field is as wrong as
    // ignoring a real one.
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
                 bound anything (non-finite or <= 0) — IGNORING IT. The risky \
                 ladder ceiling of 0.50 stands unclamped. Set a fraction in \
                 (0, 1] to bind it"
            );
            None
        }
        None => None,
    };
    if trading_mode_risky {
        let ladder_ceiling = neoethos_core::domain::risky_mode::RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION;
        match risky_configured_ceiling {
            Some(cfg) if cfg < ladder_ceiling => tracing::warn!(
                target: "neoethos_app::live_trading",
                %symbol,
                configured = cfg,
                ladder_ceiling,
                effective = cfg,
                bound_by = "risk.risky_max_risk_per_trade",
                "RISKY SIZING DISAGREEMENT — the config and the bankroll ladder \
                 name different per-trade ceilings. The LOWER one binds: every \
                 entry is capped at the configured fraction, so the early ladder \
                 rungs size DOWN. Change risk.risky_max_risk_per_trade to lift it"
            ),
            Some(cfg) => tracing::warn!(
                target: "neoethos_app::live_trading",
                %symbol,
                configured = cfg,
                ladder_ceiling,
                effective = ladder_ceiling,
                bound_by = "RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION",
                "RISKY SIZING DISAGREEMENT — the configured ceiling is at or \
                 above the ladder's. The LOWER one binds, so the ladder's \
                 constant governs; the config raises nothing"
            ),
            None => tracing::warn!(
                target: "neoethos_app::live_trading",
                %symbol,
                ladder_ceiling,
                effective = ladder_ceiling,
                bound_by = "RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION",
                "RISKY MODE with NO configured per-trade ceiling \
                 (risk.risky_max_risk_per_trade is unset) — only the ladder's \
                 constant bounds entry size"
            ),
        }
    }
    // Load the soft-voting ensemble ONCE at engine start (loading ~30 expert
    // artifacts takes seconds — far too slow per bar). Fail-soft: if the gate
    // is on but the ensemble can't load (nothing trained yet, wrong symbol/TF
    // dir), log loudly and run gene-only — never block trading on ML infra.
    let live_ensemble: Option<
        std::sync::Arc<neoethos_models::ensemble_inference::soft_voting::SoftVotingEnsemble>,
    > = if live_ml_gate {
        let sym = symbol.clone();
        let tf = base_tf.clone();
        match tokio::task::spawn_blocking(move || {
            neoethos_models::ensemble_inference::build_ensemble_for_symbol(
                std::path::Path::new("models"),
                &sym,
                &tf,
            )
        })
        .await
        {
            Ok(Ok(ensemble)) => {
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
                    missing = outcome.missing_count(),
                    degraded = outcome.degraded_count(),
                    voting = ensemble.voting_expert_count(),
                    unused_for_voting = %unused,
                    "LIVE ML gate armed — ensemble voters loaded (genes still pick direction; ML only scales size)"
                );
                Some(std::sync::Arc::new(ensemble))
            }
            Ok(Err(err)) => {
                tracing::warn!(
                    target: "neoethos_app::live_trading",
                    %symbol, %base_tf, error = %err,
                    "models.live_ml_gate is ON but the ensemble failed to load — running gene-only"
                );
                None
            }
            Err(join_err) => {
                tracing::warn!(
                    target: "neoethos_app::live_trading",
                    %symbol, error = %join_err,
                    "ensemble loader task failed — running gene-only"
                );
                None
            }
        }
    } else {
        None
    };
    let sym_meta = neoethos_core::symbol_metadata::resolve(&symbol);
    let quote_ccy = sym_meta.as_ref().map(|m| m.quote.clone());
    let sizing_tf = base_tf.clone();
    // Balance + REAL account currency + quote→account FX, all on one blocking hop.
    let (account_balance, account_ccy, fx_quote_to_account) =
        tokio::task::spawn_blocking(move || {
            match crate::app_services::broker_api::fetch_account_runtime_blocking() {
                Ok(snap) => {
                    let bal = snap.trader.balance;
                    let ccy = crate::server::bridge::asset_id_to_currency(
                        snap.trader.deposit_asset_id,
                    )
                    .to_string();
                    let fx = quote_ccy
                        .as_deref()
                        .and_then(|q| resolve_quote_to_account_rate(q, &ccy, &sizing_tf));
                    (bal, ccy, fx)
                }
                Err(_) => (0.0, String::new(), None),
            }
        })
        .await
        .unwrap_or((0.0, String::new(), None));
    tracing::info!(
        target: "neoethos_app::live_trading",
        %symbol,
        balance = account_balance,
        account_ccy = %account_ccy,
        risk_fraction,
        fx_quote_to_account = ?fx_quote_to_account,
        "risk-sizing context resolved"
    );

    // ── Risky Mode kill switch (W3, 2026-08-09) ───────────────────────────────
    // `RiskyModeManager` is 1,848 lines implementing seven kill-switch tiers, a
    // pre-send sanity ceiling and daily/weekly/monthly loss accumulators. Until
    // now its ONLY construction in the workspace was inside
    // `GET /risky/scenarios` (`server/risky.rs:144`), which called
    // `time_to_target_scenarios()` and threw the manager away — while THIS loop
    // sized entries at 30–50 % of the live balance through the free function
    // `stage_risk_fraction_for_bankroll` with nothing behind it. The brakes were
    // compiled and disconnected.
    //
    // Scope: the manager exists ONLY when `system.trading_mode == "risky"`. The
    // default `prop_firm` path constructs nothing and its behaviour is
    // byte-identical to before this change.
    //
    // `autonomous_only_contract_accepted: true` is the manager's construction
    // gate (`RiskyModeManager::new` bails without it). It is factually correct
    // here and nothing more: this loop IS the autonomous producer — every order
    // it sends comes from a gene signal, never from an operator click — and the
    // flag's only behavioural reader, `rejects_manual_orders()`, is not
    // consulted from this file. It does NOT clamp the manual `POST /orders`
    // path, which the operator has ruled respects him.
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
            stages: rm::build_logarithmic_stages(
                risky_start_balance,
                risky_target_balance,
                rm::DEFAULT_DOUBLING_FACTOR,
            ),
            autonomous_only_contract_accepted: true,
            allow_live_broker: true,
            ..rm::RiskyModeConfig::default()
        };
        // FAIL CLOSED. If the ladder cannot be validated, the 30–50 % sizing it
        // authorises must not run either. Before this change the same bad
        // config silently degraded to `risk.risk_per_trade` with no kill switch
        // at all; now the engine refuses to start and says why.
        let manager = rm::RiskyModeManager::new(cfg, bankroll).with_context(|| {
            format!(
                "Risky Mode is ON (system.trading_mode = \"risky\") but its kill switch could \
                 not be built from system.risky_start_balance_usd = {risky_start_balance} / \
                 system.risky_target_balance_usd = {risky_target_balance} with a live balance \
                 of {bankroll}. REFUSING TO START: Risky Mode sizes entries at 30-50% of the \
                 account and will not run without its daily/stage/monthly loss caps and its \
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
            presend_ceiling = manager.config().presend_sanity_ceiling_fraction,
            monthly_loss_cap = manager.config().monthly_loss_cap_fraction,
            high_water_stage_idx = manager.high_water_stage_idx(),
            "RISKY MODE KILL SWITCH ARMED — every entry is checked before the \
             order is sent. Tiers that can actually fire: PreSendSanity (this \
             order's risk >= 55% of bankroll), PerDay (the ACCOUNT's realized \
             loss this UTC day reached the stage cap), PerStage (the bankroll \
             retreated below the rung under the highest stage reached), \
             PerMonth (inert at the shipped 0.99 cap — the day cap binds \
             first) and HardwareConnLoss (a broker margin-call or \
             account-disconnect event, produced by app_services::margin_call \
             as of 2026-08-09 — a sticky 24 h halt). Manual still has no \
             producer and is inert; do not count on it."
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

    // Session-level circuit breakers (audit S03, 2026-07-13): the config has
    // always carried `risk.daily_drawdown_limit` / `risk.total_drawdown_limit`
    // (fractions of balance), but NOTHING enforced them live — the autopilot
    // had per-trade sizing caps yet could bleed the account all day with no
    // automatic stop. Enforced below at entry time, on the fresh broker
    // balance (realized PnL — equity-based tracking is a follow-up). The
    // breakers only BLOCK NEW ENTRIES; exit management is untouched.
    // `0.0` disables, matching the other risk caps.
    let daily_dd_limit = sizing
        .as_ref()
        .map(|s| s.risk.daily_drawdown_limit)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let total_dd_limit = sizing
        .as_ref()
        .map(|s| s.risk.total_drawdown_limit)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let initial_balance_cfg = sizing
        .as_ref()
        .map(|s| s.risk.initial_balance)
        .filter(|b| b.is_finite() && *b > 0.0)
        .unwrap_or(account_balance);
    // Account-wide daily entry cap (2026-08-08): `risk.max_trades_per_day`
    // sat in the operator config with NOTHING on the entry path reading it —
    // his engines took 20-47 entries/day each against a configured 8. Armed
    // only by `risk.max_trades_per_day_enabled` (default false ⇒ `None` here
    // ⇒ behaviour unchanged); `max_trades_per_day: 0` disables like the other
    // caps. The counter is the process-wide `ACCOUNT_DAILY_ENTRIES` static —
    // per ACCOUNT, shared across every running engine, NOT per engine.
    let daily_entry_cap: Option<u32> = sizing
        .as_ref()
        .filter(|s| s.risk.max_trades_per_day_enabled)
        .map(|s| s.risk.max_trades_per_day)
        .filter(|&cap| cap > 0)
        .map(|cap| u32::try_from(cap).unwrap_or(u32::MAX));
    if let Some(cap) = daily_entry_cap {
        tracing::warn!(
            target: "neoethos_app::live_trading",
            %symbol, cap,
            "DAILY ENTRY CAP ARMED (risk.max_trades_per_day_enabled) — at most \
             this many entries per UTC day on the WHOLE account, shared across \
             every running engine; counter resets at UTC midnight and on app \
             restart"
        );
    }
    // (UTC date id, balance at first entry-consideration of that day).
    let mut day_start: Option<(u32, f64)> = None;
    // Log-once latches so a tripped breaker doesn't flood the log every bar.
    let mut daily_tripped_on: Option<u32> = None;
    let mut total_tripped = false;

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
        let mut base_bars_opt = None;
        for attempt in 0..max_tries {
            let sym = symbol.clone();
            let tf = base_tf.clone();
            match tokio::task::spawn_blocking(move || {
                fetch_recent_chart_bars_blocking(&sym, &tf, warmup)
            })
            .await?
            {
                Ok(b) => { base_bars_opt = Some(b); break; }
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
        let base_bars = match base_bars_opt {
            Some(b) => b,
            None => continue,
        };

        // Check if there really is a new bar
        let latest_ts = base_bars.last().map(|b| b.timestamp_ms).unwrap_or(0);
        if latest_ts <= last_bar_ts {
            tracing::debug!(
                target: "neoethos_app::live_trading",
                last_bar_ts, latest_ts,
                "no new bar yet"
            );
            continue;
        }
        last_bar_ts = latest_ts;

        // ── Weekend kill zone — PARITY with the backtest ──────────────────────
        // The backtest force-closes every position on Friday ≥ 20:00 UTC (no
        // validated strategy ever held through a weekend gap). Mirror it live.
        if kill_zones_enabled {
            let (force_close, _) = weekend_kill_zone(latest_ts);
            if force_close {
                if let Some((pos_id, vol)) = open_position.take() {
                    let result = tokio::task::spawn_blocking(move || {
                        // Same guard as the entry submit: never touch an
                        // account this engine was not admitted to. The
                        // expectation is validated by the same
                        // `broker_credentials.toml` read that routes the close,
                        // so there is no window between check and send.
                        close_position_blocking(pos_id, vol, Some(gated_env_is_live))
                    })
                    .await?;
                    match result {
                        Ok(_) => tracing::info!(
                            target: "neoethos_app::live_trading",
                            %symbol, position_id = pos_id,
                            "weekend kill zone — position force-closed (parity with backtest)"
                        ),
                        Err(e) => tracing::warn!(
                            target: "neoethos_app::live_trading",
                            error = %e, position_id = pos_id,
                            "weekend kill zone close failed — will retry next bar"
                        ),
                    }
                    pos_sl_pips = 0.0;
                    if let Ok(mut s) = status.lock() {
                        s.open_position_id = None;
                        s.last_signal = Some("weekend kill zone — flat".to_string());
                    }
                }
            }
        }

        // ── Broker reconcile: account THIS engine's closed trades ─────────────
        // Reads the broker's closing deals (net_profit) for positions we opened
        // — catches SL/TP exits, not just engine-initiated closes. On N
        // consecutive losses the strategy is permanently retired (blacklisted)
        // and the engine stops.
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
                for deal in &runtime.recent_deals {
                    let Some(net) = deal.net_profit else { continue };
                    if opened_ids.remove(&deal.position_id) {
                        net_pnl_running += net;
                        // W3 (2026-08-09): feed the Risky Mode kill switch its
                        // ONLY input. Without this the daily / weekly / monthly
                        // loss accumulators stay at zero forever and
                        // `check_trade_allowed` can never trip a loss tier —
                        // the gate would be wired but blind. `net` is this
                        // engine's realized PnL on a position it opened, in the
                        // account's deposit currency, which is the same unit
                        // the bankroll was seeded in.
                        if let Some(m) = risky_manager.as_mut() {
                            m.record_trade_outcome(net);
                            tracing::info!(
                                target: "neoethos_app::live_trading",
                                position_id = deal.position_id,
                                net_profit = net,
                                bankroll = m.current_bankroll_usd(),
                                stage_idx = m.current_stage().stage_idx,
                                daily_loss = m.daily_loss_accumulated_usd(),
                                monthly_loss = m.monthly_loss_accumulated_usd(),
                                "risky-mode kill switch: trade outcome recorded"
                            );
                        }
                        if net < 0.0 {
                            consecutive_losses += 1;
                        } else {
                            consecutive_losses = 0;
                        }
                        // Complete + persist the experience pair (entry features
                        // → realized outcome) for offline live-learning.
                        if let Some(mut exp) = pending_experience.remove(&deal.position_id) {
                            exp.close_ts_ms = Some(deal.execution_timestamp_ms);
                            exp.net_profit = Some(net);
                            crate::app_services::experience_store::record(&exp);
                        }
                        recent_results.push_back(net > 0.0);
                        while recent_results.len() > cull_window {
                            recent_results.pop_front();
                        }
                        // If the broker closed OUR tracked position (SL/TP), drop it
                        // so trailing doesn't try to amend a dead position.
                        if open_position.map(|(id, _)| id) == Some(deal.position_id) {
                            open_position = None;
                            pos_sl_pips = 0.0;
                        }
                        tracing::info!(
                            target: "neoethos_app::live_trading",
                            position_id = deal.position_id, net_profit = net,
                            consecutive_losses, "auto-cull: closed trade accounted"
                        );
                    }
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
                let mut cull_reason: Option<String> = None;
                if cull_threshold > 0 && consecutive_losses >= cull_threshold {
                    cull_reason = Some(format!(
                        "{consecutive_losses} consecutive losing trades (demo/live auto-cull)"
                    ));
                } else if cull_min_wr > 0.0 && recent_results.len() >= cull_window {
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
                    tracing::warn!(
                        target: "neoethos_app::live_trading",
                        %symbol, portfolio_path = %portfolio_path,
                        %reason, net_pnl = net_pnl_running,
                        "AUTO-CULL: retiring strategy (blacklist)"
                    );
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
                    // Flatten any position we still hold before retiring.
                    if let Some((pos_id, vol)) = open_position.take() {
                        let _ = tokio::task::spawn_blocking(move || {
                            close_position_blocking(pos_id, vol, Some(gated_env_is_live))
                        })
                        .await;
                    }
                    if let Ok(mut s) = status.lock() {
                        s.retired = true;
                        s.running = false;
                        s.open_position_id = None;
                    }
                    // Close the loop: the retirement left a coverage gap on this
                    // (symbol, base_tf) — queue a fresh Discovery to refill it.
                    // The retired strategy itself can never return (blacklisted).
                    crate::app_services::rediscovery::request(symbol.clone(), base_tf.clone());
                    break;
                }
            }
        }

        // ── Build multi-TF SymbolDataset ──────────────────────────────────────
        let mut frames: HashMap<String, Ohlcv> = HashMap::new();
        let base_ohlcv = bars_to_ohlcv(&base_bars);
        frames.insert(base_tf.clone(), base_ohlcv.clone());

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
        // `models.exit_policy` via `strategy_gene.rs:867`, and the shipped
        // default is now `trailing_enabled: false`. So "ALWAYS ON" is wrong,
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
        // When the policy is OFF — or unresolvable — NOTHING here runs and no
        // stop is moved.
        let trailing = exit_policy.filter(|p| p.trailing_enabled);
        if let (Some(policy), Some((pos_id, _))) = (trailing, open_position) {
            if pos_sl_pips > 0.0 && pos_entry_px > 0.0 {
                let pip = sym_meta
                    .as_ref()
                    .map(|m| m.pip_size)
                    .filter(|p| p.is_finite() && *p > 0.0)
                    .unwrap_or(0.0001);
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
                        "models.exit_policy has trailing ENABLED but its geometry \
                         is unusable — REFUSING TO MOVE THE STOP this bar. Fix \
                         trailing_be_trigger_r / trailing_stop_multiplier / \
                         trailing_min_lock_pips"
                    );
                } else {
                    let r_dist = pos_sl_pips * pip; // 1R in price units
                    let trigger_dist = trigger_r * r_dist;
                    let trail_dist = stop_mult * r_dist;
                    let hi = base_ohlcv.high.last().copied().unwrap_or(pos_entry_px);
                    let lo = base_ohlcv.low.last().copied().unwrap_or(pos_entry_px);
                    let mut new_trail: Option<f64> = None;
                    // Same floor the backtest applies (`eval.rs:1052`): once the
                    // trail engages it never sits closer to entry than the locked
                    // profit. Without it the live stop protects a different amount
                    // than the strategy was scored on.
                    let locked = lock_pips * pip;
                    if pos_is_long {
                        pos_extreme = pos_extreme.max(hi);
                        if pos_extreme - pos_entry_px >= trigger_dist {
                            let candidate =
                                (pos_extreme - trail_dist).max(pos_entry_px + locked);
                            if pos_trail_px == 0.0 || candidate > pos_trail_px {
                                pos_trail_px = candidate;
                                new_trail = Some(candidate);
                            }
                        }
                    } else {
                        pos_extreme =
                            if pos_extreme > 0.0 { pos_extreme.min(lo) } else { lo };
                        if pos_entry_px - pos_extreme >= trigger_dist {
                            let candidate =
                                (pos_extreme + trail_dist).min(pos_entry_px - locked);
                            if pos_trail_px == 0.0 || candidate < pos_trail_px {
                                pos_trail_px = candidate;
                                new_trail = Some(candidate);
                            }
                        }
                    }
                    if let Some(raw) = new_trail {
                        let sl_price = (raw / pip).round() * pip; // snap to a broker-valid pip grid
                        // NOTE (#199): this amend is the one order-path call not
                        // bound to the admitted environment, and the result is
                        // dropped here. `amend_position_sltp_blocking` logs its
                        // own failure loudly; the environment binding is being
                        // added to that function separately.
                        let _ = tokio::task::spawn_blocking(move || {
                            amend_position_sltp_blocking(pos_id, Some(sl_price), None, None)
                        })
                        .await;
                        tracing::info!(
                            target: "neoethos_app::live_trading",
                            position_id = pos_id, new_sl = sl_price, extreme = pos_extreme,
                            be_trigger_r = trigger_r,
                            stop_multiplier = stop_mult,
                            min_lock_pips = lock_pips,
                            "trailing stop advanced (models.exit_policy geometry)"
                        );
                    }
                }
            }
        }

        for htf in &higher_tfs {
            let sym = symbol.clone();
            let tf = htf.clone();
            match tokio::task::spawn_blocking(move || {
                fetch_recent_chart_bars_blocking(&sym, &tf, warmup)
            })
            .await?
            {
                Ok(htf_bars) => {
                    frames.insert(htf.clone(), bars_to_ohlcv(&htf_bars));
                }
                Err(e) => {
                    tracing::warn!(
                        target: "neoethos_app::live_trading",
                        tf = %htf, error = %e,
                        "failed to fetch higher-TF bars, continuing with partial dataset"
                    );
                }
            }
        }

        let dataset = SymbolDataset {
            symbol: symbol.clone(),
            frames,
        };

        // ── Feature computation ───────────────────────────────────────────────
        let higher_refs: Vec<&str> = higher_tfs.iter().map(|s| s.as_str()).collect();
        let raw_features =
            match neoethos_data::prepare_multitimeframe_features(&dataset, &base_tf, &higher_refs) {
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

        let aligned =
            match neoethos_search::project_features_to_effective(&raw_features, &effective_names) {
                Ok(a) => a,
                Err(e) => {
                    tracing::warn!(
                        target: "neoethos_app::live_trading",
                        error = %e,
                        "feature projection failed (effective_names mismatch?), skipping bar"
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
        // Pass the symbol pip size so adaptive-stop genes scale their bracket by
        // live volatility exactly like the discovery backtest (parity).
        let bracket_pip_size = sym_meta
            .as_ref()
            .map(|m| m.pip_size)
            .filter(|p| p.is_finite() && *p > 0.0)
            .unwrap_or(0.0001);
        let (directions, sl_arr, tp_arr) = neoethos_trader::combine_gene_signals_with_brackets(
            &genes,
            &aligned,
            &base_ohlcv,
            bracket_pip_size,
        );
        let direction = directions.last().copied().unwrap_or(Direction::Flat);
        // Gene-derived SL/TP (pips) for THIS bar: we place the STRATEGY'S own
        // brackets, never an imposed stop. 0.0 ⇒ a signal-exit-only strategy, so
        // the live order stays bracket-free (exactly what the backtest does).
        let gene_sl = sl_arr.last().copied().unwrap_or(0.0);
        let gene_tp = tp_arr.last().copied().unwrap_or(0.0);

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
            s.open_position_id = open_position.map(|(id, _)| id);
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
                if open_position.is_some() {
                    // Hold to bracket — the trailing block above keeps the
                    // broker-side stop in sync WHEN the exit policy arms it
                    // (otherwise the original SL/TP stands); nothing to
                    // execute this bar.
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

                // Live spread gate: the validated edge budgeted
                // `backtest_spread_pips` per round trip. If the CURRENT spread
                // is blown out (rollover, thin book, news aftermath), entering
                // pays costs the backtest never charged — skip the bar. Uses
                // the live tick cache; a stale/missing tick fails OPEN (never
                // blocks on our own data gap).
                {
                    let now_ms = chrono::Utc::now().timestamp_millis();
                    let tick = crate::app_services::live_spots::snapshot_all()
                        .into_iter()
                        .find(|t| t.symbol_name.eq_ignore_ascii_case(&symbol));
                    if let Some(t) = tick {
                        if now_ms - t.received_at_unix_ms <= 120_000 {
                            if let (Some(bid), Some(ask)) = (t.bid, t.ask) {
                                let pip = sym_meta
                                    .as_ref()
                                    .map(|m| m.pip_size)
                                    .filter(|p| p.is_finite() && *p > 0.0)
                                    .unwrap_or(0.0001);
                                let spread_pips = (ask - bid) / pip;
                                let limit = backtest_spread_pips * 2.5;
                                if spread_pips.is_finite() && spread_pips > limit {
                                    tracing::warn!(
                                        target: "neoethos_app::live_trading",
                                        %symbol, spread_pips, limit,
                                        "entry blocked — live spread far above the backtest's \
                                         cost assumption (skipping this bar)"
                                    );
                                    if let Ok(mut s) = status.lock() {
                                        s.last_signal = Some(format!(
                                            "blocked: spread {spread_pips:.1} pips > {limit:.1}"
                                        ));
                                    }
                                    continue;
                                }
                            }
                        }
                    }
                }

                // Open new position
                let side = if direction == Direction::Long {
                    OrderSide::Buy
                } else {
                    OrderSide::Sell
                };
                // Fresh account state at ENTRY time: (a) the balance compounds —
                // risky mode must size off what the account is NOW, not at engine
                // start; (b) the broker's live open-position count feeds the
                // portfolio-level risk budget. Fail-soft to start-time values.
                let (entry_balance, open_positions_now) = match tokio::task::spawn_blocking(
                    crate::app_services::broker_api::fetch_account_runtime_blocking,
                )
                .await
                {
                    Ok(Ok(rt)) => (rt.trader.balance, Some(rt.reconcile.positions.len())),
                    _ => (account_balance, None),
                };

                // ── Session circuit breakers (audit S03) — NEW ENTRIES only ──
                // Total-drawdown halt: balance at/below
                // initial_balance × (1 − total_drawdown_limit) stops every
                // further entry until restart (a blown account must not keep
                // trading itself deeper). Daily-loss stop: losing more than
                // daily_drawdown_limit of the day's starting balance blocks
                // entries until the next UTC day.
                let today: u32 = {
                    use chrono::Datelike;
                    let d = chrono::Utc::now().date_naive();
                    (d.year().max(0) as u32) * 10_000 + d.month() * 100 + d.day()
                };
                match day_start {
                    Some((d, _)) if d == today => {}
                    _ => day_start = Some((today, entry_balance)),
                }
                if total_dd_limit > 0.0 {
                    let floor = initial_balance_cfg * (1.0 - total_dd_limit);
                    if entry_balance <= floor {
                        if !total_tripped {
                            total_tripped = true;
                            tracing::error!(
                                target: "neoethos_app::live_trading",
                                %symbol, balance = entry_balance,
                                initial_balance = initial_balance_cfg,
                                limit = total_dd_limit,
                                "CIRCUIT BREAKER: total drawdown limit hit — ALL new \
                                 entries halted (exit management continues). Restart \
                                 the engine after reviewing the account."
                            );
                        }
                        if let Ok(mut s) = status.lock() {
                            s.last_signal =
                                Some("HALTED: total drawdown limit hit".to_string());
                        }
                        continue;
                    }
                }
                if daily_dd_limit > 0.0
                    && let Some((d, start_bal)) = day_start
                    && d == today
                    && start_bal > 0.0
                {
                    let floor = start_bal * (1.0 - daily_dd_limit);
                    if entry_balance <= floor {
                        if daily_tripped_on != Some(today) {
                            daily_tripped_on = Some(today);
                            tracing::warn!(
                                target: "neoethos_app::live_trading",
                                %symbol, balance = entry_balance,
                                day_start_balance = start_bal,
                                limit = daily_dd_limit,
                                "CIRCUIT BREAKER: daily loss limit hit — new entries \
                                 blocked until the next UTC day (exits continue)"
                            );
                        }
                        if let Ok(mut s) = status.lock() {
                            s.last_signal =
                                Some("blocked: daily loss limit (resumes next UTC day)".to_string());
                        }
                        continue;
                    }
                }
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
                //    clock `bridge.rs:240 auto_re_arm_if_ready` clears and the
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
                         previous entry hit a per-day / per-stage / per-month loss \
                         tier and started the 24h halt. No Risky-Mode entry will be \
                         sent until it elapses (the bridge auto re-arms; the Risk \
                         screen shows the remaining time). Exits and trailing \
                         continue normally."
                    );
                    if let Ok(mut s) = status.lock() {
                        s.last_signal = Some(format!(
                            "HALTED: risky-mode kill switch, auto re-arm in {}h {}m",
                            remaining / 3600,
                            (remaining % 3600) / 60
                        ));
                    }
                    continue;
                }

                // ── Daily entry cap (risk.max_trades_per_day) — same block as
                // the breakers above so the entry rules live together. The slot
                // is RESERVED here, before the order exists, so two engines
                // racing at count = cap−1 cannot both pass; every skip/failure
                // path between here and a filled order gives the slot back.
                // Disarmed (`daily_entry_cap = None`) this only counts.
                match ACCOUNT_DAILY_ENTRIES.try_reserve(today, daily_entry_cap) {
                    Ok(_) => {}
                    Err(refusal) => {
                        // Say WHICH rule fired and WHAT it compared — a refusal
                        // the operator cannot explain is a control he disables.
                        tracing::warn!(
                            target: "neoethos_app::live_trading",
                            %symbol,
                            rule = "risk.max_trades_per_day",
                            entries_today = refusal.count,
                            cap = refusal.cap,
                            utc_day = today,
                            "entry refused — account-wide daily trade cap \
                             reached (count is shared across every running \
                             engine; resumes next UTC day, exits continue)"
                        );
                        if let Ok(mut s) = status.lock() {
                            s.last_signal = Some(format!(
                                "blocked: max_trades_per_day {}/{} account-wide \
                                 (resumes next UTC day)",
                                refusal.count, refusal.cap
                            ));
                        }
                        continue;
                    }
                }

                // Base per-trade risk. In Risky Mode we size off the bankroll-
                // stage ladder (50 %→30 % as the account compounds, resolved
                // from the LIVE balance) rather than the static prop-firm
                // `risk_per_trade`. Strictly gated on `trading_mode == "risky"`;
                // the "prop_firm" path is unchanged. Falls back to the
                // configured fraction when the ladder inputs are degenerate —
                // never a wrong size.
                //
                // THEN CLAMPED (#209/#210, 2026-08-09) by
                // `risk.risky_max_risk_per_trade` when that is lower than the
                // rung. Before this the config value had no reader outside the
                // search, so the operator's written-down 0.30 and the ladder's
                // 0.50 disagreed with nothing reconciling them.
                let base_risk = if trading_mode_risky {
                    let ladder = neoethos_core::domain::risky_mode::stage_risk_fraction_for_bankroll(
                        risky_start_balance,
                        risky_target_balance,
                        neoethos_core::domain::risky_mode::DEFAULT_DOUBLING_FACTOR,
                        entry_balance,
                    )
                    .unwrap_or(risk_fraction);
                    // #209/#210 — the config's ceiling binds when it is lower
                    // than the rung the ladder chose. The disagreement was
                    // already logged loudly once at engine start; here we only
                    // record the per-entry effect, so the operator can see the
                    // rung he would have got next to the size he actually got.
                    let frac = match risky_configured_ceiling {
                        Some(cap) if ladder > cap => {
                            tracing::warn!(
                                target: "neoethos_app::live_trading",
                                %symbol, bankroll = entry_balance,
                                ladder_rung = ladder,
                                configured_ceiling = cap,
                                risk_pct = cap,
                                "risky-mode entry CLAMPED by \
                                 risk.risky_max_risk_per_trade — the ladder rung \
                                 is above the configured ceiling and the lower \
                                 number binds"
                            );
                            cap
                        }
                        _ => ladder,
                    };
                    tracing::info!(
                        target: "neoethos_app::live_trading",
                        %symbol, bankroll = entry_balance, risk_pct = frac,
                        ladder_rung = ladder,
                        "risky-mode stage sizing (bankroll-ladder, not the 3% prop cap)"
                    );
                    frac
                } else {
                    risk_fraction
                };

                // LIVE ML gate: the genes chose the direction above; the
                // ensemble may only SHRINK the size (agreement × regime ×
                // anomaly, MlScale mode) or skip the bar on a hard collapse.
                // Any ensemble error ⇒ loud log + unchanged gene-only sizing.
                let base_risk = if let Some(ens) = live_ensemble.as_deref() {
                    match neoethos_models::ensemble_inference::bootstrap::role_decision_for_last_row(
                        ens,
                        &raw_features,
                    ) {
                        Ok(d) => {
                            let ml = neoethos_trader::MlDecision {
                                dir_probs: d.dir_probs,
                                regime_gate: d.regime_gate,
                                anomaly_scale: d.anomaly_scale,
                            };
                            let cfg = neoethos_trader::BlendConfig {
                                mode: neoethos_trader::BlendMode::MlScale,
                                ..Default::default()
                            };
                            let (out_dir, conf) = neoethos_trader::blend_decision(direction, &ml, &cfg);
                            if matches!(out_dir, Direction::Flat) {
                                tracing::warn!(
                                    target: "neoethos_app::live_trading",
                                    %symbol,
                                    p_buy = d.dir_probs[1], p_sell = d.dir_probs[2],
                                    regime_gate = d.regime_gate, anomaly = d.anomaly_scale,
                                    "entry skipped — ML gate hard collapse (regime/anomaly veto)"
                                );
                                if let Ok(mut s) = status.lock() {
                                    s.last_signal = Some(format!(
                                        "skipped by ML gate (regime {:.2} × anomaly {:.2})",
                                        d.regime_gate, d.anomaly_scale
                                    ));
                                }
                                // No entry happened — give the reserved daily
                                // entry slot back (the cap counts entries, not
                                // attempts).
                                ACCOUNT_DAILY_ENTRIES.release(today);
                                continue;
                            }
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
                            base_risk * conf
                        }
                        Err(err) => {
                            tracing::warn!(
                                target: "neoethos_app::live_trading",
                                %symbol, error = %err,
                                "ML gate abstained (gene-only sizing this bar)"
                            );
                            base_risk
                        }
                    }
                } else {
                    base_risk
                };

                // Portfolio-level concurrent-risk budget (max_portfolio_risk):
                // remaining = cap − open_positions × base_risk. Skip the
                // entry when the budget is spent; size down when only part fits.
                let mut effective_risk = base_risk;
                if portfolio_risk_cap > 0.0 {
                    let open_n = open_positions_now.unwrap_or(0) as f64;
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
                            s.last_signal =
                                Some("blocked: portfolio risk budget spent".to_string());
                        }
                        // No entry happened — release the daily entry slot.
                        ACCOUNT_DAILY_ENTRIES.release(today);
                        continue;
                    }
                    effective_risk = base_risk.min(remaining);
                }

                // Size by the account's risk %, using the EFFECTIVE stop
                // distance actually placed on the order (gene SL / override /
                // default) so risk-per-trade matches the real bracket; falls
                // back to req.lot_size when not computable.
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
                let last_price = base_ohlcv.close.last().copied();
                let lot = risk_based_lots(
                    entry_balance,
                    effective_risk,
                    sl.unwrap_or(0.0),
                    sym_meta.as_ref(),
                    &account_ccy,
                    fx_quote_to_account,
                    last_price,
                    req.lot_size,
                    max_lot_cap,
                );

                // ── Risky Mode kill switch: THE PRE-SEND CHECK (W3) ──────────
                // The last thing before the order leaves the process. Checks
                // the sticky Manual / HardwareConnLoss halts (Manual still has
                // no producer and cannot arm; HardwareConnLoss gained one on
                // 2026-08-09 — see `tier_halts_for_24h`), plus
                // per-trade bracket validity, the pre-send sanity ceiling
                // (55% of bankroll), the per-day loss cap, the per-stage
                // retreat trigger and the per-month cap.
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
                            meta.pip_value_in_account(
                                &account_ccy,
                                fx_quote_to_account,
                                last_price,
                            )
                        })
                        .filter(|v| v.is_finite() && *v > 0.0);
                    let Some(pip_val) = pip_val else {
                        // FAIL CLOSED. No pip value ⇒ `risk_based_lots` already
                        // fell back to the fixed `req.lot_size`, so this entry
                        // is NOT the ladder size the operator authorised, AND
                        // the pre-send ceiling cannot be evaluated. Sending a
                        // position whose risk we cannot price, in the mode that
                        // risks 30-50% per trade, is exactly the shape this
                        // gate exists to refuse.
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
                             30-50% stage size nor the pre-send ceiling can be \
                             computed. Add the symbol to symbol_metadata.json or \
                             fix the quote->account FX rate. (prop_firm mode is \
                             unaffected by this rule.)"
                        );
                        if let Ok(mut s) = status.lock() {
                            s.last_signal = Some(
                                "blocked: risky-mode cannot price pip value for this symbol"
                                    .to_string(),
                            );
                        }
                        ACCOUNT_DAILY_ENTRIES.release(today);
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
                        let at_broker_min_lot = broker_min_lot
                            .is_some_and(|min| lot <= min * 1.000_001);
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
                            monthly_loss = m.monthly_loss_accumulated_usd(),
                            stage_idx = m.current_stage().stage_idx,
                            "ENTRY REFUSED BY THE RISKY MODE KILL SWITCH"
                        );
                        if halts_for_24h {
                            // Persist it, so the halt survives a restart and the
                            // Risk screen's cooldown row finally means something.
                            if let Err(e) =
                                crate::app_services::risky_mode_persistence::record_kill_switch_trip()
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
                        ACCOUNT_DAILY_ENTRIES.release(today);
                        continue;
                    }
                }

                let sym = symbol.clone();

                let result = match tokio::task::spawn_blocking(move || {
                    // LAST GUARD BEFORE REAL MONEY (2026-08-09, closed
                    // 2026-08-09 second pass). The top-of-iteration environment
                    // check ran many seconds ago — before the bar fetch with
                    // its retries and the ML pass. Passing the admitted
                    // environment down makes `resolve_creds` validate it
                    // against the SAME file read that produces the routing
                    // environment for this order, so a Demo->Live flip cannot
                    // land between the check and the send. The previous version
                    // used a separate `assert_environment()` call, which still
                    // left two reads with a gap between them.
                    submit_market_order_blocking(
                        &sym,
                        side,
                        lot,
                        sl,
                        tp,
                        Some("NeoEthos-Auto".to_string()),
                        Some(gated_env_is_live),
                    )
                })
                .await
                {
                    Ok(r) => r,
                    Err(join_err) => {
                        // The submit task panicked/was cancelled — no entry
                        // happened, so free the daily entry slot before the
                        // engine dies with the error.
                        ACCOUNT_DAILY_ENTRIES.release(today);
                        return Err(join_err.into());
                    }
                };

                match result {
                    Ok(outcome) => {
                        // Derive broker wire volume for future close.
                        // volume_to_units(raw) = raw / 100  →  raw = lot_size × 100.
                        // This reversal exactly matches the execution event parser.
                        let broker_vol = outcome
                            .lot_size
                            .map(|ls| (ls * 100.0).round() as i64)
                            .or_else(|| outcome.filled_lot_size.map(|ls| (ls * 100.0).round() as i64))
                            .unwrap_or(1); // 1 = absolute minimum; broker rejects 0

                        if let Some(pos_id) = outcome.position_id {
                            open_position = Some((pos_id, broker_vol));
                            // Track for auto-cull realized-result reconciliation.
                            opened_ids.insert(pos_id);
                            // Seed trailing-stop state (parity with the backtest):
                            // entry, the EFFECTIVE stop distance (the same one the
                            // kernel trails with — gene SL, operator override, or
                            // the 20-pip default), side, running extreme. Seeded
                            // unconditionally even when `models.exit_policy` has
                            // trailing OFF, so arming the policy never finds a
                            // half-initialised position.
                            pos_entry_px = outcome.execution_price.or(last_price).unwrap_or(0.0);
                            pos_sl_pips = sl.unwrap_or(0.0);
                            pos_is_long = direction == Direction::Long;
                            pos_extreme = pos_entry_px;
                            pos_trail_px = 0.0;
                            // Experience snapshot: the exact feature row this
                            // entry acted on (paired with the outcome at close).
                            let feat_row: Vec<f32> = {
                                let ns = aligned.n_samples();
                                if ns > 0 {
                                    aligned.sample_window(ns - 1, 1).iter().copied().collect()
                                } else {
                                    Vec::new()
                                }
                            };
                            pending_experience.insert(
                                pos_id,
                                crate::app_services::experience_store::LiveExperience {
                                    schema_version: 1,
                                    position_id: pos_id,
                                    symbol: symbol.clone(),
                                    base_tf: base_tf.clone(),
                                    portfolio_path: portfolio_path.clone(),
                                    direction: if pos_is_long { 1 } else { -1 },
                                    // The EFFECTIVE brackets placed on the order
                                    // (gene / override / kernel default) — what
                                    // actually governed this trade's exit.
                                    sl_pips: sl.unwrap_or(0.0),
                                    tp_pips: tp.unwrap_or(0.0),
                                    lots: lot,
                                    entry_ts_ms: latest_ts,
                                    entry_price: outcome.execution_price.or(last_price),
                                    features: feat_row,
                                    close_ts_ms: None,
                                    net_profit: None,
                                },
                            );
                        }

                        if let Ok(mut s) = status.lock() {
                            s.open_position_id = open_position.map(|(id, _)| id);
                        }

                        tracing::info!(
                            target: "neoethos_app::live_trading",
                            side = ?side,
                            position_id = ?open_position.map(|(id, _)| id),
                            fill_price = ?outcome.execution_price,
                            "order placed"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "neoethos_app::live_trading",
                            error = %e,
                            side = ?side,
                            "order placement failed"
                        );
                        // The broker refused the order — no entry happened, so
                        // the daily entry slot goes back (the cap counts
                        // ENTRIES on the account, not attempts).
                        ACCOUNT_DAILY_ENTRIES.release(today);
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
        s.open_position_id = None;
    }

    tracing::info!(target: "neoethos_app::live_trading", "live trading loop exited");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::journal_store::ClosedTrade;
    use neoethos_core::domain::risky_mode as rm;

    #[test]
    fn account_level_tiers_halt_for_24h_and_order_level_tiers_do_not() {
        use rm::KillSwitchTier as T;
        for tier in [
            T::PerDay,
            T::PerStage,
            T::PerMonth,
            T::Manual,
            T::HardwareConnLoss,
        ] {
            assert!(
                tier_halts_for_24h(tier),
                "{tier:?} is a bankroll-level event and must start the persisted halt"
            );
        }
        for tier in [
            T::PerTrade,
            T::PreSendSanity,
            T::ManualOrderWhileAutonomousOnly,
        ] {
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
            autonomous_only_contract_accepted: true,
            allow_live_broker: true,
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
                - rm::RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION)
                .abs()
                < 1e-9,
            "stage 0 of the shipped ladder is the 50% rung"
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
            closed(-10.0, Some("A"), NOW_MS - 3_600_000),  // today
            closed(-20.0, Some("A"), NOW_MS - 2 * day),    // this ISO week
            closed(-40.0, Some("A"), NOW_MS - 7 * day),    // 2 Aug: this month, BEFORE Monday 3 Aug
            closed(-80.0, Some("A"), NOW_MS - 60 * day),   // a previous month
            closed(500.0, Some("A"), NOW_MS - 3_600_000),  // a WIN — never counted
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
