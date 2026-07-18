//! Live autonomous trading service (Path A).
//!
//! Polls the broker for new closed bars, computes features, evaluates gene
//! signals, and places/closes orders via cTrader.  Uses the exact same
//! pipeline as `neoethos_trader::replay_portfolio_from_dir` so live signals
//! are byte-identical to the offline backtest — the core parity mandate.
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
    if crate::app_services::live_gate::active_env_is_live() {
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
        if let Err(e) = run(req, stop_clone, status_clone.clone()).await {
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

    // ── Trailing-stop parity (discovery hardcodes break-even + trailing ALWAYS
    // ON: BE at +1R, then trail 1×SL behind the running extreme). The backtest
    // measured every strategy's edge WITH this; live MUST replicate it or trades
    // the backtest saved at break-even become full losses. State per open pos:
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
    // Weekend kill zones — PARITY with the backtest (eval.rs): discovery runs
    // with kill_zones_enabled from the SAME config flag, force-closing before
    // the weekend and blocking Fri-late/Mon-open entries. Live must match or
    // positions ride weekend gaps no validated strategy ever held through.
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
                tracing::info!(
                    target: "neoethos_app::live_trading",
                    %symbol, %base_tf,
                    loaded = outcome.loaded_count(),
                    missing = outcome.missing_count(),
                    degraded = outcome.degraded_count(),
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
                        close_position_blocking(pos_id, vol)
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
                            close_position_blocking(pos_id, vol)
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
        // eval.rs hardcodes break-even + trailing ALWAYS ON: once the favorable
        // move reaches +1R (= sl_pips) the stop trails 1×SL behind the running
        // extreme (which is exactly break-even at +1R). Ratchet-only; no intra-bar
        // look-ahead (we act on the just-closed bar, the broker enforces it next).
        // Without this, trades the backtest saved at break-even become full losses.
        if let Some((pos_id, _)) = open_position {
            if pos_sl_pips > 0.0 && pos_entry_px > 0.0 {
                let pip = sym_meta
                    .as_ref()
                    .map(|m| m.pip_size)
                    .filter(|p| p.is_finite() && *p > 0.0)
                    .unwrap_or(0.0001);
                let r_dist = pos_sl_pips * pip; // 1R in price units
                let hi = base_ohlcv.high.last().copied().unwrap_or(pos_entry_px);
                let lo = base_ohlcv.low.last().copied().unwrap_or(pos_entry_px);
                let mut new_trail: Option<f64> = None;
                if pos_is_long {
                    pos_extreme = pos_extreme.max(hi);
                    if pos_extreme - pos_entry_px >= r_dist {
                        let candidate = pos_extreme - r_dist;
                        if pos_trail_px == 0.0 || candidate > pos_trail_px {
                            pos_trail_px = candidate;
                            new_trail = Some(candidate);
                        }
                    }
                } else {
                    pos_extreme = if pos_extreme > 0.0 { pos_extreme.min(lo) } else { lo };
                    if pos_entry_px - pos_extreme >= r_dist {
                        let candidate = pos_extreme + r_dist;
                        if pos_trail_px == 0.0 || candidate < pos_trail_px {
                            pos_trail_px = candidate;
                            new_trail = Some(candidate);
                        }
                    }
                }
                if let Some(raw) = new_trail {
                    let sl_price = (raw / pip).round() * pip; // snap to a broker-valid pip grid
                    let _ = tokio::task::spawn_blocking(move || {
                        amend_position_sltp_blocking(pos_id, Some(sl_price), None, None)
                    })
                    .await;
                    tracing::info!(
                        target: "neoethos_app::live_trading",
                        position_id = pos_id, new_sl = sl_price, extreme = pos_extreme,
                        "trailing stop advanced (parity: BE at +1R, then trail 1×SL)"
                    );
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
            match neoethos_data::prepare_multitimeframe_features(&dataset, &base_tf, &higher_refs, None) {
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
        let (directions, sl_arr, tp_arr) =
            neoethos_trader::combine_gene_signals_with_brackets(&genes, &aligned, &base_ohlcv);
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
        // signal is ignored entirely — exits happen exclusively via SL/TP/
        // trailing (broker-enforced live) plus the weekend force-close; the
        // production EvaluationConfig ships max_hold_bars = 0. The previous
        // live code closed + reopened on EVERY non-flat bar (paying the
        // spread per bar and resetting the trailing state) and closed on a
        // Flat signal — a trade profile no validated backtest ever had.
        match direction {
            Direction::Long | Direction::Short => {
                if open_position.is_some() {
                    // Hold to bracket — the trailing block above keeps the
                    // broker-side stop in sync; nothing to execute this bar.
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

                // Base per-trade risk. In Risky Mode we size off the bankroll-
                // stage ladder (30 %→50 % by stage, resolved from the account's
                // LIVE balance so it tapers as it compounds) rather than the
                // static prop-firm `risk_per_trade`. Strictly gated on
                // `trading_mode == "risky"`; the "prop_firm" path is unchanged.
                // Falls back to the configured fraction when the ladder inputs
                // are degenerate — never a wrong size.
                let base_risk = if trading_mode_risky {
                    let frac = neoethos_core::domain::risky_mode::stage_risk_fraction_for_bankroll(
                        risky_start_balance,
                        risky_target_balance,
                        neoethos_core::domain::risky_mode::DEFAULT_DOUBLING_FACTOR,
                        entry_balance,
                    )
                    .unwrap_or(risk_fraction);
                    tracing::info!(
                        target: "neoethos_app::live_trading",
                        %symbol, bankroll = entry_balance, risk_pct = frac,
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
                let sym = symbol.clone();

                let result = tokio::task::spawn_blocking(move || {
                    submit_market_order_blocking(
                        &sym,
                        side,
                        lot,
                        sl,
                        tp,
                        Some("NeoEthos-Auto".to_string()),
                    )
                })
                .await?;

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
                            // the 20-pip default), side, running extreme.
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
