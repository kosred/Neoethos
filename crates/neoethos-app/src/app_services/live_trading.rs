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
    OrderSide, close_position_blocking, fetch_recent_chart_bars_blocking,
    submit_market_order_blocking,
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
}

pub fn default_lot_size() -> f64 {
    0.01
}
pub fn default_warmup_bars() -> usize {
    1000
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

fn bars_to_ohlcv(bars: &[crate::app_services::ctrader_data::HistoricalBar]) -> Ohlcv {
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
    let max_lot_cap = sizing
        .as_ref()
        .map(|s| s.risk.max_lot_size)
        .filter(|v| *v > 0.0)
        .unwrap_or(f64::INFINITY);
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

    loop {
        if stop.load(Ordering::Relaxed) {
            tracing::info!(target: "neoethos_app::live_trading", "stop requested");
            break;
        }

        // Sleep until just after the next bar boundary
        let now_ms = chrono::Utc::now().timestamp_millis();
        let next_boundary = (now_ms / bar_ms + 1) * bar_ms;
        let wait_ms = (next_boundary - now_ms + 3_000).max(5_000) as u64;
        tracing::debug!(
            target: "neoethos_app::live_trading",
            wait_secs = wait_ms / 1000,
            "waiting for next bar"
        );
        tokio::time::sleep(Duration::from_millis(wait_ms)).await;

        if stop.load(Ordering::Relaxed) {
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

        // ── Build multi-TF SymbolDataset ──────────────────────────────────────
        let mut frames: HashMap<String, Ohlcv> = HashMap::new();
        let base_ohlcv = bars_to_ohlcv(&base_bars);
        frames.insert(base_tf.clone(), base_ohlcv.clone());

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
        match direction {
            Direction::Long | Direction::Short => {
                // Flip: close any open position first
                if let Some((pos_id, vol)) = open_position.take() {
                    let result = tokio::task::spawn_blocking(move || {
                        close_position_blocking(pos_id, vol)
                    })
                    .await?;
                    match result {
                        Ok(_) => tracing::info!(
                            target: "neoethos_app::live_trading",
                            position_id = pos_id, "closed previous position"
                        ),
                        Err(e) => tracing::warn!(
                            target: "neoethos_app::live_trading",
                            error = %e, position_id = pos_id,
                            "failed to close previous position"
                        ),
                    }
                }

                // Open new position
                let side = if direction == Direction::Long {
                    OrderSide::Buy
                } else {
                    OrderSide::Sell
                };
                // Size by the account's risk %, using the strategy's own stop
                // distance; falls back to req.lot_size when not computable.
                let last_price = base_ohlcv.close.last().copied();
                let lot = risk_based_lots(
                    account_balance,
                    risk_fraction,
                    gene_sl,
                    sym_meta.as_ref(),
                    &account_ccy,
                    fx_quote_to_account,
                    last_price,
                    req.lot_size,
                    max_lot_cap,
                );
                // Default to the strategy's OWN bracket; `req.*` is only an
                // explicit operator override (Autopilot sends none, so the
                // gene's discovered SL/TP is what actually gets placed).
                let sl = req.stop_loss_pips.or((gene_sl > 0.0).then_some(gene_sl));
                let tp = req.take_profit_pips.or((gene_tp > 0.0).then_some(gene_tp));
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
                // Close any open position
                if let Some((pos_id, vol)) = open_position.take() {
                    let result = tokio::task::spawn_blocking(move || {
                        close_position_blocking(pos_id, vol)
                    })
                    .await?;
                    match result {
                        Ok(_) => tracing::info!(
                            target: "neoethos_app::live_trading",
                            position_id = pos_id,
                            "closed position on Flat signal"
                        ),
                        Err(e) => tracing::warn!(
                            target: "neoethos_app::live_trading",
                            error = %e, position_id = pos_id,
                            "failed to close on Flat"
                        ),
                    }

                    if let Ok(mut s) = status.lock() {
                        s.open_position_id = None;
                    }
                }
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
