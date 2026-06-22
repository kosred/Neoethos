//! cTrader broker commands — reuse the EXISTING tested `neoethos-app`
//! app_services in-process. No second process, no HTTP, no port. Each command
//! is a thin wrapper that runs the blocking broker call on a worker thread and
//! returns the value straight to the web UI.
//!
//! Auth is automatic: `broker_api` resolves credentials through the silent
//! token-refresh path, so after the one-time OAuth the broker just works.

use serde::Serialize;
use tauri::async_runtime::spawn_blocking;

use neoethos_app::app_services::broker_api;
use neoethos_app::app_services::broker_persistence::load_broker_settings;
use neoethos_app::app_services::secure_store::production_ctrader_token_store;
use neoethos_app::app_services::{live_spots, live_spots_streamer};

// ── DTOs (camelCase for the web UI) ───────────────────────────────────────────

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerStatus {
    pub configured: bool,
    pub has_token: bool,
    pub environment: String,
    pub account_id: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Candle {
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SymbolInfo {
    pub symbol_id: i64,
    pub name: String,
    pub enabled: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountInfo {
    pub account_id: String,
    pub broker_title: String,
    pub account_name: String,
    pub is_live: Option<bool>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Position {
    pub position_id: i64,
    pub symbol_id: i64,
    pub side: String,
    pub volume: f64,
    pub price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountSnapshot {
    pub account_id: i64,
    pub balance: f64,
    pub equity: f64,
    pub unrealized_pnl: f64,
    pub currency: String,
    pub open_positions: usize,
    pub positions: Vec<Position>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecResult {
    pub status: String,
    pub order_id: Option<i64>,
    pub position_id: Option<i64>,
    pub deal_id: Option<i64>,
    pub side: Option<String>,
    pub fill_price: Option<f64>,
    pub message: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpotPrice {
    pub symbol_id: i64,
    pub name: String,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub mid: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReauthResult {
    pub callback_port: u16,
    pub refresh_token_present: bool,
    pub access_token_len: usize,
    pub message: String,
}

fn asset_currency(asset_id: Option<i64>) -> String {
    match asset_id {
        Some(4) => "GBP",
        Some(5) => "CHF",
        Some(6) => "EUR",
        Some(8) => "USD",
        Some(14) => "JPY",
        Some(23) => "AUD",
        Some(25) => "NZD",
        Some(27) => "CAD",
        _ => "EUR",
    }
    .to_string()
}

// ── Commands ──────────────────────────────────────────────────────────────────

/// Cheap status (no network): is the broker configured + is a token stored.
#[tauri::command]
pub async fn broker_status() -> Result<BrokerStatus, String> {
    spawn_blocking(|| {
        let s = load_broker_settings();
        let ct = &s.ctrader;
        let configured = !ct.client_id.is_empty() && !ct.client_secret.is_empty();
        let has_token = production_ctrader_token_store()
            .load_token_bundle_with_legacy_fallback()
            .ok()
            .flatten()
            .map(|b| !b.access_token.is_empty())
            .unwrap_or(false);
        let environment = format!("{:?}", ct.environment);
        // Report the account the broker calls actually use (enabled-for-execution
        // first, matching resolve_creds), not just accounts.first().
        let account_id = ct
            .accounts
            .iter()
            .find(|a| a.enabled_for_execution)
            .or_else(|| ct.accounts.first())
            .map(|a| a.account_id.clone());
        BrokerStatus { configured, has_token, environment, account_id }
    })
    .await
    .map_err(|e| e.to_string())
}

/// Candles straight from the broker (`ProtoOAGetTrendbarsReq`).
#[tauri::command]
pub async fn broker_chart(
    symbol: String,
    timeframe: String,
    limit: Option<usize>,
) -> Result<Vec<Candle>, String> {
    spawn_blocking(move || {
        let bars = broker_api::fetch_recent_chart_bars_blocking(
            &symbol,
            &timeframe,
            limit.unwrap_or(1000),
        )
        .map_err(|e| e.to_string())?;
        let mut out = Vec::with_capacity(bars.len());
        let mut last = i64::MIN;
        for b in bars {
            let t = b.timestamp_ms / 1000;
            if t <= last {
                continue;
            }
            last = t;
            out.push(Candle { time: t, open: b.open, high: b.high, low: b.low, close: b.close });
        }
        Ok::<Vec<Candle>, String>(out)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Broker symbol catalog.
#[tauri::command]
pub async fn broker_symbols() -> Result<Vec<SymbolInfo>, String> {
    spawn_blocking(|| {
        let bundle = broker_api::fetch_broker_symbols_blocking().map_err(|e| e.to_string())?;
        Ok::<Vec<SymbolInfo>, String>(
            bundle
                .symbols
                .into_iter()
                .map(|s| SymbolInfo { symbol_id: s.symbol_id, name: s.symbol_name, enabled: s.enabled })
                .collect(),
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Accounts granted by the OAuth token.
#[tauri::command]
pub async fn broker_accounts() -> Result<Vec<AccountInfo>, String> {
    spawn_blocking(|| {
        let bundle = broker_api::fetch_broker_accounts_blocking().map_err(|e| e.to_string())?;
        Ok::<Vec<AccountInfo>, String>(
            bundle
                .accounts
                .into_iter()
                .map(|a| AccountInfo {
                    account_id: a.account_id,
                    broker_title: a.broker_title,
                    account_name: a.account_name,
                    is_live: a.is_live,
                })
                .collect(),
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Live account snapshot: balance, equity, open positions.
#[tauri::command]
pub async fn account_snapshot() -> Result<AccountSnapshot, String> {
    spawn_blocking(|| {
        let snap = broker_api::fetch_account_runtime_blocking().map_err(|e| e.to_string())?;
        let positions: Vec<Position> = snap
            .reconcile
            .positions
            .into_iter()
            .map(|p| Position {
                position_id: p.position_id,
                symbol_id: p.symbol_id,
                side: p.trade_side,
                volume: p.volume,
                price: p.price,
                stop_loss: p.stop_loss,
                take_profit: p.take_profit,
            })
            .collect();
        Ok::<AccountSnapshot, String>(AccountSnapshot {
            account_id: snap.trader.account_id,
            balance: snap.trader.balance,
            equity: snap.trader.balance + snap.trader.unrealized_pnl,
            unrealized_pnl: snap.trader.unrealized_pnl,
            currency: asset_currency(snap.trader.deposit_asset_id),
            open_positions: positions.len(),
            positions,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Submit a market order. `side` is "buy" or "sell".
#[tauri::command]
pub async fn place_order(
    symbol: String,
    side: String,
    volume_lots: f64,
    stop_loss_pips: Option<f64>,
    take_profit_pips: Option<f64>,
) -> Result<ExecResult, String> {
    spawn_blocking(move || {
        let order_side = match side.to_ascii_lowercase().as_str() {
            "buy" => broker_api::OrderSide::Buy,
            "sell" => broker_api::OrderSide::Sell,
            other => return Err(format!("invalid side '{other}' (expected buy/sell)")),
        };
        let outcome = broker_api::submit_market_order_blocking(
            &symbol,
            order_side,
            volume_lots,
            stop_loss_pips,
            take_profit_pips,
            Some("NeoEthos".to_string()),
        )
        .map_err(|e| e.to_string())?;
        Ok::<ExecResult, String>(ExecResult {
            status: format!("{:?}", outcome.status),
            order_id: outcome.order_id,
            position_id: outcome.position_id,
            deal_id: outcome.deal_id,
            side: outcome.trade_side,
            fill_price: outcome.execution_price,
            message: outcome.description.unwrap_or_default(),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Close (or partial-close) a position. `volume` is in cTrader wire units.
#[tauri::command]
pub async fn close_position(position_id: i64, volume: i64) -> Result<ExecResult, String> {
    spawn_blocking(move || {
        let outcome =
            broker_api::close_position_blocking(position_id, volume).map_err(|e| e.to_string())?;
        Ok::<ExecResult, String>(ExecResult {
            status: format!("{:?}", outcome.status),
            order_id: outcome.order_id,
            position_id: outcome.position_id,
            deal_id: outcome.deal_id,
            side: outcome.trade_side,
            fill_price: outcome.execution_price,
            message: outcome.description.unwrap_or_default(),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Start the live spot-price streamer (cTrader WebSocket). Idempotent-ish:
/// safe to call once at startup. Returns whether it spawned (false = creds /
/// token missing — the UI then shows no live prices until re-auth).
pub fn start_spot_streamer() {
    // Run in the tokio context (the streamer does an internal tokio::spawn for
    // its reconnect loop) but off the UI thread.
    tauri::async_runtime::spawn(async {
        let spawned = live_spots_streamer::try_spawn_with_defaults_blocking();
        log::info!("spot streamer spawned: {spawned}");
    });
}

/// Snapshot of the latest live bid/ask per subscribed symbol (the UI polls
/// this every ~1.5 s — plenty for forex). Populated by the streamer above.
#[tauri::command]
pub async fn spot_prices() -> Result<Vec<SpotPrice>, String> {
    spawn_blocking(|| {
        live_spots::snapshot_all()
            .into_iter()
            .map(|t| {
                let mid = match (t.bid, t.ask) {
                    (Some(b), Some(a)) => Some((b + a) / 2.0),
                    (Some(b), None) => Some(b),
                    (None, Some(a)) => Some(a),
                    _ => None,
                };
                SpotPrice { symbol_id: t.symbol_id, name: t.symbol_name, bid: t.bid, ask: t.ask, mid }
            })
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|e| e.to_string())
}

/// One-time interactive OAuth (opens the browser). After this the silent
/// refresh keeps the session alive automatically — never needed again unless
/// the broker revokes the refresh token.
#[tauri::command]
pub async fn reauth_broker() -> Result<ReauthResult, String> {
    spawn_blocking(|| {
        let o = neoethos_app::app_services::reauth::run_reauth_flow_blocking()
            .map_err(|e| e.to_string())?;
        Ok::<ReauthResult, String>(ReauthResult {
            callback_port: o.callback_port,
            refresh_token_present: o.refresh_token_present,
            access_token_len: o.access_token_len,
            message: o.message,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}
