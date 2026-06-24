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
use neoethos_app::app_services::live_spots_streamer;

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
pub struct AccountInfo {
    pub account_id: String,
    pub broker_title: String,
    pub account_name: String,
    pub is_live: Option<bool>,
    pub login: Option<i64>,
    pub enabled: bool,
    /// e.g. "DEMO · Spotware · login 5789955".
    pub label: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Position {
    pub position_id: i64,
    pub symbol_id: i64,
    pub side: String,
    /// Human-readable volume (scaled units) for display.
    pub volume: f64,
    /// Raw cTrader WIRE volume = `volume * 100` — this is what the close
    /// endpoint wants, and it is guaranteed a multiple of the symbol's
    /// volumeStep. Mirrors the server bridge's computation (bridge.rs).
    /// Passing the scaled `volume` to close is what caused TRADING_BAD_VOLUME
    /// ("closeVolume 1170 not a multiple of volumeStep 117000").
    pub volume_units: i64,
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
    // Rich account identity (auto-detected from the broker):
    pub live: bool,
    pub broker_name: Option<String>,
    pub leverage: Option<f64>,
    pub login: Option<i64>,
    pub account_type: Option<String>,
    /// Friendly one-line descriptor, e.g. "LIVE · FTMO · 200k USD · 1:30".
    pub label: String,
}

/// "200000" → "200k", "1000" → "1k", "1234567" → "1.2M", "950" → "950".
fn human_size(balance: f64, currency: &str) -> String {
    let v = balance.abs();
    let num = if v >= 1_000_000.0 {
        format!("{:.1}M", v / 1_000_000.0)
    } else if v >= 1_000.0 {
        let k = v / 1_000.0;
        if (k - k.round()).abs() < 0.05 {
            format!("{}k", k.round() as i64)
        } else {
            format!("{:.1}k", k)
        }
    } else {
        format!("{:.0}", v)
    };
    format!("{num} {currency}")
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

/// Accounts granted by the OAuth token.
#[tauri::command]
pub async fn broker_accounts() -> Result<Vec<AccountInfo>, String> {
    spawn_blocking(|| {
        let bundle = broker_api::fetch_broker_accounts_blocking().map_err(|e| e.to_string())?;
        // Which account is currently active for execution (from config)?
        let enabled_id = load_broker_settings()
            .ctrader
            .accounts
            .iter()
            .find(|a| a.enabled_for_execution)
            .map(|a| a.account_id.clone());
        Ok::<Vec<AccountInfo>, String>(
            bundle
                .accounts
                .into_iter()
                .map(|a| {
                    let kind = match a.is_live {
                        Some(true) => "LIVE",
                        Some(false) => "DEMO",
                        None => "?",
                    };
                    let label = format!(
                        "{kind} · {} · login {}",
                        a.broker_title,
                        a.trader_login.map(|l| l.to_string()).unwrap_or_else(|| "—".into())
                    );
                    let enabled = enabled_id.as_deref() == Some(a.account_id.as_str());
                    AccountInfo {
                        account_id: a.account_id,
                        broker_title: a.broker_title,
                        account_name: a.account_name,
                        is_live: a.is_live,
                        login: a.trader_login,
                        enabled,
                        label,
                    }
                })
                .collect(),
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Make `account_id` the active execution account, and set the broker
/// environment to match its type (live ⇒ Live endpoint, demo ⇒ Demo). Writes
/// broker_credentials.toml. Account-scoped calls pick it up immediately (they
/// resolve creds fresh); the spot streamer re-subscribes on next launch.
#[tauri::command]
pub async fn select_account(
    account_id: String,
    live: bool,
    label: Option<String>,
) -> Result<BrokerStatus, String> {
    spawn_blocking(move || {
        use neoethos_core::broker_config::{
            BrokerAccountTarget, CTraderBrokerEnvironment, credentials_file_path, load_from_disk,
            save_to_disk,
        };
        let path = credentials_file_path().map_err(|e| e.to_string())?;
        let mut state = load_from_disk(&path)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "no broker_credentials.toml — authenticate first".to_string())?;

        if !state.ctrader.accounts.iter().any(|a| a.account_id == account_id) {
            state.ctrader.accounts.push(BrokerAccountTarget {
                account_id: account_id.clone(),
                label: label.unwrap_or_else(|| account_id.clone()),
                enabled_for_execution: false,
            });
        }
        for a in state.ctrader.accounts.iter_mut() {
            a.enabled_for_execution = a.account_id == account_id;
        }
        // chosen account first (stable: keeps the rest in order)
        state
            .ctrader
            .accounts
            .sort_by_key(|a| u8::from(a.account_id != account_id));
        state.ctrader.environment = if live {
            CTraderBrokerEnvironment::Live
        } else {
            CTraderBrokerEnvironment::Demo
        };
        save_to_disk(&path, &state).map_err(|e| e.to_string())?;

        let configured =
            !state.ctrader.client_id.is_empty() && !state.ctrader.client_secret.is_empty();
        let has_token = production_ctrader_token_store()
            .load_token_bundle_with_legacy_fallback()
            .ok()
            .flatten()
            .map(|b| !b.access_token.is_empty())
            .unwrap_or(false);
        Ok::<BrokerStatus, String>(BrokerStatus {
            configured,
            has_token,
            environment: format!("{:?}", state.ctrader.environment),
            account_id: Some(account_id),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Live account snapshot: balance, equity, open positions.
#[tauri::command]
pub async fn account_snapshot() -> Result<AccountSnapshot, String> {
    spawn_blocking(|| {
        // Live/Demo is determined by the connected environment.
        let live = format!("{:?}", load_broker_settings().ctrader.environment)
            .eq_ignore_ascii_case("live");
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
                volume_units: (p.volume * 100.0).round() as i64,
                price: p.price,
                stop_loss: p.stop_loss,
                take_profit: p.take_profit,
            })
            .collect();
        let currency = asset_currency(snap.trader.deposit_asset_id);
        let broker = snap.trader.broker_name.clone().unwrap_or_else(|| "cTrader".to_string());
        // Auto-built descriptor, e.g. "LIVE · FTMO · 200k USD · 1:30".
        let mut label = format!(
            "{} · {} · {}",
            if live { "LIVE" } else { "DEMO" },
            broker,
            human_size(snap.trader.balance, &currency)
        );
        if let Some(lev) = snap.trader.leverage {
            if lev > 0.0 {
                label.push_str(&format!(" · 1:{}", lev.round() as i64));
            }
        }
        Ok::<AccountSnapshot, String>(AccountSnapshot {
            account_id: snap.trader.account_id,
            balance: snap.trader.balance,
            equity: snap.trader.balance + snap.trader.unrealized_pnl,
            unrealized_pnl: snap.trader.unrealized_pnl,
            currency,
            open_positions: positions.len(),
            positions,
            live,
            broker_name: snap.trader.broker_name,
            leverage: snap.trader.leverage,
            login: snap.trader.trader_login,
            account_type: snap.trader.account_type,
            label,
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
