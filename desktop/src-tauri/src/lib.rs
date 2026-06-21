//! NeoEthos desktop (Tauri) — single-process Rust core + web UI.
//!
//! PoC commands read the operator's on-disk Vortex data directly through the
//! existing `neoethos-data` / `neoethos-core` crates — NO separate backend
//! process, NO HTTP, NO port 7423, NO supervisor/watchdog. The whole class of
//! Flutter+HTTP bugs (spawn spirals, /healthz timeouts, SSE EventFluxException,
//! config seeded to a different dir) cannot exist here: one process, one CWD.

use std::path::PathBuf;

use serde::Serialize;

mod broker;

/// Resolve the data root the same way the engine does: the operator's
/// `config.yaml` `system.data_dir`. Falls back to the dev repo path if the
/// configured dir is missing (PoC convenience).
fn resolve_data_root() -> PathBuf {
    if let Ok(s) = neoethos_core::Settings::load() {
        let d = s.system.data_dir.clone();
        if d.exists() {
            return d;
        }
    }
    PathBuf::from(r"C:\Users\konst\development\forex-ai\data")
}

#[derive(Serialize)]
struct AppInfo {
    version: String,
    data_root: String,
    data_root_exists: bool,
}

/// App identity + resolved data root (shown in the status bar so the operator
/// always knows which dataset the charts come from).
#[tauri::command]
fn app_info() -> AppInfo {
    let root = resolve_data_root();
    AppInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        data_root_exists: root.exists(),
        data_root: root.display().to_string(),
    }
}

/// Symbols present on disk (e.g. EURUSD, GBPUSD, XAUUSD …).
#[tauri::command]
async fn list_symbols() -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(|| {
        let root = resolve_data_root();
        neoethos_data::discover_symbols(&root).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Timeframes available for a symbol (M1, M5, H1, …), ordered.
#[tauri::command]
async fn list_timeframes(symbol: String) -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = resolve_data_root();
        neoethos_data::discover_timeframes(&root, &symbol).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// One OHLC bar in the shape TradingView Lightweight Charts wants:
/// `time` is a UTC timestamp in SECONDS (Vortex stores ms → /1000).
#[derive(Serialize)]
struct Candle {
    time: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

/// Trailing `limit` candles for (symbol, timeframe), read straight from the
/// Vortex file. Returned ascending by time (the loader normalises order).
#[tauri::command]
async fn chart(
    symbol: String,
    timeframe: String,
    limit: Option<usize>,
) -> Result<Vec<Candle>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = resolve_data_root();
        let ohlcv = neoethos_data::load_symbol_timeframe(&root, &symbol, &timeframe)
            .map_err(|e| e.to_string())?;
        let n = ohlcv.close.len();
        if n == 0 {
            return Ok::<Vec<Candle>, String>(Vec::new());
        }
        let ts = ohlcv.timestamp.clone().unwrap_or_default();
        let take = limit.unwrap_or(1500).min(n);
        let start = n - take;
        let mut out = Vec::with_capacity(take);
        let mut last_t = i64::MIN;
        for i in start..n {
            let t = ts.get(i).copied().unwrap_or(0) / 1000; // ms → s
            // Lightweight Charts requires strictly-ascending unique times.
            if t <= last_t {
                continue;
            }
            last_t = t;
            out.push(Candle {
                time: t,
                open: ohlcv.open[i],
                high: ohlcv.high[i],
                low: ohlcv.low[i],
                close: ohlcv.close[i],
            });
        }
        Ok(out)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            // Start the live cTrader spot-price streamer (best-effort; no-op
            // until the broker is authenticated). The UI polls spot_prices().
            broker::start_spot_streamer();
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // local vortex data
            app_info,
            list_symbols,
            list_timeframes,
            chart,
            // live cTrader broker (in-process, auto-auth)
            broker::broker_status,
            broker::broker_chart,
            broker::broker_symbols,
            broker::broker_accounts,
            broker::account_snapshot,
            broker::place_order,
            broker::close_position,
            broker::reauth_broker,
            broker::spot_prices,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
