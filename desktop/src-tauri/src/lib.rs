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

/// In-process backend: the full neoethos-app axum API, served on an ephemeral
/// loopback port inside THIS process (a tokio task, not a separate exe). It
/// starts with the app and dies with it — no supervisor, no spawn, no fixed
/// port. The web UI reads the port via the `api_base` command and calls the
/// same ~50 handlers the old Flutter client used. One binary, one process.
mod backend {
    use std::net::TcpListener;
    use std::sync::OnceLock;

    use neoethos_app::server;

    static API_PORT: OnceLock<u16> = OnceLock::new();

    /// Bind an ephemeral loopback port *synchronously* (so the port is known
    /// before the window loads), then serve the full API on it.
    pub fn start() {
        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(l) => l,
            Err(e) => {
                eprintln!("FATAL: could not bind the in-process backend: {e}");
                return;
            }
        };
        let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
        let _ = API_PORT.set(port);
        eprintln!("in-process backend bound on 127.0.0.1:{port}");

        tauri::async_runtime::spawn(async move {
            // Mirror main.rs bootstrap, minus the Flutter-supervisor bits.
            // Same default config path the CLI/main.rs use when no override.
            server::state::install_config_path("config.yaml");
            let state = server::state::AppApiState::new();
            server::state::install_account_refresh_trigger(state.account_refresh_tx_clone());
            server::bridge::spawn(state.clone());
            if let Err(e) = server::serve_on(listener, state).await {
                eprintln!("in-process backend exited: {e:#}");
            }
        });
    }

    pub fn base_url() -> String {
        format!("http://127.0.0.1:{}", API_PORT.get().copied().unwrap_or(0))
    }
}

/// Base URL of the in-process backend (e.g. `http://127.0.0.1:54321`). The web
/// UI fetches this once at startup and uses it for every backend call.
#[tauri::command]
fn api_base() -> String {
    backend::base_url()
}

/// Reveal a file or folder in the OS file manager (Windows Explorer). Files are
/// highlighted via `/select,`; folders open directly. Lets the user find any
/// data/model/log the app stores with one click.
#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    let p = std::path::Path::new(&path);
    let mut cmd = std::process::Command::new("explorer");
    if p.is_file() {
        cmd.arg("/select,").arg(&path);
    } else {
        cmd.arg(&path);
    }
    // explorer.exe returns non-zero exit codes even on success; ignore status.
    cmd.spawn().map(|_| ()).map_err(|e| e.to_string())
}

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

/// The engine reads config.yaml + data/ + cache/ + models/ RELATIVE to the
/// process CWD. Launched from the installer (Start menu / Program Files) the CWD
/// is NOT the project dir, so every relative path 404s ("config.yaml not
/// loadable", everything shows "(missing)"). Point the CWD at the project root
/// (where config.yaml lives) before anything reads it.
fn ensure_working_dir() {
    if std::path::Path::new("config.yaml").exists() {
        return; // already in the project dir (dev launch)
    }
    let root = std::path::Path::new(r"C:\Users\konst\development\forex-ai");
    if root.join("config.yaml").exists() {
        let _ = std::env::set_current_dir(root);
        eprintln!("working dir set → {}", root.display());
    } else {
        eprintln!("WARNING: config.yaml not found in CWD or project root — paths will 404");
    }
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
    ensure_working_dir();
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            // Start the full neoethos-app API in-process (Discovery, Training,
            // Risk, Journal, News, Intelligence, Data, Hardware, Autonomous,
            // Codex, …) — every old Flutter feature, reachable from the new UI.
            backend::start();
            // Start the live cTrader spot-price streamer (best-effort; no-op
            // until the broker is authenticated). Feeds the in-process server's
            // /live/spots/stream (SSE) that the UI subscribes to.
            broker::start_spot_streamer();
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // in-process backend base URL + file manager
            api_base,
            open_path,
            // local vortex data
            app_info,
            list_symbols,
            list_timeframes,
            chart,
            // live cTrader broker (in-process, auto-auth)
            broker::broker_status,
            broker::broker_chart,
            broker::broker_accounts,
            broker::select_account,
            broker::account_snapshot,
            broker::place_order,
            broker::close_position,
            broker::reauth_broker,
            broker::refresh_broker_costs,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
