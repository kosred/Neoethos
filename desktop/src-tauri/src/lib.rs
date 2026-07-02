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
            // Autonomous LLM supervisor heartbeat (no-op until enabled in the UI).
            neoethos_app::app_services::supervisor::spawn(state.clone());
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

/// Establish the per-user data root the engine reads (config.yaml + data/ +
/// cache/ + models/), the cross-platform STANDARD way — no hardcoded paths, so
/// it works for any user on any machine:
///   1. `NEOETHOS_USER_DATA_DIR` — explicit override (dev points it at a repo).
///   2. The current dir IF it already holds config.yaml (dev/portable launch).
///   3. OS-standard per-user dir: `%LOCALAPPDATA%\neoethos` /
///      `~/.local/share/neoethos` / `~/Library/Application Support/neoethos`
///      (via `neoethos_core::config::user_config_path()`).
/// On first run it SEEDS config.yaml (+ default symbol costs) from the bundled
/// read-only defaults, then chdirs to the root so every relative read resolves.
fn prepare_data_root(app: &tauri::App) {
    use tauri::Manager;

    let overridden = std::env::var("NEOETHOS_USER_DATA_DIR")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .is_some();

    // Dev/portable: launched from a dir that already has config.yaml → keep it.
    if !overridden && std::path::Path::new("config.yaml").exists() {
        eprintln!("data root → current dir (config.yaml present)");
        return;
    }

    // Override-aware canonical path (user_config_path honours the override).
    let cfg_path = neoethos_core::config::user_config_path();
    let root = cfg_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    if let Err(e) = std::fs::create_dir_all(&root) {
        eprintln!("could not create data root {}: {e}", root.display());
    }

    // First run: seed the editable config + default symbol costs from the
    // bundled read-only defaults so a fresh install works out of the box.
    if !cfg_path.exists() {
        if let Ok(res) = app
            .path()
            .resolve("resources/config.yaml", tauri::path::BaseDirectory::Resource)
        {
            match std::fs::copy(&res, &cfg_path) {
                Ok(_) => eprintln!("seeded default config → {}", cfg_path.display()),
                Err(e) => eprintln!("seed config failed: {e}"),
            }
        }
        let data = root.join("data");
        let _ = std::fs::create_dir_all(&data);
        if let Ok(res) = app
            .path()
            .resolve("resources/symbol_metadata.json", tauri::path::BaseDirectory::Resource)
        {
            let dst = data.join("symbol_metadata.json");
            if !dst.exists() {
                let _ = std::fs::copy(&res, &dst);
            }
        }
    }

    if let Err(e) = std::env::set_current_dir(&root) {
        eprintln!("set working dir {} failed: {e}", root.display());
    }
    eprintln!("data root → {}", root.display());
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

/// Open a native OS file picker for a data file to import (CSV/Parquet/TSV).
/// Returns the chosen absolute path, or `None` if the user cancelled — the
/// webview's `<input type=file>` can't expose a real path, so the import needs
/// this native dialog.
#[tauri::command]
async fn pick_data_file() -> Result<Option<String>, String> {
    let file = rfd::AsyncFileDialog::new()
        .set_title("Choose a data file to import")
        .add_filter("Data files", &["csv", "tsv", "parquet", "txt"])
        .add_filter("All files", &["*"])
        .pick_file()
        .await;
    Ok(file.map(|f| f.path().to_string_lossy().to_string()))
}

/// How much local history exists for a symbol on a given timeframe — so the
/// Discovery pre-flight can show the operator EXACTLY what's about to be
/// searched (years of data + bar count per pair) before they start.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SymbolCoverage {
    symbol: String,
    bars: usize,
    first_ms: i64,
    last_ms: i64,
    years: f64,
}

#[tauri::command]
async fn data_coverage(
    symbols: Vec<String>,
    timeframe: String,
) -> Result<Vec<SymbolCoverage>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = resolve_data_root();
        let out = symbols
            .iter()
            .map(|sym| match neoethos_data::load_symbol_timeframe(&root, sym, &timeframe) {
                Ok(o) => {
                    let ts = o.timestamp.unwrap_or_default();
                    let first = ts.first().copied().unwrap_or(0);
                    let last = ts.last().copied().unwrap_or(0);
                    // 365.25 d/yr in ms.
                    let years = if last > first {
                        (last - first) as f64 / 31_557_600_000.0
                    } else {
                        0.0
                    };
                    SymbolCoverage { symbol: sym.clone(), bars: o.close.len(), first_ms: first, last_ms: last, years }
                }
                Err(_) => SymbolCoverage { symbol: sym.clone(), bars: 0, first_ms: 0, last_ms: 0, years: 0.0 },
            })
            .collect::<Vec<_>>();
        Ok(out)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // STANDARD per-user data root (no hardcoded paths) + first-run seed.
            prepare_data_root(app);
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
            pick_data_file,
            data_coverage,
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
