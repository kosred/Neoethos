//! Session-aware spread statistics — recorded from the broker's OWN ticks.
//!
//! Backtests that charge one flat spread are systematically optimistic: real
//! FX spreads breathe with the session (tight in London/NY overlap, 2-5× wider
//! through the Asian lull and the 21-22 UTC rollover). This module samples the
//! live tick cache once a minute and accumulates per-(symbol, UTC-hour) spread
//! stats — count / mean / max in pips — persisted to
//! `<data_dir>/spread_stats.json`.
//!
//! Consumers today: `GET /data/spread-stats` + the Data screen table, so the
//! operator can SEE the broker's true hourly cost surface and set
//! `risk.backtest_spread_pips` honestly. Consumer tomorrow: the per-bar
//! spread array for the eval kernels (charge each simulated fill the spread
//! of ITS hour) — the stats recorded from today feed that directly.
//!
//! Best-effort by contract: sampling errors are skipped, persistence failures
//! log and never propagate; the sampler must never disturb trading.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

/// One UTC hour's accumulated spread stats for one symbol.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HourStats {
    pub samples: u64,
    pub mean_pips: f64,
    pub max_pips: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SymbolSpreadStats {
    /// Index = UTC hour 0..24.
    pub hourly: Vec<HourStats>, // always 24 entries once touched
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpreadStatsFile {
    pub symbols: HashMap<String, SymbolSpreadStats>,
    /// Unix-ms of the last persist — staleness hint for the UI.
    pub updated_ms: i64,
}

fn store() -> &'static Mutex<SpreadStatsFile> {
    static S: OnceLock<Mutex<SpreadStatsFile>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(load_from_disk().unwrap_or_default()))
}

fn stats_path() -> Option<PathBuf> {
    neoethos_core::Settings::from_yaml(&crate::server::state::current_config_path())
        .ok()
        .map(|s| s.system.data_dir.join("spread_stats.json"))
}

fn load_from_disk() -> Option<SpreadStatsFile> {
    let raw = std::fs::read_to_string(stats_path()?).ok()?;
    serde_json::from_str(&raw).ok()
}

fn persist_locked(file: &SpreadStatsFile) {
    let Some(path) = stats_path() else { return };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    match serde_json::to_string(file) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::warn!(
                    target: "neoethos_app::spread_stats",
                    error = %e, "failed to persist spread stats"
                );
            }
        }
        Err(e) => tracing::warn!(
            target: "neoethos_app::spread_stats",
            error = %e, "failed to serialize spread stats"
        ),
    }
}

/// Snapshot for the API/UI.
pub fn snapshot() -> SpreadStatsFile {
    store().lock().map(|s| s.clone()).unwrap_or_default()
}

/// Fold one observed spread sample into the running stats.
fn record(symbol: &str, hour: usize, spread_pips: f64) {
    if !(spread_pips.is_finite() && spread_pips >= 0.0 && spread_pips < 1000.0) {
        return;
    }
    let Ok(mut s) = store().lock() else { return };
    let entry = s.symbols.entry(symbol.to_string()).or_default();
    if entry.hourly.len() != 24 {
        entry.hourly = vec![HourStats::default(); 24];
    }
    let h = &mut entry.hourly[hour.min(23)];
    h.samples += 1;
    // Streaming mean; max tracked directly.
    h.mean_pips += (spread_pips - h.mean_pips) / h.samples as f64;
    if spread_pips > h.max_pips {
        h.max_pips = spread_pips;
    }
}

/// Spawn the background sampler: every 60s read the live tick cache and fold
/// each fresh bid/ask into the per-hour stats; persist every ~10 minutes.
pub fn spawn() {
    tokio::spawn(async move {
        let mut ticks_since_persist = 0u32;
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            let now_ms = chrono::Utc::now().timestamp_millis();
            let hour = chrono::Utc::now().format("%H").to_string().parse::<usize>().unwrap_or(0);
            for t in crate::app_services::live_spots::snapshot_all() {
                // Only FRESH ticks — a stale cache entry's spread is history.
                if now_ms - t.received_at_unix_ms > 90_000 {
                    continue;
                }
                let (Some(bid), Some(ask)) = (t.bid, t.ask) else { continue };
                if ask <= bid {
                    continue; // crossed/invalid quote — skip, never record garbage
                }
                let Some(meta) = neoethos_core::symbol_metadata::resolve(&t.symbol_name) else {
                    continue;
                };
                if !(meta.pip_size.is_finite() && meta.pip_size > 0.0) {
                    continue;
                }
                record(&t.symbol_name, hour, (ask - bid) / meta.pip_size);
            }
            ticks_since_persist += 1;
            if ticks_since_persist >= 10 {
                ticks_since_persist = 0;
                if let Ok(mut s) = store().lock() {
                    s.updated_ms = now_ms;
                    persist_locked(&s);
                }
            }
        }
    });
}
