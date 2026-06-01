//! Closed-trade + equity-curve journal — JSONL, append-only.
//!
//! The persistence layer for the professional trade journal. Mirrors the
//! proven append-only-JSONL + `OnceLock<Mutex>` shape of
//! [`crate::app_services::live_journal`] (NOT raw SQLite) so it adds zero
//! new integration surface and no `AppApiState` field. Two files under
//! `<data_dir>/journal/`:
//!   - `closed_trades.jsonl` — one [`ClosedTrade`] per line (round-trip P/L)
//!   - `equity_samples.jsonl` — one [`EquitySample`] per line (balance+equity)
//!
//! Stats (win%, profit factor, Sharpe, max drawdown, …) are derived from
//! these two raw artifacts by `journal_stats.rs`.
//!
//! Defensive by contract (operator directive: clear errors, never panic):
//! no `.unwrap()`/`.expect()`; a missing file reads as empty; a malformed
//! line is skipped with a debug log; writes on the trade path are
//! best-effort (a journal hiccup never aborts a trade). Closed-trade
//! writes are idempotent on `position_id`, so a per-tick reconcile loop
//! cannot duplicate rows.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const SCHEMA_VERSION: u32 = 1;

/// One closed round-trip trade — the unit a trade journal reports on.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClosedTrade {
    pub schema_version: u32,
    pub recorded_at_unix_ms: i64,
    /// Broker position id — the idempotency key (one closed trade / position).
    pub position_id: i64,
    pub symbol: String,
    /// "BUY" | "SELL".
    pub side: String,
    pub lots: f64,
    pub entry_ts_ms: Option<i64>,
    pub entry_price: Option<f64>,
    pub exit_ts_ms: Option<i64>,
    pub exit_price: Option<f64>,
    pub gross_profit: f64,
    pub commission: f64,
    pub swap: f64,
    pub net_profit: f64,
    /// Account balance immediately after this trade closed (if known).
    pub balance_after: Option<f64>,
}

impl ClosedTrade {
    /// Best timestamp to bucket this trade by (exit time, else recorded).
    pub fn effective_ts_ms(&self) -> i64 {
        self.exit_ts_ms.unwrap_or(self.recorded_at_unix_ms)
    }
}

/// One equity-curve sample (for drawdown / Sharpe / the equity chart).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EquitySample {
    pub ts_ms: i64,
    pub balance: f64,
    pub equity: f64,
}

pub fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn new_schema_version() -> u32 {
    SCHEMA_VERSION
}

fn journal_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("journal")
}
fn closed_trades_path(data_dir: &Path) -> PathBuf {
    journal_dir(data_dir).join("closed_trades.jsonl")
}
fn equity_path(data_dir: &Path) -> PathBuf {
    journal_dir(data_dir).join("equity_samples.jsonl")
}

fn writer_lock() -> &'static Mutex<()> {
    static WRITER: OnceLock<Mutex<()>> = OnceLock::new();
    WRITER.get_or_init(|| Mutex::new(()))
}

/// Process-global set of already-recorded `position_id`s, lazily seeded
/// from the file on first use — guards against the per-tick reconcile
/// appending the same closed trade repeatedly.
fn seen_positions() -> &'static Mutex<HashMap<PathBuf, HashSet<i64>>> {
    static SEEN: OnceLock<Mutex<HashMap<PathBuf, HashSet<i64>>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashMap::new()))
}

fn append_line(path: &Path, line: &str) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create journal dir {}", parent.display()))?;
    }
    let _guard = writer_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("journal writer lock poisoned"))?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open journal file {}", path.display()))?;
    writeln!(file, "{line}").with_context(|| format!("failed to append to {}", path.display()))?;
    Ok(())
}

/// Read + parse a JSONL file into `Vec<T>`, skipping malformed lines
/// (logged at debug). A missing file is an empty `Vec`, not an error.
fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path, label: &str) -> Vec<T> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            tracing::warn!(
                target: "neoethos_app::journal_store",
                error = %e, path = %path.display(),
                "could not read {label} journal; treating as empty"
            );
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<T>(line) {
            Ok(v) => out.push(v),
            Err(e) => {
                tracing::debug!(
                    target: "neoethos_app::journal_store",
                    error = %e, line_no = idx + 1,
                    "skipping malformed {label} journal line"
                );
            }
        }
    }
    out
}

/// Record a closed trade. Idempotent on `position_id`. Returns `Ok(true)`
/// if newly written, `Ok(false)` if it was already recorded.
pub fn record_closed_trade(data_dir: &Path, trade: &ClosedTrade) -> Result<bool> {
    {
        let mut guard = seen_positions()
            .lock()
            .map_err(|_| anyhow::anyhow!("journal seen-set lock poisoned"))?;
        // Per-data-dir dedup cache, seeded from the file the first time
        // this dir is touched — correct across multiple data dirs / tests,
        // not a single process-wide set keyed to whatever dir came first.
        let set = guard.entry(data_dir.to_path_buf()).or_insert_with(|| {
            read_jsonl::<ClosedTrade>(&closed_trades_path(data_dir), "closed-trades")
                .iter()
                .map(|t| t.position_id)
                .collect()
        });
        if !set.insert(trade.position_id) {
            return Ok(false); // already recorded
        }
    }
    let line = serde_json::to_string(trade).context("serialise closed trade")?;
    append_line(&closed_trades_path(data_dir), &line)?;
    Ok(true)
}

/// Fire-and-forget variant for the trading/refresh hot path — logs a
/// warning instead of propagating, so a journal failure never aborts a
/// trade or stalls the account-refresh loop.
pub fn record_closed_trade_best_effort(data_dir: &Path, trade: &ClosedTrade) {
    if let Err(e) = record_closed_trade(data_dir, trade) {
        tracing::warn!(
            target: "neoethos_app::journal_store",
            error = %e,
            "failed to record closed trade (continuing)"
        );
    }
}

pub fn append_equity_sample_best_effort(data_dir: &Path, sample: &EquitySample) {
    let line = match serde_json::to_string(sample) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(target: "neoethos_app::journal_store", error = %e, "serialise equity sample");
            return;
        }
    };
    if let Err(e) = append_line(&equity_path(data_dir), &line) {
        tracing::warn!(
            target: "neoethos_app::journal_store",
            error = %e,
            "append equity sample (continuing)"
        );
    }
}

/// Closed trades with effective timestamp in `[from_ms, to_ms)` (None =
/// unbounded). Deduped on `position_id` (last line wins), sorted ascending.
pub fn query_closed_trades(
    data_dir: &Path,
    from_ms: Option<i64>,
    to_ms: Option<i64>,
) -> Vec<ClosedTrade> {
    let all: Vec<ClosedTrade> = read_jsonl(&closed_trades_path(data_dir), "closed-trades");
    let mut by_pos: HashMap<i64, ClosedTrade> = HashMap::new();
    for t in all {
        by_pos.insert(t.position_id, t);
    }
    let mut out: Vec<ClosedTrade> = by_pos
        .into_values()
        .filter(|t| {
            let ts = t.effective_ts_ms();
            from_ms.map_or(true, |f| ts >= f) && to_ms.map_or(true, |to| ts < to)
        })
        .collect();
    out.sort_by_key(ClosedTrade::effective_ts_ms);
    out
}

pub fn query_equity(data_dir: &Path, from_ms: Option<i64>, to_ms: Option<i64>) -> Vec<EquitySample> {
    let mut all: Vec<EquitySample> = read_jsonl(&equity_path(data_dir), "equity");
    all.retain(|s| from_ms.map_or(true, |f| s.ts_ms >= f) && to_ms.map_or(true, |to| s.ts_ms < to));
    all.sort_by_key(|s| s.ts_ms);
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("neoethos_journal_test_{tag}_{}", now_unix_ms()));
        p
    }

    fn trade(pos: i64, net: f64, exit_ms: i64) -> ClosedTrade {
        ClosedTrade {
            schema_version: SCHEMA_VERSION,
            recorded_at_unix_ms: exit_ms,
            position_id: pos,
            symbol: "EURUSD".to_string(),
            side: "BUY".to_string(),
            lots: 0.1,
            entry_ts_ms: Some(exit_ms - 3_600_000),
            entry_price: Some(1.1000),
            exit_ts_ms: Some(exit_ms),
            exit_price: Some(1.1000 + net / 1000.0),
            gross_profit: net,
            commission: 0.0,
            swap: 0.0,
            net_profit: net,
            balance_after: None,
        }
    }

    #[test]
    fn record_is_idempotent_on_position_id() -> Result<()> {
        let dir = tmp_dir("idem");
        std::fs::create_dir_all(&dir)?;
        assert!(record_closed_trade(&dir, &trade(1, 10.0, 1000))?);
        // Same position id again → not re-written.
        assert!(!record_closed_trade(&dir, &trade(1, 10.0, 1000))?);
        assert!(record_closed_trade(&dir, &trade(2, -5.0, 2000))?);
        let all = query_closed_trades(&dir, None, None);
        assert_eq!(all.len(), 2, "expected 2 unique trades, got {}", all.len());
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn query_filters_by_time_and_sorts() -> Result<()> {
        let dir = tmp_dir("time");
        std::fs::create_dir_all(&dir)?;
        record_closed_trade_best_effort(&dir, &trade(10, 1.0, 3000));
        record_closed_trade_best_effort(&dir, &trade(11, 2.0, 1000));
        record_closed_trade_best_effort(&dir, &trade(12, 3.0, 2000));
        let in_window = query_closed_trades(&dir, Some(1500), Some(3500));
        assert_eq!(in_window.len(), 2);
        assert_eq!(in_window[0].position_id, 12); // ts=2000 sorts before ts=3000
        assert_eq!(in_window[1].position_id, 10);
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn missing_file_reads_empty() {
        let dir = tmp_dir("missing");
        // Never created — must not panic, must be empty.
        assert!(query_closed_trades(&dir, None, None).is_empty());
        assert!(query_equity(&dir, None, None).is_empty());
    }
}
