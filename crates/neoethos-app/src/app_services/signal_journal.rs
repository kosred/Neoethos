//! In-memory journal of dispatched auto-trade signals (#127).
//!
//! Every time the auto-trade dispatcher sends a signal toward the
//! broker, it records the signal here BEFORE the fill happens. The
//! Gemma `explain_recent_trades` tool reads this journal back and
//! joins it against `AppApiState.account.positions` via the
//! `(symbol, side, timestamp_ms)` heuristic — the model is good
//! enough to correlate "BUY EURUSD signal at 14:32:01 conf 0.78"
//! with "open EURUSD long opened 14:32:03" without us building a
//! position_id round-trip on the order path.
//!
//! ## Why a separate module + global
//!
//! The dispatcher lives in `trading::auto_trade::TradingSession`
//! (legacy egui-era surface kept as test fixture, see #107
//! cleanup). The LLM tool runs against `AppApiState` (the HTTP
//! server's state). Threading the API state through TradingSession
//! to write a single signal would touch ~10 files. A global
//! Mutex-wrapped VecDeque decouples the two completely — both ends
//! call free functions, no shared types beyond the [`SignalRecord`]
//! struct defined here.
//!
//! ## Bounded
//!
//! Cap is `SIGNAL_JOURNAL_CAPACITY = 128`. FIFO eviction (oldest
//! out, newest in) so the journal never grows unbounded inside a
//! long-running process. 128 is enough for a few hours of an
//! active scalper at M1 cadence.
//!
//! ## Persistence (#131)
//!
//! Each call to `record()` ALSO append-writes the JSON row to
//! `<data_dir>/neoethos/signal_journal.jsonl` (path overridable via
//! `NEOETHOS_SIGNAL_JOURNAL_PATH`). On process startup the journal
//! is hydrated from disk via `restore_from_disk()` — the last N
//! records (capped at SIGNAL_JOURNAL_CAPACITY) are loaded into the
//! in-memory deque so `explain_recent_trades` can narrate signals
//! that fired before the most recent restart.
//!
//! The on-disk file is bounded by size: when it crosses
//! `JOURNAL_ROTATE_BYTES`, the existing file is renamed to a
//! `.YYYY-MM-DD.jsonl` archive and a fresh one is started. This
//! avoids the unbounded-growth foot-gun of any append-only log
//! while still preserving multi-day history for the operator's
//! audit.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Rotate the on-disk JSONL when it grows past this many bytes.
/// 10 MB at ~250 B per row is roughly 40 000 signals — months of
/// active scalper output. The cap protects operators against
/// runaway producers without choking long-running processes.
pub const JOURNAL_ROTATE_BYTES: u64 = 10 * 1024 * 1024;

/// Maximum signals kept in the journal. FIFO eviction once this is
/// exceeded. 128 picked for the same reason `BOT_DECISION_BUFFER_
/// CAPACITY` is 512 in the trading session — a few hours of active
/// scalping fits easily, and the deque's memory footprint at this
/// size is dominated by the feature_snapshot HashMaps (~256 B each
/// when populated, total ~32 KB worst-case).
pub const SIGNAL_JOURNAL_CAPACITY: usize = 128;

/// One row in the journal — the data the dispatcher knows at
/// signal-dispatch time. We do NOT include the fill outcome
/// (position_id, execution_price) because that information is not
/// available at the point we record — `execute_ctrader_order` is
/// fire-and-forget by design. The LLM joins this with current
/// positions on (symbol, side, timestamp).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignalRecord {
    /// Unix millis when the signal was emitted.
    pub timestamp_ms: i64,
    pub symbol: String,
    /// "BUY" / "SELL" / "FLAT". String rather than the enum so the
    /// module can serialize/deserialize without pulling
    /// `AutoTradeSide`'s serde derive into scope.
    pub side: String,
    pub confidence: f64,
    /// Free-form label the chart overlay paints.
    pub label: String,
    /// Stable identifier of the strategy/source that fired this
    /// signal. `None` for paths that don't track one (manual UI
    /// clicks routed through this struct).
    pub strategy_id: Option<String>,
    /// Stable identifier of the model ensemble that emitted the
    /// prediction. `None` for non-ML paths.
    pub model_id: Option<String>,
    /// Snapshot of feature values the model saw at inference time
    /// (e.g. `{"rsi_14": 58.2, "macd_hist": 0.0014}`). Empty for now
    /// — populated when the predictor stack returns richer
    /// PredictionOutput (follow-up).
    pub feature_snapshot: std::collections::HashMap<String, f64>,
    /// Whether the dispatcher actually handed off to the broker
    /// path. False when an early gate rejected the signal (news
    /// blackout, halt, risky-mode kill switch, etc.). The reason
    /// goes in `dispatch_note`.
    pub dispatched: bool,
    /// One-liner describing the outcome of the dispatch decision.
    /// "Dispatched to broker" on success; the GateDecision name
    /// on rejection ("RiskyModeKillSwitch", "NewsBlackout", etc.).
    pub dispatch_note: String,
}

static JOURNAL: OnceLock<Mutex<VecDeque<SignalRecord>>> = OnceLock::new();

fn journal() -> &'static Mutex<VecDeque<SignalRecord>> {
    JOURNAL.get_or_init(|| Mutex::new(VecDeque::with_capacity(SIGNAL_JOURNAL_CAPACITY)))
}

/// Append a signal to the journal. Idempotency is the caller's
/// problem — the dispatcher calls this exactly once per signal so
/// there's no de-dup logic here. Also appends the row to the
/// on-disk JSONL so the explain tool can narrate signals that
/// fired in earlier sessions. Failure to write to disk is logged
/// but never propagated — the in-memory journal is the source of
/// truth for the current session.
pub fn record(rec: SignalRecord) {
    {
        let Ok(mut q) = journal().lock() else {
            // Lock poisoned → another thread panicked while holding it.
            // The right move is to log + carry on (signal records are
            // ephemeral observability, not durable state). We never
            // panic out of the dispatcher path because of this.
            tracing::warn!(
                target: "neoethos_app::signal_journal",
                "signal journal mutex poisoned; dropping record"
            );
            return;
        };
        if q.len() >= SIGNAL_JOURNAL_CAPACITY {
            q.pop_front();
        }
        q.push_back(rec.clone());
    }
    // On-disk append outside the in-memory lock so a slow disk
    // can't stall callers waiting to record.
    if let Err(err) = append_to_disk(&rec) {
        tracing::warn!(
            target: "neoethos_app::signal_journal",
            error = %err,
            "failed to persist signal record to disk; in-memory journal unaffected"
        );
    }
}

/// Canonical on-disk path for the JSONL log. Honours the
/// `NEOETHOS_SIGNAL_JOURNAL_PATH` env override (tests, alt-dir
/// deployments); falls back to `<data_dir>/neoethos/
/// signal_journal.jsonl`.
pub fn default_journal_path() -> PathBuf {
    if let Ok(custom) = std::env::var("NEOETHOS_SIGNAL_JOURNAL_PATH") {
        if !custom.trim().is_empty() {
            return PathBuf::from(custom);
        }
    }
    let base = dirs::data_dir().unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".local")
    });
    base.join("neoethos").join("signal_journal.jsonl")
}

fn append_to_disk(rec: &SignalRecord) -> Result<()> {
    let path = default_journal_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("failed to create signal-journal directory at {}", parent.display())
        })?;
    }
    // Rotate before opening if the existing file is over the cap.
    if let Ok(meta) = std::fs::metadata(&path) {
        if meta.len() > JOURNAL_ROTATE_BYTES {
            rotate(&path)?;
        }
    }
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open signal journal {}", path.display()))?;
    let json = serde_json::to_string(rec)
        .context("failed to serialise signal record to JSON")?;
    writeln!(f, "{json}").context("failed to write signal record line")?;
    Ok(())
}

/// Rotate the journal: rename the current file to
/// `signal_journal.YYYY-MM-DD.jsonl` so the next record() opens a
/// fresh file. We don't compress / delete archives — the operator
/// owns the audit trail and can prune by hand. Errors propagate
/// so the next record() write goes through `append_to_disk`'s
/// regular "failed to persist" path.
fn rotate(path: &PathBuf) -> Result<()> {
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let mut archive = path.clone();
    archive.set_file_name(format!("signal_journal.{today}.jsonl"));
    // If today's archive already exists (process restarted twice
    // in one day after rotation), pick the next numbered suffix.
    let mut variant = 1u32;
    while archive.exists() {
        archive.set_file_name(format!("signal_journal.{today}.{variant}.jsonl"));
        variant += 1;
    }
    std::fs::rename(path, &archive).with_context(|| {
        format!(
            "failed to rotate signal journal {} -> {}",
            path.display(),
            archive.display()
        )
    })?;
    tracing::info!(
        target: "neoethos_app::signal_journal",
        archive = %archive.display(),
        "rotated signal journal"
    );
    Ok(())
}

/// Hydrate the in-memory journal from the on-disk JSONL. Called
/// once at startup so `explain_recent_trades` can narrate signals
/// from prior sessions. Reads the file line-by-line; bad lines
/// are skipped with a warn-level log rather than failing the
/// whole restore. Returns the number of records loaded (or 0
/// when the file is missing — first-launch).
pub fn restore_from_disk() -> usize {
    let path = default_journal_path();
    let Ok(f) = std::fs::File::open(&path) else {
        // First-launch — file doesn't exist yet. Not an error.
        return 0;
    };
    let mut records: VecDeque<SignalRecord> = VecDeque::with_capacity(SIGNAL_JOURNAL_CAPACITY);
    for line in BufReader::new(f).lines() {
        let Ok(line) = line else { continue };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<SignalRecord>(line) {
            Ok(rec) => {
                if records.len() >= SIGNAL_JOURNAL_CAPACITY {
                    records.pop_front();
                }
                records.push_back(rec);
            }
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::signal_journal",
                    error = %err,
                    "skipping malformed signal-journal line"
                );
            }
        }
    }
    let count = records.len();
    if let Ok(mut q) = journal().lock() {
        // Overwrite — restore is called at startup, before any
        // record() can have appended.
        *q = records;
        tracing::info!(
            target: "neoethos_app::signal_journal",
            count,
            path = %path.display(),
            "restored signal journal from disk"
        );
    }
    count
}

/// Return the N most-recent signals, newest first. `limit` is
/// clamped to `SIGNAL_JOURNAL_CAPACITY`. Returns an empty Vec on
/// lock poisoning (treat as "no data" rather than crashing the
/// tool that asked).
///
/// The only production caller lives in
/// `gemma_tools::ExplainRecentTradesTool`, which is itself
/// `#[cfg(feature = "gemma-backend")]`. In default builds this
/// function compiles but has no caller; the tests below keep it
/// from being truly dead. Annotate at the function level so the
/// allow stays narrow.
#[allow(dead_code)]
pub fn recent(limit: usize) -> Vec<SignalRecord> {
    let limit = limit.min(SIGNAL_JOURNAL_CAPACITY);
    let Ok(q) = journal().lock() else {
        tracing::warn!(
            target: "neoethos-app::signal_journal",
            "signal journal mutex poisoned; returning empty"
        );
        return Vec::new();
    };
    q.iter().rev().take(limit).cloned().collect()
}

/// Test-only escape hatch — clears the journal so consecutive tests
/// see a known-empty starting state. Production never calls this.
#[cfg(test)]
pub fn clear() {
    if let Ok(mut q) = journal().lock() {
        q.clear();
    }
}

/// Test-only — remove the on-disk JSONL between tests. Idempotent.
#[cfg(test)]
pub fn clear_disk() {
    let path = default_journal_path();
    let _ = std::fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(ts: i64, side: &str) -> SignalRecord {
        SignalRecord {
            timestamp_ms: ts,
            symbol: "EURUSD".to_string(),
            side: side.to_string(),
            confidence: 0.72,
            label: format!("AI {side} · 0.72"),
            strategy_id: Some("ema_cross_v3".to_string()),
            model_id: Some("ensemble:EURUSD".to_string()),
            feature_snapshot: std::collections::HashMap::new(),
            dispatched: true,
            dispatch_note: "Dispatched to broker".to_string(),
        }
    }

    /// All tests in this module share the global JOURNAL; serialise
    /// via a once-init mutex so they don't stomp each other.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn record_then_recent_roundtrips() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        record(sample(1_000, "BUY"));
        record(sample(2_000, "SELL"));
        let got = recent(10);
        assert_eq!(got.len(), 2);
        // newest first
        assert_eq!(got[0].timestamp_ms, 2_000);
        assert_eq!(got[0].side, "SELL");
        assert_eq!(got[1].timestamp_ms, 1_000);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        // Push one over the cap.
        for i in 0..(SIGNAL_JOURNAL_CAPACITY + 1) {
            record(sample(i as i64, "BUY"));
        }
        let all = recent(SIGNAL_JOURNAL_CAPACITY * 2);
        assert_eq!(all.len(), SIGNAL_JOURNAL_CAPACITY);
        // The very first one (ts=0) should have been evicted.
        assert!(all.iter().all(|r| r.timestamp_ms > 0));
        // The newest record (ts = CAP) is at the front.
        assert_eq!(all[0].timestamp_ms as usize, SIGNAL_JOURNAL_CAPACITY);
    }

    #[test]
    fn recent_clamps_limit_to_capacity() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        record(sample(1, "BUY"));
        // Asking for a huge number doesn't panic / over-allocate;
        // returns just what's there.
        let got = recent(10_000);
        assert_eq!(got.len(), 1);
    }

    /// Helper — point the journal at a temp path so the test
    /// doesn't touch the operator's real signal_journal.jsonl.
    /// Returns a guard that restores the env on drop.
    struct TempJournalPath {
        path: std::path::PathBuf,
        prior: Option<String>,
    }
    impl TempJournalPath {
        fn new(name: &str) -> Self {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "neoethos-signal-journal-{name}-{}.jsonl",
                std::process::id()
            ));
            let _ = std::fs::remove_file(&path);
            let prior = std::env::var("NEOETHOS_SIGNAL_JOURNAL_PATH").ok();
            // SAFETY: env mutation is process-global; TEST_LOCK
            // ensures no concurrent journal test races.
            unsafe {
                std::env::set_var(
                    "NEOETHOS_SIGNAL_JOURNAL_PATH",
                    path.to_str().unwrap_or(""),
                );
            }
            Self { path, prior }
        }
    }
    impl Drop for TempJournalPath {
        fn drop(&mut self) {
            unsafe {
                if let Some(ref v) = self.prior {
                    std::env::set_var("NEOETHOS_SIGNAL_JOURNAL_PATH", v);
                } else {
                    std::env::remove_var("NEOETHOS_SIGNAL_JOURNAL_PATH");
                }
            }
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn record_appends_jsonl_to_disk() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = TempJournalPath::new("appends");
        clear();
        clear_disk();
        record(sample(1_000, "BUY"));
        record(sample(2_000, "SELL"));
        let body = std::fs::read_to_string(&tmp.path).expect("file should exist");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        // Each line must round-trip through serde.
        for line in lines {
            let _: SignalRecord = serde_json::from_str(line).expect("valid JSON record");
        }
    }

    #[test]
    fn restore_from_disk_populates_in_memory_journal() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _tmp = TempJournalPath::new("restore");
        clear();
        clear_disk();
        // Persist two records, then wipe the in-memory deque and
        // verify restore_from_disk re-hydrates from the file.
        record(sample(5_000, "BUY"));
        record(sample(6_000, "SELL"));
        clear();
        assert!(recent(10).is_empty(), "in-memory should be empty pre-restore");
        let restored = restore_from_disk();
        assert_eq!(restored, 2);
        let got = recent(10);
        assert_eq!(got.len(), 2);
        // newest first
        assert_eq!(got[0].timestamp_ms, 6_000);
        assert_eq!(got[1].timestamp_ms, 5_000);
    }

    #[test]
    fn restore_handles_missing_file_as_zero() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _tmp = TempJournalPath::new("missing");
        clear();
        clear_disk();
        let n = restore_from_disk();
        assert_eq!(n, 0);
    }

    #[test]
    fn restore_skips_malformed_lines() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = TempJournalPath::new("malformed");
        clear();
        clear_disk();
        // Write one good record + a broken line + another good one.
        std::fs::create_dir_all(tmp.path.parent().unwrap()).ok();
        let good = serde_json::to_string(&sample(1, "BUY")).unwrap();
        let good2 = serde_json::to_string(&sample(2, "SELL")).unwrap();
        std::fs::write(
            &tmp.path,
            format!("{good}\n{{not valid json}}\n{good2}\n"),
        )
        .unwrap();
        let n = restore_from_disk();
        assert_eq!(n, 2, "malformed line should be skipped, not abort restore");
    }

    #[test]
    fn default_journal_path_honours_env_override() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _tmp = TempJournalPath::new("env-override");
        let p = default_journal_path();
        // The TempJournalPath helper set the env to its own path —
        // verify default_journal_path picked that up rather than
        // falling back to the platform default.
        assert!(
            p.to_string_lossy().contains("neoethos-signal-journal-env-override"),
            "path {} did not honour env override",
            p.display()
        );
    }
}
