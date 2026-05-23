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
//! ## Lost on restart
//!
//! In-memory only. A process restart wipes the journal. The
//! follow-up under #128 will flip the existing JSONL live-journal
//! path on by default so signals survive across sessions; until
//! then, `explain_recent_trades` can only narrate same-session
//! fills.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

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
/// there's no de-dup logic here.
pub fn record(rec: SignalRecord) {
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
    q.push_back(rec);
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
}
