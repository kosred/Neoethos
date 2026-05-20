//! Models ↔ Gemma communication bridge.
//!
//! Phase G0 — trait + a small in-memory ring-buffer
//! implementation that holds the most recent ML events Gemma's
//! context window should mention. G4 wires this to the existing
//! `ServiceEvent` channel (`crate::app_services::ServiceEvent`
//! in `forex-app`) so the producer's `AutoTradeSignal` /
//! `TrainingUpdated` / `DiscoveryUpdated` events become Gemma
//! context fragments.
//!
//! ## Push direction (models → Gemma)
//!
//! When the ensemble emits an `AutoTradeSignal`, the producer
//! pushes a `GemmaContextEvent` onto this bridge. The runtime
//! pulls the most recent N events ahead of each user prompt and
//! injects them as a system-message snippet:
//!
//! > "Recent model output: the ensemble predicted UP on EUR/USD
//! > with confidence 0.78 at 14:23 UTC; you have 2 open
//! > positions; news blackout window is active for FOMC at
//! > 19:00."
//!
//! ## Pull direction (Gemma → models)
//!
//! Implemented via `BotTool` trait calls in G3 (e.g.
//! `get_recent_predictions`, `explain_last_decision`,
//! `get_model_confidence`). Read-only — no `ToolCategory` gate
//! required.
//!
//! ## Look-ahead bias
//!
//! The bridge accepts `event_timestamp_unix_ms` on every push.
//! When the runtime pulls events for a prompt, it filters at
//! `timestamp < ToolContext.past_data_cutoff_unix_ms` — same
//! discipline as the tool layer. The bridge itself doesn't
//! enforce the cutoff (events are pushed before bars close in
//! the producer thread); the filter lives in the puller.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Mutex;

/// One ML-side event Gemma might be told about. Variants are
/// tagged so the runtime can format them differently into the
/// system-message snippet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GemmaContextEvent {
    /// Ensemble emitted an `AutoTradeSignal`.
    AutoTradeSignal {
        timestamp_unix_ms: i64,
        symbol: String,
        direction: SignalDirection,
        confidence: f64,
    },
    /// A training job changed state (started / finished / failed).
    TrainingUpdated {
        timestamp_unix_ms: i64,
        job_id: String,
        status: String,
    },
    /// News blackout window opened / closed.
    NewsBlackout {
        timestamp_unix_ms: i64,
        active: bool,
        reason: String,
    },
    /// A trade was placed / modified / cancelled — origin-tagged
    /// so the helper knows whether the user did it manually or
    /// Gemma did via a gated tool.
    TradeLifecycle {
        timestamp_unix_ms: i64,
        order_code: String,
        origin: TradeOrigin,
        new_status: String,
    },
}

impl GemmaContextEvent {
    pub fn timestamp_unix_ms(&self) -> i64 {
        match self {
            Self::AutoTradeSignal {
                timestamp_unix_ms, ..
            }
            | Self::TrainingUpdated {
                timestamp_unix_ms, ..
            }
            | Self::NewsBlackout {
                timestamp_unix_ms, ..
            }
            | Self::TradeLifecycle {
                timestamp_unix_ms, ..
            } => *timestamp_unix_ms,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalDirection {
    Long,
    Short,
    Flat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TradeOrigin {
    /// User clicked manually in the UI. Risky Mode rejects these
    /// when autonomous-only mode is on (see `OrderSource::Manual`
    /// in `forex-app`).
    Manual,
    /// AI-initiated — either the autonomous producer OR Gemma
    /// via a gated tool. Distinguished from `Manual` so Risky
    /// Mode's §7.1 contract holds.
    Ai,
    /// Specifically Gemma-initiated via a gated tool. Subset of
    /// `Ai`; tagged for the audit log.
    Gemma,
}

/// Bridge trait. The runtime holds an
/// `Arc<dyn ModelGemmaBridge>`; the producer pushes events to
/// it; the runtime pulls them ahead of each prompt.
pub trait ModelGemmaBridge: Send + Sync {
    /// Push a new event. Should be O(1).
    fn push(&self, event: GemmaContextEvent);

    /// Pull the most recent `n` events, oldest-first within the
    /// window. Caller MAY filter further (e.g. by timestamp for
    /// look-ahead bias).
    fn recent(&self, n: usize) -> Vec<GemmaContextEvent>;

    /// Total events buffered. Useful for diagnostics + tests.
    fn len(&self) -> usize;

    /// Convenience.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// G0 ring-buffer impl. Holds the last `capacity` events in
/// memory. Drops the oldest when full. Thread-safe via `Mutex`
/// — contention is fine because pushes are infrequent
/// (one per signal / training event, not one per tick).
pub struct InMemoryModelGemmaBridge {
    capacity: usize,
    inner: Mutex<VecDeque<GemmaContextEvent>>,
}

impl InMemoryModelGemmaBridge {
    /// Default capacity matches the operator-approved
    /// "last 10 events" push policy from the design doc.
    pub const DEFAULT_CAPACITY: usize = 10;

    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            inner: Mutex::new(VecDeque::with_capacity(capacity.max(1))),
        }
    }
}

impl Default for InMemoryModelGemmaBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelGemmaBridge for InMemoryModelGemmaBridge {
    fn push(&self, event: GemmaContextEvent) {
        let mut q = self.inner.lock().expect("bridge mutex poisoned");
        if q.len() == self.capacity {
            q.pop_front();
        }
        q.push_back(event);
    }

    fn recent(&self, n: usize) -> Vec<GemmaContextEvent> {
        let q = self.inner.lock().expect("bridge mutex poisoned");
        let take = n.min(q.len());
        // Skip the older overflow; return the last `take` items
        // in insertion order (oldest of the window first).
        q.iter().skip(q.len() - take).cloned().collect()
    }

    fn len(&self) -> usize {
        self.inner.lock().expect("bridge mutex poisoned").len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(ts: i64) -> GemmaContextEvent {
        GemmaContextEvent::AutoTradeSignal {
            timestamp_unix_ms: ts,
            symbol: "EUR/USD".to_string(),
            direction: SignalDirection::Long,
            confidence: 0.78,
        }
    }

    #[test]
    fn ring_buffer_starts_empty() {
        let b = InMemoryModelGemmaBridge::new();
        assert!(b.is_empty());
        assert_eq!(b.recent(5), Vec::<GemmaContextEvent>::new());
    }

    #[test]
    fn push_then_recent_returns_in_insertion_order() {
        let b = InMemoryModelGemmaBridge::with_capacity(5);
        for ts in 1..=3 {
            b.push(signal(ts));
        }
        let out = b.recent(5);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].timestamp_unix_ms(), 1);
        assert_eq!(out[2].timestamp_unix_ms(), 3);
    }

    #[test]
    fn ring_buffer_drops_oldest_when_full() {
        let b = InMemoryModelGemmaBridge::with_capacity(3);
        for ts in 1..=5 {
            b.push(signal(ts));
        }
        let out = b.recent(10);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].timestamp_unix_ms(), 3);
        assert_eq!(out[2].timestamp_unix_ms(), 5);
    }

    #[test]
    fn recent_with_smaller_n_returns_latest_window() {
        let b = InMemoryModelGemmaBridge::with_capacity(10);
        for ts in 1..=10 {
            b.push(signal(ts));
        }
        let out = b.recent(3);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].timestamp_unix_ms(), 8);
        assert_eq!(out[2].timestamp_unix_ms(), 10);
    }

    #[test]
    fn capacity_zero_is_normalized_to_one() {
        // Pin behaviour: a zero capacity would deadlock the
        // ring-buffer logic; we clamp to 1.
        let b = InMemoryModelGemmaBridge::with_capacity(0);
        b.push(signal(42));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn timestamp_accessor_unifies_event_variants() {
        let cases = [
            signal(100),
            GemmaContextEvent::TrainingUpdated {
                timestamp_unix_ms: 200,
                job_id: "job-1".to_string(),
                status: "completed".to_string(),
            },
            GemmaContextEvent::NewsBlackout {
                timestamp_unix_ms: 300,
                active: true,
                reason: "FOMC".to_string(),
            },
            GemmaContextEvent::TradeLifecycle {
                timestamp_unix_ms: 400,
                order_code: "x".to_string(),
                origin: TradeOrigin::Gemma,
                new_status: "Filled".to_string(),
            },
        ];
        let expected = [100, 200, 300, 400];
        for (case, want) in cases.iter().zip(expected.iter()) {
            assert_eq!(case.timestamp_unix_ms(), *want);
        }
    }

    #[test]
    fn trade_origin_distinguishes_manual_ai_and_gemma() {
        assert_ne!(TradeOrigin::Manual, TradeOrigin::Ai);
        assert_ne!(TradeOrigin::Ai, TradeOrigin::Gemma);
        assert_ne!(TradeOrigin::Manual, TradeOrigin::Gemma);
    }

    #[test]
    fn event_round_trips_through_json() {
        // The bridge events flow through the audit log (G7);
        // pin the JSON shape now so the audit-log writer
        // doesn't surprise us later.
        let evt = signal(1700_000_000_000);
        let json = serde_json::to_string(&evt).expect("ser");
        assert!(json.contains("auto_trade_signal"));
        let back: GemmaContextEvent = serde_json::from_str(&json).expect("de");
        assert_eq!(back, evt);
    }
}
