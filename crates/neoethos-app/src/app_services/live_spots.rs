//! Shared cache of live spot ticks per cTrader `symbol_id` (#137).
//!
//! Background `live_spots_streamer` updates this on every incoming
//! `ProtoOASpotEvent`; HTTP `/live/spots` endpoint + Flutter chart
//! widget read from it. Single source of truth so the UI sees the
//! same tick the bridge would use for PnL calculations.
//!
//! ## Why a global singleton
//!
//! The streamer is one long-running tokio task; the HTTP layer is
//! many short-lived axum handlers. Threading a `Arc<RwLock<...>>`
//! through every layer of state would be a lot of plumbing for
//! something that is conceptually a single in-process broadcast
//! channel. The `OnceLock` makes it explicit that the cache is
//! initialised once and lives for the process lifetime — matching
//! the `pending_actions` module pattern.
//!
//! ## Concurrency
//!
//! `RwLock` because reads vastly outnumber writes. Worst case at
//! 50 ticks/sec across all symbols and 10 concurrent UI clients
//! polling at 1Hz: 50 writes/sec + 10 reads/sec. The RwLock
//! contention is dominated by the writes; no need for a more
//! sophisticated structure (DashMap, sharded locks) yet.

use serde::Serialize;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// One row in the cache. Mirrors enough of
/// `CTraderLiveChartUpdate` for the UI's needs without dragging
/// the trendbar payload along — the chart screen computes its own
/// current-candle delta from `(bid + ask) / 2`, so we only need
/// the two prices and a freshness timestamp.
#[derive(Debug, Clone, Serialize)]
pub struct SpotTick {
    /// cTrader's numeric symbol id. Same key as `symbol_id` in
    /// `/broker/symbols`.
    pub symbol_id: i64,
    /// Human-readable name pulled from the broker's symbol list
    /// at subscription time (e.g. "EURUSD"). Convenient for the
    /// UI so it doesn't have to cross-reference IDs.
    pub symbol_name: String,
    /// Last seen bid, in absolute price units (already divided by
    /// 10^digits).
    pub bid: Option<f64>,
    /// Last seen ask, same units as bid.
    pub ask: Option<f64>,
    /// Unix-ms when the streamer received the spot event.
    /// Useful for the UI freshness badge ("updated 2 s ago").
    pub received_at_unix_ms: i64,
    /// Broker-stamped tick timestamp from `ProtoOASpotEvent.
    /// timestamp` when present. Often missing on free demo
    /// accounts (which is why we keep `received_at` separately).
    pub broker_timestamp_ms: Option<i64>,
}

impl SpotTick {
    /// Mid-price helper for callers that don't care about the
    /// spread. Returns `None` when either side is missing — same
    /// semantics as `CTraderLiveChartUpdate::mid_price`.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.bid, self.ask) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        }
    }
}

static CACHE: OnceLock<RwLock<HashMap<i64, SpotTick>>> = OnceLock::new();

fn cache() -> &'static RwLock<HashMap<i64, SpotTick>> {
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Insert / overwrite the cached row for [symbol_id]. Called by
/// the streamer on every parsed spot event.
pub fn update_tick(
    symbol_id: i64,
    symbol_name: impl Into<String>,
    bid: Option<f64>,
    ask: Option<f64>,
    broker_timestamp_ms: Option<i64>,
) {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let tick = SpotTick {
        symbol_id,
        symbol_name: symbol_name.into(),
        bid,
        ask,
        received_at_unix_ms: now_ms,
        broker_timestamp_ms,
    };
    if let Ok(mut g) = cache().write() {
        g.insert(symbol_id, tick);
    }
}

/// Snapshot every cached tick. Newest-by-symbol; ordering is by
/// symbol_id (HashMap iteration order is not preserved, so the
/// HTTP handler sorts after this returns if a stable order
/// matters).
pub fn snapshot_all() -> Vec<SpotTick> {
    cache()
        .read()
        .map(|g| g.values().cloned().collect())
        .unwrap_or_default()
}

/// Look up a single symbol. None when the streamer hasn't seen
/// a tick for that symbol yet (still subscribing, just connected,
/// or the symbol was never subscribed).
pub fn get_tick(symbol_id: i64) -> Option<SpotTick> {
    cache().read().ok().and_then(|g| g.get(&symbol_id).cloned())
}

/// Drop everything. Used by tests + intended for the streamer
/// when the session is invalidated (e.g. token expired) so stale
/// ticks don't survive a re-auth. The streamer doesn't call this
/// today (it just reconnects and overwrites on next tick), so
/// the function shows up as dead code in non-test builds —
/// allow-listed rather than removed because the streamer-on-
/// reauth use case is on the near-term roadmap.
#[allow(dead_code)]
pub fn clear() {
    if let Ok(mut g) = cache().write() {
        g.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests share the global `CACHE`. Run them serially under one
    /// mutex so a parallel pass doesn't see another test's writes.
    /// Matches the `pending_actions::tests::TEST_LOCK` pattern.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn update_then_snapshot_round_trips() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        update_tick(1, "EURUSD", Some(1.0850), Some(1.0852), Some(1_700_000_000));
        update_tick(2, "GBPUSD", Some(1.2700), Some(1.2702), None);
        let snap = snapshot_all();
        assert_eq!(snap.len(), 2);
        let eur = snap.iter().find(|t| t.symbol_id == 1).expect("eur");
        assert_eq!(eur.symbol_name, "EURUSD");
        assert_eq!(eur.bid, Some(1.0850));
        assert_eq!(eur.ask, Some(1.0852));
        assert_eq!(eur.broker_timestamp_ms, Some(1_700_000_000));
        clear();
    }

    #[test]
    fn update_overwrites_existing_row() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        update_tick(1, "EURUSD", Some(1.0850), Some(1.0852), None);
        update_tick(1, "EURUSD", Some(1.0860), Some(1.0862), None);
        let tick = get_tick(1).expect("present");
        assert_eq!(tick.bid, Some(1.0860));
        assert_eq!(tick.ask, Some(1.0862));
        clear();
    }

    #[test]
    fn mid_price_requires_both_sides() {
        let with_both = SpotTick {
            symbol_id: 1,
            symbol_name: "EURUSD".to_string(),
            bid: Some(1.0850),
            ask: Some(1.0852),
            received_at_unix_ms: 0,
            broker_timestamp_ms: None,
        };
        assert_eq!(with_both.mid_price(), Some(1.0851));

        let no_ask = SpotTick {
            ask: None,
            ..with_both.clone()
        };
        assert_eq!(no_ask.mid_price(), None);
    }

    #[test]
    fn get_tick_returns_none_for_unknown_symbol() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        update_tick(1, "EURUSD", Some(1.0), Some(1.0), None);
        assert!(get_tick(999).is_none());
        clear();
    }
}
