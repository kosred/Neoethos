//! Shared cache of live spot ticks per cTrader `symbol_id` (#137).
//!
//! Background `live_spots_streamer` updates this on every incoming
//! `ProtoOASpotEvent`; HTTP `/live/spots` endpoint + Flutter chart
//! widget read from it. Display rows remain compatible; checked financial
//! reads additionally require the active account-bound connection and raw sides.
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

use super::ctrader_live_auth::CTraderEnvironment;
use neoethos_core::utils::now_unix_ms;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use tokio::sync::broadcast;

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
    /// Last seen bid, decoded from protocol units and rounded for display.
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

/// In-process connection identity, never serialized or accepted as trading permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpotSessionId(u64);

#[cfg(test)]
impl SpotSessionId {
    pub(crate) fn synthetic_for_test(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SessionBinding {
    id: SpotSessionId,
    account_id: i64,
    environment: CTraderEnvironment,
}

/// A checked cache observation, not an order-admission or broker-financial permit.
#[derive(Debug, Clone, PartialEq)]
pub struct FreshSpotQuote {
    pub session_id: SpotSessionId,
    pub account_id: i64,
    pub environment: CTraderEnvironment,
    pub symbol_id: i64,
    pub bid: f64,
    pub ask: f64,
    pub bid_received_at_unix_ms: i64,
    pub ask_received_at_unix_ms: i64,
    pub bid_broker_timestamp_ms: i64,
    pub ask_broker_timestamp_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpotQuoteRefusal {
    CacheUnavailable,
    InvalidTimeBudget,
    NoActiveSession,
    SessionMismatch,
    MissingQuote,
    MissingBid,
    MissingAsk,
    MissingBidTimestamp,
    MissingAskTimestamp,
    InvalidBidTimestamp,
    InvalidAskTimestamp,
    StaleBid,
    StaleAsk,
    FutureBid,
    FutureAsk,
    InvalidPrices,
}

#[derive(Debug, Clone, Copy)]
struct SideObservation {
    raw_price: u64,
    received_at_unix_ms: i64,
    broker_timestamp_ms: Option<i64>,
}

#[derive(Debug)]
struct CachedSpotTick {
    display: SpotTick,
    session: Option<SessionBinding>,
    bid: Option<SideObservation>,
    ask: Option<SideObservation>,
}

#[derive(Default)]
struct SpotCache {
    rows: HashMap<i64, CachedSpotTick>,
    active_session: Option<SessionBinding>,
    stream_generation: u64,
    last_session_id: u64,
}

impl SpotCache {
    fn begin_session(
        &mut self,
        account_id: i64,
        environment: CTraderEnvironment,
        generation: u64,
    ) -> Option<SessionBinding> {
        if account_id <= 0 || generation < self.stream_generation {
            return None;
        }
        self.active_session = None;
        self.stream_generation = generation;
        self.last_session_id = self.last_session_id.checked_add(1)?;
        let binding = SessionBinding {
            id: SpotSessionId(self.last_session_id),
            account_id,
            environment,
        };
        self.active_session = Some(binding);
        Some(binding)
    }

    fn invalidate_generation(&mut self, generation: u64) {
        if generation >= self.stream_generation {
            self.stream_generation = generation;
            self.active_session = None;
        }
    }

    fn end_session(&mut self, binding: SessionBinding) {
        // An old connection's delayed teardown must not invalidate its replacement.
        if self.active_session == Some(binding) {
            self.active_session = None;
        }
    }

    fn update(
        &mut self,
        mut display: SpotTick,
        session: Option<SessionBinding>,
        mut bid: Option<SideObservation>,
        mut ask: Option<SideObservation>,
    ) -> Option<SpotTick> {
        if session.is_some() && self.active_session != session {
            return None;
        }
        if let Some(previous) = self.rows.get(&display.symbol_id) {
            // Display-only legacy updates remain untrusted, even after a trusted row.
            // A new connection may never borrow the old connection's opposite side.
            if session.is_none()
                || (session == previous.session
                    && display.symbol_name == previous.display.symbol_name)
            {
                if display.symbol_name.is_empty() {
                    display.symbol_name = previous.display.symbol_name.clone();
                }
                for (new_side, old_side, new_price, old_price) in [
                    (
                        &mut bid,
                        previous.bid,
                        &mut display.bid,
                        previous.display.bid,
                    ),
                    (
                        &mut ask,
                        previous.ask,
                        &mut display.ask,
                        previous.display.ask,
                    ),
                ] {
                    if let (Some(new), Some(old)) = (*new_side, old_side) {
                        if let (Some(new_ts), Some(old_ts)) =
                            (new.broker_timestamp_ms, old.broker_timestamp_ms)
                        {
                            if old_ts > 0
                                && old_ts <= old.received_at_unix_ms
                                && new_ts > 0
                                && new_ts < old_ts
                            {
                                *new_side = None;
                                *new_price = None;
                            }
                        }
                    }
                    if new_price.is_none() {
                        *new_price = old_price;
                        if session.is_some() {
                            *new_side = old_side;
                        }
                    }
                }
                display.broker_timestamp_ms = match (
                    display.broker_timestamp_ms,
                    previous.display.broker_timestamp_ms,
                ) {
                    (Some(new), Some(old)) => Some(new.max(old)),
                    (new, old) => new.or(old),
                };
            }
        }
        self.rows.insert(
            display.symbol_id,
            CachedSpotTick {
                display: display.clone(),
                session,
                bid,
                ask,
            },
        );
        Some(display)
    }

    fn fresh_quote(
        &self,
        expected_account_id: i64,
        expected_environment: CTraderEnvironment,
        expected_session: SpotSessionId,
        symbol_id: i64,
        now_ms: i64,
        max_age_ms: i64,
    ) -> Result<FreshSpotQuote, SpotQuoteRefusal> {
        if now_ms <= 0 || max_age_ms < 0 {
            return Err(SpotQuoteRefusal::InvalidTimeBudget);
        }
        let active = self
            .active_session
            .ok_or(SpotQuoteRefusal::NoActiveSession)?;
        if active.account_id != expected_account_id
            || active.environment != expected_environment
            || active.id != expected_session
        {
            return Err(SpotQuoteRefusal::SessionMismatch);
        }
        let row = self
            .rows
            .get(&symbol_id)
            .ok_or(SpotQuoteRefusal::MissingQuote)?;
        if row.session != Some(active) {
            return Err(SpotQuoteRefusal::SessionMismatch);
        }
        let bid = row.bid.ok_or(SpotQuoteRefusal::MissingBid)?;
        let ask = row.ask.ok_or(SpotQuoteRefusal::MissingAsk)?;
        let bid_ts = validate_side_age(bid, now_ms, max_age_ms, true)?;
        let ask_ts = validate_side_age(ask, now_ms, max_age_ms, false)?;
        // Protocol prices are unsigned 1/100000 units. Do not use display rounding
        // or guess symbol digits, and do not silently round oversized integers.
        const MAX_EXACT_PRICE: u64 = 1_u64 << 53;
        if bid.raw_price == 0
            || ask.raw_price == 0
            || bid.raw_price > MAX_EXACT_PRICE
            || ask.raw_price > MAX_EXACT_PRICE
            || bid.raw_price > ask.raw_price
        {
            return Err(SpotQuoteRefusal::InvalidPrices);
        }
        let bid_price = bid.raw_price as f64 / 100_000.0;
        let ask_price = ask.raw_price as f64 / 100_000.0;
        if !bid_price.is_finite() || !ask_price.is_finite() || bid_price > ask_price {
            return Err(SpotQuoteRefusal::InvalidPrices);
        }
        Ok(FreshSpotQuote {
            session_id: active.id,
            account_id: active.account_id,
            environment: active.environment,
            symbol_id,
            bid: bid_price,
            ask: ask_price,
            bid_received_at_unix_ms: bid.received_at_unix_ms,
            ask_received_at_unix_ms: ask.received_at_unix_ms,
            bid_broker_timestamp_ms: bid_ts,
            ask_broker_timestamp_ms: ask_ts,
        })
    }
}

fn validate_side_age(
    side: SideObservation,
    now_ms: i64,
    max_age_ms: i64,
    is_bid: bool,
) -> Result<i64, SpotQuoteRefusal> {
    let timestamp = side.broker_timestamp_ms.ok_or(if is_bid {
        SpotQuoteRefusal::MissingBidTimestamp
    } else {
        SpotQuoteRefusal::MissingAskTimestamp
    })?;
    if timestamp <= 0 || side.received_at_unix_ms <= 0 {
        return Err(if is_bid {
            SpotQuoteRefusal::InvalidBidTimestamp
        } else {
            SpotQuoteRefusal::InvalidAskTimestamp
        });
    }
    if timestamp > side.received_at_unix_ms || side.received_at_unix_ms > now_ms {
        return Err(if is_bid {
            SpotQuoteRefusal::FutureBid
        } else {
            SpotQuoteRefusal::FutureAsk
        });
    }
    if now_ms - timestamp > max_age_ms || now_ms - side.received_at_unix_ms > max_age_ms {
        return Err(if is_bid {
            SpotQuoteRefusal::StaleBid
        } else {
            SpotQuoteRefusal::StaleAsk
        });
    }
    Ok(timestamp)
}

static CACHE: OnceLock<RwLock<SpotCache>> = OnceLock::new();

fn cache() -> &'static RwLock<SpotCache> {
    CACHE.get_or_init(|| RwLock::new(SpotCache::default()))
}

/// Owns financial freshness for exactly one connection, including all error/unwind exits.
pub(super) struct SpotSessionGuard<'a> {
    binding: SessionBinding,
    state: &'a RwLock<SpotCache>,
}

impl Drop for SpotSessionGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.write() {
            state.end_session(self.binding);
        }
    }
}

pub(super) fn begin_session(
    account_id: i64,
    environment: CTraderEnvironment,
    generation: u64,
) -> Option<SpotSessionGuard<'static>> {
    begin_session_in(cache(), account_id, environment, generation)
}

fn begin_session_in(
    state: &RwLock<SpotCache>,
    account_id: i64,
    environment: CTraderEnvironment,
    generation: u64,
) -> Option<SpotSessionGuard<'_>> {
    let binding = state
        .write()
        .ok()?
        .begin_session(account_id, environment, generation)?;
    Some(SpotSessionGuard { binding, state })
}

pub(super) fn invalidate_stream_generation(generation: u64) {
    if let Ok(mut state) = cache().write() {
        state.invalidate_generation(generation);
    }
}

/// Capture the active connection identity for a pinned account and environment.
/// This does not prove that either side of any quote is presently usable.
pub fn current_session(
    expected_account_id: i64,
    expected_environment: CTraderEnvironment,
) -> Option<SpotSessionId> {
    let active = cache().read().ok()?.active_session?;
    (active.account_id == expected_account_id && active.environment == expected_environment)
        .then_some(active.id)
}

/// Read raw-wire-derived sides only after checking their exact connection identity
/// and independent broker/receive ages. This is not a trading permission.
pub fn get_fresh_tick(
    expected_account_id: i64,
    expected_environment: CTraderEnvironment,
    expected_session: SpotSessionId,
    symbol_id: i64,
    now_ms: i64,
    max_age_ms: i64,
) -> Result<FreshSpotQuote, SpotQuoteRefusal> {
    cache()
        .read()
        .map_err(|_| SpotQuoteRefusal::CacheUnavailable)?
        .fresh_quote(
            expected_account_id,
            expected_environment,
            expected_session,
            symbol_id,
            now_ms,
            max_age_ms,
        )
}

/// **2026-05-25 — operator directive "push, not poll"**:
/// broadcast channel that fires on every cache update. The SSE
/// endpoint `/live/spots/stream` subscribes here and forwards each
/// tick to the Flutter UI as it arrives. Push latency is now
/// bounded by network RTT (~1-5 ms on localhost / LAN) instead of
/// the polling interval (~1000 ms before).
///
/// Capacity 1024 = generous buffer for slow consumers. If a Flutter
/// client falls behind by more than 1024 ticks, oldest ticks drop
/// — but since the cache always has the latest-known value per
/// symbol, the client can resync via `GET /live/spots` once it
/// catches up. So a dropped broadcast is never a correctness issue,
/// only a UX latency blip on the affected symbol.
const SPOT_BROADCAST_CAPACITY: usize = 1024;
static SPOT_BROADCAST: OnceLock<broadcast::Sender<SpotTick>> = OnceLock::new();

fn broadcaster() -> &'static broadcast::Sender<SpotTick> {
    SPOT_BROADCAST.get_or_init(|| broadcast::channel(SPOT_BROADCAST_CAPACITY).0)
}

/// Subscribe to the live-tick broadcast stream. Each call returns a
/// fresh receiver; multiple Flutter clients can subscribe
/// simultaneously without interfering. The receiver yields one
/// `SpotTick` per broadcast frame.
///
/// The SSE handler in `server/live_spots.rs::stream` calls this and
/// adapts the receiver into an `axum::response::sse::Sse` stream.
pub fn subscribe() -> broadcast::Receiver<SpotTick> {
    broadcaster().subscribe()
}

/// Display-only compatibility update. It never creates financial provenance.
pub fn update_tick(
    symbol_id: i64,
    symbol_name: impl Into<String>,
    bid: Option<f64>,
    ask: Option<f64>,
    broker_timestamp_ms: Option<i64>,
) {
    publish_update(
        cache(),
        SpotTick {
            symbol_id,
            symbol_name: symbol_name.into(),
            bid,
            ask,
            received_at_unix_ms: now_unix_ms(),
            broker_timestamp_ms,
        },
        None,
        None,
        None,
    );
}

/// The authenticated streamer supplies original unsigned protocol sides.
pub(super) fn update_session_tick(
    session: &SpotSessionGuard<'_>,
    symbol_id: i64,
    symbol_name: &str,
    digits: i32,
    bid: Option<u64>,
    ask: Option<u64>,
    broker_timestamp_ms: Option<i64>,
) {
    let now_ms = now_unix_ms();
    let side = |raw_price| SideObservation {
        raw_price,
        received_at_unix_ms: now_ms,
        broker_timestamp_ms,
    };
    publish_update(
        session.state,
        SpotTick {
            symbol_id,
            symbol_name: symbol_name.to_owned(),
            bid: bid.map(|value| display_price(value, digits)),
            ask: ask.map(|value| display_price(value, digits)),
            received_at_unix_ms: now_ms,
            broker_timestamp_ms,
        },
        Some(session.binding),
        bid.map(side),
        ask.map(side),
    );
}

fn display_price(value: u64, digits: i32) -> f64 {
    let raw = value as f64 / 100_000.0;
    let factor = 10_f64.powi(digits.max(0));
    (raw * factor).round() / factor
}

fn publish_update(
    state: &RwLock<SpotCache>,
    tick: SpotTick,
    session: Option<SessionBinding>,
    bid: Option<SideObservation>,
    ask: Option<SideObservation>,
) {
    let updated = state
        .write()
        .ok()
        .and_then(|mut state| state.update(tick, session, bid, ask));
    if let Some(tick) = updated {
        // No subscribers is normal; rejected old-session updates never broadcast.
        let _ = broadcaster().send(tick);
    }
}

/// Snapshot every cached tick. Newest-by-symbol; ordering is by
/// symbol_id (HashMap iteration order is not preserved, so the
/// HTTP handler sorts after this returns if a stable order
/// matters).
pub fn snapshot_all() -> Vec<SpotTick> {
    cache()
        .read()
        .map(|g| g.rows.values().map(|row| row.display.clone()).collect())
        .unwrap_or_default()
}

/// Look up a single symbol. None when the streamer hasn't seen
/// a tick for that symbol yet (still subscribing, just connected,
/// or the symbol was never subscribed).
pub fn get_tick(symbol_id: i64) -> Option<SpotTick> {
    cache()
        .read()
        .ok()
        .and_then(|g| g.rows.get(&symbol_id).map(|row| row.display.clone()))
}

/// Drop display rows and invalidate financial provenance, preserving unique IDs.
#[allow(dead_code)]
pub fn clear() {
    if let Ok(mut state) = cache().write() {
        state.rows.clear();
        state.active_session = None;
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
    fn update_preserves_previous_quote_side_when_event_is_partial() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        update_tick(1, "EURUSD", Some(1.0850), Some(1.0852), Some(1_700_000_000));
        update_tick(1, "EURUSD", Some(1.0860), None, Some(1_700_000_100));
        let bid_update = get_tick(1).expect("bid update present");
        assert_eq!(bid_update.bid, Some(1.0860));
        assert_eq!(bid_update.ask, Some(1.0852));
        assert_eq!(bid_update.broker_timestamp_ms, Some(1_700_000_100));

        update_tick(1, "EURUSD", None, Some(1.0864), Some(1_700_000_200));
        let ask_update = get_tick(1).expect("ask update present");
        assert_eq!(ask_update.bid, Some(1.0860));
        assert_eq!(ask_update.ask, Some(1.0864));
        assert_eq!(ask_update.broker_timestamp_ms, Some(1_700_000_200));
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
    fn put_raw(
        state: &mut SpotCache,
        session: SessionBinding,
        bid: Option<u64>,
        ask: Option<u64>,
        received_at: i64,
        broker_timestamp: Option<i64>,
    ) {
        let side = |raw_price| SideObservation {
            raw_price,
            received_at_unix_ms: received_at,
            broker_timestamp_ms: broker_timestamp,
        };
        assert!(
            state
                .update(
                    SpotTick {
                        symbol_id: 1,
                        symbol_name: "EURUSD".to_owned(),
                        bid: bid.map(|raw| display_price(raw, 2)),
                        ask: ask.map(|raw| display_price(raw, 2)),
                        received_at_unix_ms: received_at,
                        broker_timestamp_ms: broker_timestamp,
                    },
                    Some(session),
                    bid.map(side),
                    ask.map(side),
                )
                .is_some()
        );
    }

    fn checked(
        state: &SpotCache,
        session: SessionBinding,
        now: i64,
        age: i64,
    ) -> Result<FreshSpotQuote, SpotQuoteRefusal> {
        state.fresh_quote(
            session.account_id,
            session.environment,
            session.id,
            1,
            now,
            age,
        )
    }

    #[test]
    fn partial_side_updates_do_not_refresh_the_other_sides_financial_age() {
        let mut state = SpotCache::default();
        let session = state
            .begin_session(42, CTraderEnvironment::Demo, 0)
            .unwrap();
        put_raw(
            &mut state,
            session,
            Some(108_500),
            Some(108_540),
            1_000,
            Some(1_000),
        );
        put_raw(&mut state, session, Some(108_520), None, 1_900, Some(1_900));
        assert_eq!(
            checked(&state, session, 2_000, 500),
            Err(SpotQuoteRefusal::StaleAsk)
        );
        put_raw(&mut state, session, None, Some(108_550), 2_100, Some(2_100));
        let quote = checked(&state, session, 2_200, 500).unwrap();
        assert_eq!(
            (quote.bid_received_at_unix_ms, quote.ask_received_at_unix_ms),
            (1_900, 2_100)
        );
        assert_eq!(
            (quote.bid_broker_timestamp_ms, quote.ask_broker_timestamp_ms),
            (1_900, 2_100)
        );
        // The display still rounds to two decimals, but finance consumes exact protocol units.
        assert_eq!(state.rows[&1].display.bid, Some(1.09));
        assert_eq!(quote.bid.to_bits(), (108_520_f64 / 100_000.0).to_bits());
        assert_eq!(quote.ask.to_bits(), (108_550_f64 / 100_000.0).to_bits());
    }

    #[test]
    fn missing_broker_time_and_out_of_order_events_cannot_manufacture_freshness() {
        let mut state = SpotCache::default();
        let session = state
            .begin_session(42, CTraderEnvironment::Demo, 0)
            .unwrap();
        put_raw(
            &mut state,
            session,
            Some(100_000),
            Some(100_010),
            1_000,
            Some(900),
        );
        put_raw(&mut state, session, Some(99_000), None, 1_200, Some(800));
        assert_eq!(
            checked(&state, session, 1_300, 500)
                .unwrap()
                .bid_received_at_unix_ms,
            1_000
        );
        put_raw(&mut state, session, Some(100_001), None, 1_400, None);
        assert_eq!(state.rows[&1].display.broker_timestamp_ms, Some(900));
        assert_eq!(
            checked(&state, session, 1_500, 1_000),
            Err(SpotQuoteRefusal::MissingBidTimestamp)
        );
    }

    #[test]
    fn financial_reads_require_exact_identity_and_reject_invalid_prices_and_times() {
        let mut state = SpotCache::default();
        let session = state
            .begin_session(42, CTraderEnvironment::Demo, 0)
            .unwrap();
        put_raw(
            &mut state,
            session,
            Some(100_000),
            Some(100_010),
            1_000,
            Some(1_000),
        );
        assert_eq!(
            state.fresh_quote(99, session.environment, session.id, 1, 1_100, 500),
            Err(SpotQuoteRefusal::SessionMismatch)
        );
        assert_eq!(
            state.fresh_quote(42, CTraderEnvironment::Live, session.id, 1, 1_100, 500),
            Err(SpotQuoteRefusal::SessionMismatch)
        );
        assert_eq!(
            state.fresh_quote(42, session.environment, SpotSessionId(99), 1, 1_100, 500),
            Err(SpotQuoteRefusal::SessionMismatch)
        );
        assert_eq!(
            state.fresh_quote(42, session.environment, session.id, 99, 1_100, 500),
            Err(SpotQuoteRefusal::MissingQuote)
        );
        assert_eq!(
            checked(&state, session, 900, 500),
            Err(SpotQuoteRefusal::FutureBid)
        );
        assert_eq!(
            checked(&state, session, 1_500, -1),
            Err(SpotQuoteRefusal::InvalidTimeBudget)
        );
        assert!(checked(&state, session, 1_500, 500).is_ok());
        assert_eq!(
            checked(&state, session, 1_501, 500),
            Err(SpotQuoteRefusal::StaleBid)
        );
        for (bid, ask) in [(0, 100_010), (100_011, 100_010), (u64::MAX, u64::MAX)] {
            put_raw(
                &mut state,
                session,
                Some(bid),
                Some(ask),
                1_100,
                Some(1_100),
            );
            assert_eq!(
                checked(&state, session, 1_200, 500),
                Err(SpotQuoteRefusal::InvalidPrices)
            );
        }
        put_raw(
            &mut state,
            session,
            Some(100_000),
            Some(100_010),
            1_200,
            Some(1_300),
        );
        assert_eq!(
            checked(&state, session, 1_400, 500),
            Err(SpotQuoteRefusal::FutureBid)
        );
        // A malformed future stamp cannot permanently prevent later valid data.
        put_raw(
            &mut state,
            session,
            Some(100_000),
            Some(100_010),
            1_400,
            Some(1_250),
        );
        assert!(checked(&state, session, 1_500, 500).is_ok());
    }

    #[test]
    fn raw_crossed_prices_are_rejected_even_when_f64_division_collapses_them() {
        let bid = 9_007_199_254_740_990_u64;
        let ask = 9_007_199_254_740_989_u64;
        assert!(bid > ask);
        assert_eq!(bid as f64 / 100_000.0, ask as f64 / 100_000.0);
        let mut state = SpotCache::default();
        let session = state
            .begin_session(42, CTraderEnvironment::Demo, 0)
            .unwrap();
        put_raw(
            &mut state,
            session,
            Some(bid),
            Some(ask),
            1_000,
            Some(1_000),
        );
        assert_eq!(
            checked(&state, session, 1_100, 500),
            Err(SpotQuoteRefusal::InvalidPrices)
        );
    }

    #[test]
    fn reconnect_and_legacy_display_updates_cannot_seed_trusted_opposite_sides() {
        let mut state = SpotCache::default();
        let old = state
            .begin_session(42, CTraderEnvironment::Demo, 0)
            .unwrap();
        put_raw(
            &mut state,
            old,
            Some(100_000),
            Some(100_010),
            1_000,
            Some(1_000),
        );
        state.invalidate_generation(1);
        assert_eq!(
            checked(&state, old, 1_100, 500),
            Err(SpotQuoteRefusal::NoActiveSession)
        );
        assert!(
            state
                .begin_session(42, CTraderEnvironment::Demo, 0)
                .is_none()
        );
        let current = state
            .begin_session(42, CTraderEnvironment::Demo, 1)
            .unwrap();
        assert_ne!(old.id, current.id);
        state.end_session(old);
        state.invalidate_generation(0);
        assert_eq!(state.active_session, Some(current));
        put_raw(&mut state, current, Some(100_001), None, 1_100, Some(1_100));
        assert_eq!(
            checked(&state, current, 1_200, 500),
            Err(SpotQuoteRefusal::MissingAsk)
        );
        let stale_tick = state.rows[&1].display.clone();
        assert!(
            state
                .update(stale_tick.clone(), Some(old), None, None)
                .is_none()
        );
        assert!(state.update(stale_tick, None, None, None).is_some());
        assert_eq!(
            checked(&state, current, 1_200, 500),
            Err(SpotQuoteRefusal::SessionMismatch)
        );
    }

    #[test]
    fn actual_cache_session_guard_invalidates_only_its_own_connection() {
        // A local instance exercises the actual guard/update path without racing
        // the supervisor's separate legacy global-cache test.
        let state = RwLock::new(SpotCache::default());
        let old = begin_session_in(&state, 42, CTraderEnvironment::Demo, 0).unwrap();
        let old_id = old.binding.id;
        let current = begin_session_in(&state, 42, CTraderEnvironment::Demo, 0).unwrap();
        let binding = current.binding;
        assert_ne!(old_id, binding.id);
        let timestamp = now_unix_ms() - 1;
        update_session_tick(
            &current,
            1,
            "EURUSD",
            5,
            Some(108_500),
            Some(108_520),
            Some(timestamp),
        );
        update_session_tick(
            &old,
            1,
            "EURUSD",
            5,
            Some(99_000),
            Some(99_010),
            Some(timestamp),
        );
        drop(old);
        let quote = checked(&state.read().unwrap(), binding, now_unix_ms(), 10_000).unwrap();
        assert_eq!((quote.bid, quote.ask), (1.085, 1.0852));
        let display = serde_json::to_value(&state.read().unwrap().rows[&1].display).unwrap();
        let mut keys: Vec<_> = display
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "ask",
                "bid",
                "broker_timestamp_ms",
                "received_at_unix_ms",
                "symbol_id",
                "symbol_name"
            ]
        );
        drop(current);
        assert_eq!(
            checked(&state.read().unwrap(), binding, now_unix_ms(), 10_000),
            Err(SpotQuoteRefusal::NoActiveSession)
        );
        assert!(
            state.read().unwrap().rows.contains_key(&1),
            "display compatibility remains after disconnect"
        );
        let unwound = std::panic::catch_unwind(|| {
            let _session = begin_session_in(&state, 42, CTraderEnvironment::Demo, 0).unwrap();
            panic!("synthetic connection unwind");
        });
        assert!(unwound.is_err());
        assert_eq!(state.read().unwrap().active_session, None);
    }

    #[test]
    fn each_side_reports_its_own_missing_invalid_stale_and_future_time() {
        for (is_bid, missing, invalid, stale, future) in [
            (
                true,
                SpotQuoteRefusal::MissingBidTimestamp,
                SpotQuoteRefusal::InvalidBidTimestamp,
                SpotQuoteRefusal::StaleBid,
                SpotQuoteRefusal::FutureBid,
            ),
            (
                false,
                SpotQuoteRefusal::MissingAskTimestamp,
                SpotQuoteRefusal::InvalidAskTimestamp,
                SpotQuoteRefusal::StaleAsk,
                SpotQuoteRefusal::FutureAsk,
            ),
        ] {
            for (broker_timestamp_ms, received_at_unix_ms, expected) in [
                (None, 1000, missing),
                (Some(0), 1000, invalid),
                (Some(800), 1000, stale),
                (Some(1200), 1000, future),
            ] {
                let side = SideObservation {
                    raw_price: 100_000,
                    received_at_unix_ms,
                    broker_timestamp_ms,
                };
                assert_eq!(validate_side_age(side, 1100, 200, is_bid), Err(expected));
            }
        }
    }

    #[test]
    fn display_price_handles_5_digit_forex_pair() {
        assert_eq!(display_price(108_500, 5), 1.085);
    }
}
