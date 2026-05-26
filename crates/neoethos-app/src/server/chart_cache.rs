//! In-memory LRU cache for `ChartDto` responses.
//!
//! ## Why this exists (operator directive 2026-05-25)
//!
//! Industry research summary — what MT5 / cTrader / TradingView do for
//! timeframe-switch UX:
//!
//! - **TradingView** keeps the *current* timeframe's candles in JS heap
//!   and pre-fetches the previous + next adjacent timeframe so a click
//!   renders from RAM, never from network. They also batch incremental
//!   updates from the WSS into the same in-memory series instead of
//!   re-fetching.
//! - **cTrader** keeps a per-symbol in-memory chart cache server-side
//!   so a TF dropdown click only re-keys into the cache — no disk hit.
//! - **MT5** ships with a local SQLite history database; the terminal
//!   keeps `MaxBars` (default 10k) per (symbol, TF) in RAM at all times.
//!
//! NeoEthos before this module: every TF click ran
//! `load_symbol_timeframe_tail` which reads the entire ~30 MB Vortex
//! file from disk, then sliced. That's 120-500 ms of blocking I/O per
//! click — visible UX lag the operator flagged.
//!
//! This module is the equivalent of TradingView's in-RAM series cache.
//! Same shape: key on `(symbol, timeframe, limit)`, value is the
//! already-built `ChartDto`. A TTL guard keeps the cache from serving
//! stale data once the live bar has progressed.
//!
//! ## Design choices
//!
//! - **Capacity**: 16 entries. Covers a user juggling 4-5 symbols ×
//!   3 timeframes simultaneously — well above realistic concurrent use.
//! - **TTL**: 15 seconds. Even on M1 the current bar doesn't fully form
//!   for 60s; 15s is short enough that the operator never sees a >15s
//!   stale view but long enough to absorb typical click-storm UX.
//! - **Eviction**: LRU. Active charts stay hot; idle ones evict first.
//! - **Concurrency**: `Mutex` over the LRU map. Cache hit + clone takes
//!   ~1 µs; contention is irrelevant under realistic UI load (≤ 5 req/s).
//!
//! ## Live-tick interaction
//!
//! This cache is for the *chart skeleton* (the OHLC bars). Live tick
//! updates flow through a different path
//! (`/live/spots` → `live_spots_streamer`) which already bypasses disk.
//! The Flutter UI overlays the live tick onto the latest cached bar in
//! the JS / Dart side — same pattern as TradingView.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::chart::ChartDto;

/// Tunables. If a future operator preset wants longer/shorter caching,
/// expose these via the knob catalog. For now they live as module-
/// level consts (audit-doctrine: hardcoded operational defaults are
/// fine as long as they're grep-able and documented).
const CAPACITY: usize = 16;
const TTL: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct CacheKey {
    symbol: String,
    timeframe: String,
    limit: usize,
}

struct CacheEntry {
    dto: ChartDto,
    inserted_at: Instant,
    /// LRU rank. Higher = more recently used.
    last_touched_at: Instant,
}

struct Cache {
    entries: HashMap<CacheKey, CacheEntry>,
}

impl Cache {
    fn new() -> Self {
        Self {
            entries: HashMap::with_capacity(CAPACITY),
        }
    }

    fn get(&mut self, key: &CacheKey) -> Option<ChartDto> {
        let entry = self.entries.get_mut(key)?;
        if entry.inserted_at.elapsed() >= TTL {
            // Stale — let the caller miss + refresh. Don't remove eagerly;
            // the `put` path will overwrite, and a concurrent reader would
            // just see the same staleness and re-fetch.
            return None;
        }
        entry.last_touched_at = Instant::now();
        Some(entry.dto.clone())
    }

    fn put(&mut self, key: CacheKey, dto: ChartDto) {
        if self.entries.len() >= CAPACITY && !self.entries.contains_key(&key) {
            // Evict the LRU entry.
            if let Some(victim) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_touched_at)
                .map(|(k, _)| k.clone())
            {
                self.entries.remove(&victim);
            }
        }
        let now = Instant::now();
        self.entries.insert(
            key,
            CacheEntry {
                dto,
                inserted_at: now,
                last_touched_at: now,
            },
        );
    }
}

static CHART_CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

fn cache() -> &'static Mutex<Cache> {
    CHART_CACHE.get_or_init(|| Mutex::new(Cache::new()))
}

/// Cache-key lookup. `None` = miss; caller loads from disk + calls `put`.
pub fn get(symbol: &str, timeframe: &str, limit: usize) -> Option<ChartDto> {
    let key = CacheKey {
        symbol: symbol.to_string(),
        timeframe: timeframe.to_string(),
        limit,
    };
    let mut cache = cache().lock().ok()?;
    cache.get(&key)
}

/// Insert a freshly-loaded `ChartDto`. Overwrites any prior entry for
/// the same key. Triggers LRU eviction when capacity exceeded.
pub fn put(symbol: &str, timeframe: &str, limit: usize, dto: ChartDto) {
    let key = CacheKey {
        symbol: symbol.to_string(),
        timeframe: timeframe.to_string(),
        limit,
    };
    if let Ok(mut cache) = cache().lock() {
        cache.put(key, dto);
    }
}

/// Invalidate every cached entry. Currently only used by the
/// in-module tests to isolate test state; the production
/// invalidation path uses [`clear_symbol`] because both data-
/// modification routes (`/data/fetch` and `/data/import`) operate
/// on a single symbol at a time.
///
/// **Gated `#[cfg(test)]`** to keep the production binary's
/// dead-code surface honest — if we later add a global "reset
/// cache" admin endpoint, lift the gate at that commit.
#[cfg(test)]
fn clear_all() {
    if let Ok(mut cache) = cache().lock() {
        cache.entries.clear();
    }
}

/// Invalidate every entry for a given symbol. More targeted than
/// `clear_all`; called when only one symbol's Vortex file was rewritten.
pub fn clear_symbol(symbol: &str) {
    if let Ok(mut cache) = cache().lock() {
        cache.entries.retain(|k, _| k.symbol != symbol);
    }
}

#[cfg(test)]
mod tests {
    use super::super::chart::{ChartDataSource, ChartDto};
    use super::*;

    fn fake_dto(symbol: &str) -> ChartDto {
        ChartDto {
            symbol: symbol.to_string(),
            timeframe: "M1".to_string(),
            available_timeframes: vec!["M1".to_string()],
            candle_count: 0,
            candles: Vec::new(),
            price_min: 0.0,
            price_max: 0.0,
            latest_close: 0.0,
            price_change_pct: 0.0,
            headline: "test".to_string(),
            source: ChartDataSource::Empty,
        }
    }

    #[test]
    fn hit_after_put() {
        clear_all();
        put("TEST_HIT", "M1", 100, fake_dto("TEST_HIT"));
        let hit = get("TEST_HIT", "M1", 100);
        assert!(hit.is_some(), "freshly inserted entry must be a hit");
        assert_eq!(hit.unwrap().symbol, "TEST_HIT");
    }

    #[test]
    fn miss_on_unknown_key() {
        clear_all();
        let miss = get("NEVER_INSERTED", "M5", 100);
        assert!(miss.is_none());
    }

    #[test]
    fn clear_symbol_drops_only_that_symbol() {
        clear_all();
        put("AAA", "M1", 100, fake_dto("AAA"));
        put("BBB", "M1", 100, fake_dto("BBB"));
        clear_symbol("AAA");
        assert!(get("AAA", "M1", 100).is_none());
        assert!(get("BBB", "M1", 100).is_some());
    }
}
