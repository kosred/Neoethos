//! In-process TTL cache for news-source fetches.
//!
//! Without this, calling `get_upcoming_calendar_events` twice in a
//! row hits the upstream HTTP service twice, which (a) wastes
//! bandwidth and (b) eats the LLM's available context-window
//! budget on a redundant round-trip. We keep one entry per source
//! id, expire on a fixed TTL (default 5 min), and serve the stale
//! copy if the refresh fails — better to show day-old events than
//! nothing.

use super::{CalendarEvent, Headline};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How long a fetch is considered fresh. Most economic-calendar
/// items don't change inside a single trading session; even a 5 min
/// staleness is fine for the LLM's planning purposes.
pub const CACHE_TTL: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, Clone)]
pub struct CachedCalendar {
    pub events: Vec<CalendarEvent>,
    pub fetched_at: Instant,
}

#[derive(Debug, Clone)]
pub struct CachedHeadlines {
    pub headlines: Vec<Headline>,
    pub fetched_at: Instant,
}

#[derive(Default)]
struct CacheStore {
    calendar: HashMap<&'static str, CachedCalendar>,
    headlines: HashMap<&'static str, CachedHeadlines>,
}

static GLOBAL: OnceLock<Mutex<CacheStore>> = OnceLock::new();

fn store() -> &'static Mutex<CacheStore> {
    GLOBAL.get_or_init(|| Mutex::new(CacheStore::default()))
}

/// Look up cached calendar events for a source. Returns `None` when
/// the cache is empty OR stale beyond TTL.
pub fn get_calendar(source_id: &'static str) -> Option<Vec<CalendarEvent>> {
    let s = store().lock().ok()?;
    let entry = s.calendar.get(source_id)?;
    if entry.fetched_at.elapsed() > CACHE_TTL {
        return None;
    }
    Some(entry.events.clone())
}

/// As above but ignores TTL — returns whatever's there so the
/// aggregator can serve stale data when a refresh fails.
pub fn get_calendar_stale(source_id: &'static str) -> Option<Vec<CalendarEvent>> {
    let s = store().lock().ok()?;
    s.calendar.get(source_id).map(|c| c.events.clone())
}

pub fn put_calendar(source_id: &'static str, events: Vec<CalendarEvent>) {
    if let Ok(mut s) = store().lock() {
        s.calendar.insert(
            source_id,
            CachedCalendar {
                events,
                fetched_at: Instant::now(),
            },
        );
    }
}

pub fn get_headlines(source_id: &'static str) -> Option<Vec<Headline>> {
    let s = store().lock().ok()?;
    let entry = s.headlines.get(source_id)?;
    if entry.fetched_at.elapsed() > CACHE_TTL {
        return None;
    }
    Some(entry.headlines.clone())
}

pub fn get_headlines_stale(source_id: &'static str) -> Option<Vec<Headline>> {
    let s = store().lock().ok()?;
    s.headlines.get(source_id).map(|c| c.headlines.clone())
}

pub fn put_headlines(source_id: &'static str, headlines: Vec<Headline>) {
    if let Ok(mut s) = store().lock() {
        s.headlines.insert(
            source_id,
            CachedHeadlines {
                headlines,
                fetched_at: Instant::now(),
            },
        );
    }
}

#[cfg(test)]
pub fn clear_for_tests() {
    if let Ok(mut s) = store().lock() {
        s.calendar.clear();
        s.headlines.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::news_sources::{CalendarEvent, NewsImpact};

    fn sample_event() -> CalendarEvent {
        CalendarEvent {
            currency: "USD".to_string(),
            title: "Non-Farm Payrolls".to_string(),
            scheduled_at_unix_ms: 1_716_422_400_000,
            impact: NewsImpact::High,
            forecast: Some("+220K".to_string()),
            previous: Some("+180K".to_string()),
            actual: None,
            source: "forex_factory".to_string(),
        }
    }

    // All tests share the global cache; serialise so they don't
    // stomp each other.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn put_then_get_returns_value() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        put_calendar("forex_factory", vec![sample_event()]);
        let got = get_calendar("forex_factory").expect("hit");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].title, "Non-Farm Payrolls");
    }

    #[test]
    fn get_returns_none_for_missing_source() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        assert!(get_calendar("does_not_exist").is_none());
    }

    #[test]
    fn stale_helper_returns_value_regardless_of_age() {
        // We can't actually elapse 5 min in a test, but the helper's
        // contract is that it skips the TTL check. Verify by
        // putting + immediately reading via the stale path.
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        put_calendar("forex_factory", vec![sample_event()]);
        let stale = get_calendar_stale("forex_factory").expect("stale hit");
        assert_eq!(stale.len(), 1);
    }
}
