//! Economic-calendar news gate — the data source behind `block_on_news`.
//!
//! Fetches the public, no-API-key ForexFactory weekly calendar JSON
//! (`nfs.faireconomy.media/ff_calendar_thisweek.json`), caches it in-process,
//! and answers ONE question for the live autopilot: *"is `symbol` inside the
//! blackout window of a HIGH-impact event right now?"*
//!
//! Design mirrors `news_research.rs`: distribution-safe (no key, no CLI),
//! best-effort, fails SOFT — an unreachable calendar keeps the last good
//! snapshot; with no snapshot at all the gate answers "clear" (trading is
//! never blocked by our own network hiccup, only by a real known event).
//!
//! Only NEW entries are gated (`live_trading` allows closes during news —
//! closing reduces risk). The window is deliberately conservative-simple:
//! [-15 min, +10 min] around each high-impact event whose currency matches
//! the symbol's base or quote.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::Deserialize;

/// Public weekly calendar feed (same host ForexFactory's own site uses).
const FF_CALENDAR_URL: &str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json";
/// Re-fetch cadence — calendar contents shift rarely (revisions, adds).
const REFRESH_EVERY: Duration = Duration::from_secs(6 * 3600);
/// Blackout window around a high-impact event.
const BLACKOUT_BEFORE_MS: i64 = 15 * 60 * 1000;
const BLACKOUT_AFTER_MS: i64 = 10 * 60 * 1000;

#[derive(Debug, Deserialize)]
struct FfEvent {
    #[serde(default)]
    title: String,
    /// Currency code, e.g. "USD" (ForexFactory calls the field `country`).
    #[serde(default)]
    country: String,
    /// RFC3339 with offset, e.g. "2026-07-03T08:30:00-04:00".
    #[serde(default)]
    date: String,
    /// "High" | "Medium" | "Low" | "Holiday".
    #[serde(default)]
    impact: String,
}

#[derive(Debug, Clone)]
struct CalendarEvent {
    title: String,
    currency: String,
    ts_ms: i64,
}

struct CalendarCache {
    fetched_at: Option<Instant>,
    /// HIGH-impact events only (the ones the gate acts on).
    high_events: Vec<CalendarEvent>,
}

fn cache() -> &'static Mutex<CalendarCache> {
    static CACHE: OnceLock<Mutex<CalendarCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(CalendarCache {
            fetched_at: None,
            high_events: Vec::new(),
        })
    })
}

/// Fetch + parse the weekly feed. Blocking (reqwest::blocking) — callers run
/// on the blocking pool. Errors are returned for the caller to log; the cache
/// keeps its previous contents on failure.
fn fetch_high_impact_events() -> anyhow::Result<Vec<CalendarEvent>> {
    let body = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .user_agent("neoethos-news-gate/1.0")
        .build()?
        .get(FF_CALENDAR_URL)
        .send()?
        .error_for_status()?
        .text()?;
    let raw: Vec<FfEvent> = serde_json::from_str(&body)?;
    let events = raw
        .into_iter()
        .filter(|e| e.impact.eq_ignore_ascii_case("high"))
        .filter_map(|e| {
            let ts_ms = chrono::DateTime::parse_from_rfc3339(&e.date)
                .ok()?
                .timestamp_millis();
            Some(CalendarEvent {
                title: e.title,
                currency: e.country.trim().to_ascii_uppercase(),
                ts_ms,
            })
        })
        .collect();
    Ok(events)
}

/// Refresh the cache when stale/empty. Fail-soft: on error keep what we have.
fn refresh_if_stale() {
    let stale = {
        let Ok(c) = cache().lock() else { return };
        match c.fetched_at {
            None => true,
            Some(t) => t.elapsed() >= REFRESH_EVERY,
        }
    };
    if !stale {
        return;
    }
    match fetch_high_impact_events() {
        Ok(events) => {
            if let Ok(mut c) = cache().lock() {
                tracing::info!(
                    target: "neoethos_app::news_calendar",
                    high_impact_events = events.len(),
                    "economic calendar refreshed (weekly feed)"
                );
                c.high_events = events;
                c.fetched_at = Some(Instant::now());
            }
        }
        Err(e) => {
            tracing::warn!(
                target: "neoethos_app::news_calendar",
                error = %e,
                "economic calendar fetch failed — keeping previous snapshot \
                 (gate stays permissive rather than blocking on OUR outage)"
            );
            // Back off: stamp fetched_at so we don't hammer a dead feed every bar.
            if let Ok(mut c) = cache().lock() {
                c.fetched_at = Some(Instant::now());
            }
        }
    }
}

/// The two currencies a symbol exposes to news risk. Metadata first
/// (XAUUSD → XAU/USD), plain 3+3 split as fallback for 6-char FX names.
fn symbol_currencies(symbol: &str) -> (String, String) {
    if let Some(m) = neoethos_core::symbol_metadata::resolve(symbol) {
        return (m.base.to_ascii_uppercase(), m.quote.to_ascii_uppercase());
    }
    let s = symbol.trim().to_ascii_uppercase();
    if s.len() == 6 {
        (s[..3].to_string(), s[3..].to_string())
    } else {
        (s.clone(), String::new())
    }
}

/// BLOCKING. Returns `Some(description)` when `symbol` is inside the blackout
/// window of a high-impact event **and** the operator's `news_trading_mode`
/// says entries must pause; `None` = clear to trade.
///
/// Honors config live (read per call — the operator can flip the mode in
/// Settings mid-run): `allow_always` or calendar-disabled short-circuit to
/// clear; `warn_only` logs the hit but does not block.
pub fn entry_blackout_for(symbol: &str, now_ms: i64) -> Option<String> {
    let settings =
        neoethos_core::Settings::from_yaml(&crate::server::state::current_config_path()).ok()?;
    if !settings.news.news_calendar_enabled {
        return None;
    }
    use neoethos_core::config::NewsTradingMode;
    let mode = settings.news.news_trading_mode;
    if matches!(mode, NewsTradingMode::AllowAlways) {
        return None;
    }

    refresh_if_stale();

    let (base, quote) = symbol_currencies(symbol);
    let hit = {
        let c = cache().lock().ok()?;
        c.high_events
            .iter()
            .find(|e| {
                (e.currency == base || e.currency == quote)
                    && now_ms >= e.ts_ms - BLACKOUT_BEFORE_MS
                    && now_ms <= e.ts_ms + BLACKOUT_AFTER_MS
            })
            .cloned()
    }?;

    let mins = (hit.ts_ms - now_ms) / 60_000;
    let when = if mins >= 0 {
        format!("in {mins} min")
    } else {
        format!("{} min ago", -mins)
    };
    let desc = format!("{} {} ({when})", hit.currency, hit.title);

    match mode {
        NewsTradingMode::WarnOnly => {
            tracing::warn!(
                target: "neoethos_app::news_calendar",
                %symbol, event = %desc,
                "high-impact news window (warn_only — entry NOT blocked)"
            );
            None
        }
        _ => Some(desc), // BlockOnNews (and any future restrictive mode)
    }
}
