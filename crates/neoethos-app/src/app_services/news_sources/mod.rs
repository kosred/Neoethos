//! Multi-source news + economic-calendar aggregator (#129).
//!
//! The Gemma news watcher (#128) needs real upstream data to decide
//! when to fire ADAPTIVE_POLL and what to summarise on MORNING_SCAN.
//! The `fetch_url` tool exists but the LLM had to invent the URLs and
//! parse arbitrary HTML — fragile and slow. This module bundles a
//! curated set of trustworthy upstream sources behind a uniform
//! interface so the watcher (and the LLM via two new tools) gets
//! consistent, structured data without having to scrape.
//!
//! ## Sources shipped this commit
//!
//! - **ForexFactory weekly XML calendar** — free, no API key, the
//!   de-facto FX calendar standard. URL:
//!   `https://nfs.faireconomy.media/ff_calendar_thisweek.xml`.
//! - **FXStreet RSS news** — `https://www.fxstreet.com/rss/news`.
//! - **DailyFX RSS news** — `https://www.dailyfx.com/feeds/market-news`.
//! - **Investing.com RSS news** — `https://www.investing.com/rss/news.rss`.
//!
//! Each source is enable-toggleable in `NewsConfig`. The aggregator
//! merges all enabled sources, dedupes calendar events by
//! `(currency, scheduled_at_unix_ms, title)` and headlines by URL +
//! published_at, then sorts by time.
//!
//! ## Boundaries
//!
//! - No API keys (everything is free / public).
//! - SSRF guard NOT applied here — these URLs are hardcoded constants
//!   the operator can override only via env var, not via LLM input.
//!   The `fetch_url` tool retains its SSRF guard for arbitrary URLs.
//! - In-process cache with TTL (default 5 min) so calling the
//!   `get_upcoming_calendar_events` tool twice doesn't re-fetch
//!   immediately.
//! - Timeouts: 10 s per source. A slow source doesn't block the
//!   aggregator; it just returns its existing cache.

pub mod cache;
pub mod forex_factory;
#[cfg(feature = "headless-browser")]
pub mod headless_browser;
pub mod rss_feed;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Severity bucket the source assigns to an event. ForexFactory
/// publishes `Low|Medium|High|Holiday`; we coerce everything else
/// into this shape so downstream consumers don't have to branch on
/// per-source strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NewsImpact {
    Holiday,
    Low,
    Medium,
    High,
    /// Unknown / unrated — most RSS headlines fall here because
    /// they don't carry an impact field at all.
    Unrated,
}

impl NewsImpact {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Holiday => "holiday",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Unrated => "unrated",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "high" => Self::High,
            "medium" => Self::Medium,
            "low" => Self::Low,
            "holiday" => Self::Holiday,
            _ => Self::Unrated,
        }
    }

    /// Numeric weight for sort + "is it impactful enough" predicates.
    /// Higher = more impactful. High = 3, Medium = 2, Low = 1,
    /// Holiday = 0 (markets shut, no impact in the trading sense),
    /// Unrated = 0.
    pub fn weight(self) -> u8 {
        match self {
            Self::High => 3,
            Self::Medium => 2,
            Self::Low => 1,
            Self::Holiday | Self::Unrated => 0,
        }
    }
}

/// Calendar event: scheduled economic release, central-bank speech,
/// holiday, etc. ForexFactory's primary shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalendarEvent {
    /// Three-letter currency code (USD, EUR, GBP, JPY, …).
    pub currency: String,
    /// Event title — e.g. "Non-Farm Employment Change", "ECB Press
    /// Conference", "Bank Holiday".
    pub title: String,
    /// When the event prints, Unix milliseconds (UTC). Sources that
    /// only give a date with no time get the day's 00:00 UTC.
    pub scheduled_at_unix_ms: i64,
    pub impact: NewsImpact,
    /// Forecast value as published by the source (e.g. "+220K").
    /// Free-form string because units / format vary by series.
    pub forecast: Option<String>,
    /// Previous reading.
    pub previous: Option<String>,
    /// Actual print, once published.
    pub actual: Option<String>,
    /// Which source produced this row (used for dedupe + audit).
    pub source: String,
}

/// Free-form news headline. RSS feeds primarily.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Headline {
    pub title: String,
    pub link: String,
    /// Plain-text excerpt (one short paragraph). Stripped of HTML
    /// tags by the RSS parser — feeds that publish HTML in the
    /// description field lose styling on the way through.
    pub summary: String,
    pub published_at_unix_ms: i64,
    pub source: String,
}

/// Single source contract. Implementations may be either calendar
/// (returns events) or headline (returns headlines) or both — the
/// trait splits them so each impl only implements what it can.
pub trait NewsSource: Send + Sync {
    /// Stable identifier (`forex_factory`, `fxstreet`, `dailyfx`,
    /// `investing`). Used as the `source` field on rows.
    fn id(&self) -> &'static str;

    /// Human-readable name for UI / log lines.
    fn display_name(&self) -> &'static str;

    /// Pull the latest calendar events. Default impl returns an
    /// empty Vec for headline-only sources.
    fn fetch_calendar_events(&self) -> Result<Vec<CalendarEvent>> {
        Ok(Vec::new())
    }

    /// Pull the latest headlines. Default impl returns an empty Vec
    /// for calendar-only sources.
    fn fetch_headlines(&self) -> Result<Vec<Headline>> {
        Ok(Vec::new())
    }
}

/// Build the standard registry: ForexFactory + the three RSS feeds.
/// Each source's enable flag is checked at use-time, not here —
/// disabled sources stay in the registry but their fetches are
/// skipped by the aggregator. This keeps the registry shape stable
/// across feature toggles.
pub fn default_sources() -> Vec<Box<dyn NewsSource>> {
    vec![
        Box::new(forex_factory::ForexFactorySource::new()),
        Box::new(rss_feed::RssFeedSource::fxstreet()),
        Box::new(rss_feed::RssFeedSource::dailyfx()),
        Box::new(rss_feed::RssFeedSource::investing()),
    ]
}

/// Currency codes ForexFactory uses for each major pair leg. Used
/// by the ADAPTIVE_POLL predicate to filter "is there a high-impact
/// event for any currency in my watched-symbol set". Hardcoded for
/// the canonical FX majors + commodities; operators trading exotics
/// fall back to the full unfiltered event list.
pub fn currencies_for_symbol(symbol: &str) -> Vec<&'static str> {
    let s = symbol.to_uppercase();
    let mut out = Vec::new();
    for (idx, ccy) in [
        ("USD", "USD"),
        ("EUR", "EUR"),
        ("GBP", "GBP"),
        ("JPY", "JPY"),
        ("CHF", "CHF"),
        ("AUD", "AUD"),
        ("NZD", "NZD"),
        ("CAD", "CAD"),
    ]
    .iter()
    {
        if s.contains(idx) {
            out.push(*ccy);
        }
    }
    // Commodities / index treat as USD-driven by default — gold,
    // silver, oil, equity indices all move with the dollar.
    if s.starts_with("XAU") || s.starts_with("XAG") || s.starts_with("WTI") || s.starts_with("US30")
        || s.starts_with("NAS") || s.starts_with("SPX")
    {
        if !out.contains(&"USD") {
            out.push("USD");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impact_parse_is_case_insensitive() {
        assert_eq!(NewsImpact::parse("HIGH"), NewsImpact::High);
        assert_eq!(NewsImpact::parse("medium"), NewsImpact::Medium);
        assert_eq!(NewsImpact::parse(" low "), NewsImpact::Low);
        assert_eq!(NewsImpact::parse("garbage"), NewsImpact::Unrated);
    }

    #[test]
    fn impact_weight_orders_correctly() {
        assert!(NewsImpact::High.weight() > NewsImpact::Medium.weight());
        assert!(NewsImpact::Medium.weight() > NewsImpact::Low.weight());
        assert_eq!(NewsImpact::Holiday.weight(), 0);
        assert_eq!(NewsImpact::Unrated.weight(), 0);
    }

    #[test]
    fn currencies_for_eurusd_returns_eur_and_usd() {
        let ccys = currencies_for_symbol("EURUSD");
        assert_eq!(ccys, vec!["USD", "EUR"]);
    }

    #[test]
    fn currencies_for_gbpjpy_returns_gbp_and_jpy() {
        let ccys = currencies_for_symbol("GBPJPY");
        assert_eq!(ccys, vec!["GBP", "JPY"]);
    }

    #[test]
    fn currencies_for_xauusd_returns_usd() {
        let ccys = currencies_for_symbol("XAUUSD");
        assert_eq!(ccys, vec!["USD"]);
    }

    #[test]
    fn currencies_for_unknown_symbol_returns_empty() {
        let ccys = currencies_for_symbol("WEIRDX");
        assert!(ccys.is_empty());
    }

    #[test]
    fn default_sources_registers_four_sources() {
        let sources = default_sources();
        assert_eq!(sources.len(), 4);
        let ids: Vec<&str> = sources.iter().map(|s| s.id()).collect();
        assert!(ids.contains(&"forex_factory"));
        assert!(ids.contains(&"fxstreet"));
        assert!(ids.contains(&"dailyfx"));
        assert!(ids.contains(&"investing"));
    }
}
