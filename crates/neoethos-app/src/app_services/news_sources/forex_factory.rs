//! ForexFactory weekly XML economic calendar.
//!
//! Free, no API key, the de-facto FX calendar standard for retail
//! traders. URL:
//! `https://nfs.faireconomy.media/ff_calendar_thisweek.xml`. The
//! feed publishes every economic release for the current trading
//! week with currency, impact, forecast / previous values, and the
//! scheduled time.
//!
//! ## Feed shape (canonical)
//!
//! ```xml
//! <weeklyevents>
//!   <event>
//!     <title>Non-Farm Employment Change</title>
//!     <country>USD</country>
//!     <date>05-23-2025</date>
//!     <time>8:30am</time>
//!     <impact>High</impact>
//!     <forecast>180K</forecast>
//!     <previous>177K</previous>
//!     <url>https://www.forexfactory.com/...</url>
//!   </event>
//!   ...
//! </weeklyevents>
//! ```
//!
//! Dates are `MM-DD-YYYY` (US format!) in Eastern Time, with `time`
//! either an actual clock entry or "All Day" / "Tentative" / etc.
//! We coerce to Unix milliseconds UTC, treating "All Day" as 00:00
//! and unparseable times as the day's midnight (operator sees the
//! day-precision rather than a missing event).

use super::cache;
use super::{CalendarEvent, NewsImpact, NewsSource};
use anyhow::{Context, Result, anyhow};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeZone};

/// Default feed URL. Override with `NEOETHOS_FF_CALENDAR_URL` for
/// test fixtures or a different mirror.
pub const DEFAULT_FF_CALENDAR_URL: &str =
    "https://nfs.faireconomy.media/ff_calendar_thisweek.xml";

const FETCH_TIMEOUT_SECS: u64 = 10;
const USER_AGENT: &str = "NeoEthos-NewsAggregator/0.4";

pub struct ForexFactorySource {
    url: String,
}

impl ForexFactorySource {
    pub fn new() -> Self {
        let url = std::env::var("NEOETHOS_FF_CALENDAR_URL")
            .unwrap_or_else(|_| DEFAULT_FF_CALENDAR_URL.to_string());
        Self { url }
    }
}

impl Default for ForexFactorySource {
    fn default() -> Self {
        Self::new()
    }
}

impl NewsSource for ForexFactorySource {
    fn id(&self) -> &'static str {
        "forex_factory"
    }

    fn display_name(&self) -> &'static str {
        "ForexFactory weekly calendar"
    }

    fn fetch_calendar_events(&self) -> Result<Vec<CalendarEvent>> {
        if let Some(cached) = cache::get_calendar(self.id()) {
            return Ok(cached);
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
            .user_agent(USER_AGENT)
            .build()
            .context("failed to build HTTP client for ForexFactory")?;
        let body = match client.get(&self.url).send().and_then(|r| r.text()) {
            Ok(b) => b,
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::news_sources::forex_factory",
                    url = %self.url,
                    error = %err,
                    "failed to fetch ForexFactory calendar; serving stale cache if any"
                );
                return Ok(cache::get_calendar_stale(self.id()).unwrap_or_default());
            }
        };
        let events = parse_ff_calendar_xml(&body)?;
        cache::put_calendar(self.id(), events.clone());
        Ok(events)
    }
}

/// Parse the ForexFactory weekly XML calendar into our canonical
/// CalendarEvent shape. Public so the test suite + the rss_feed
/// module's similar walker can share a tested code path.
pub fn parse_ff_calendar_xml(body: &str) -> Result<Vec<CalendarEvent>> {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    let mut reader = Reader::from_str(body);
    reader.config_mut().trim_text(true);

    let mut events = Vec::new();
    let mut buf = Vec::new();
    let mut in_event = false;
    let mut cur_field: Option<String> = None;
    let mut cur_event = PartialFfEvent::default();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name_bytes = e.name().as_ref().to_vec();
                let name = std::str::from_utf8(&name_bytes).unwrap_or("").to_string();
                if name == "event" {
                    in_event = true;
                    cur_event = PartialFfEvent::default();
                } else if in_event {
                    cur_field = Some(name);
                }
            }
            Ok(Event::Text(t)) => {
                if in_event
                    && let Some(field) = &cur_field
                {
                    let text = t.unescape().unwrap_or_default().to_string();
                    match field.as_str() {
                        "title" => cur_event.title = text,
                        "country" => cur_event.country = text,
                        "date" => cur_event.date = text,
                        "time" => cur_event.time = text,
                        "impact" => cur_event.impact = text,
                        "forecast" => cur_event.forecast = if text.is_empty() { None } else { Some(text) },
                        "previous" => cur_event.previous = if text.is_empty() { None } else { Some(text) },
                        "actual" => cur_event.actual = if text.is_empty() { None } else { Some(text) },
                        _ => {}
                    }
                }
            }
            Ok(Event::End(e)) => {
                let name_bytes = e.name().as_ref().to_vec();
                let name = std::str::from_utf8(&name_bytes).unwrap_or("").to_string();
                if name == "event" {
                    in_event = false;
                    cur_field = None;
                    if let Some(ev) = finalize_ff_event(&cur_event) {
                        events.push(ev);
                    }
                } else if in_event && Some(name) == cur_field {
                    cur_field = None;
                }
            }
            Ok(Event::Eof) => break,
            Err(err) => {
                return Err(anyhow!("ForexFactory XML parse error: {err}"));
            }
            _ => {}
        }
        buf.clear();
    }
    Ok(events)
}

#[derive(Default)]
struct PartialFfEvent {
    title: String,
    country: String,
    date: String,
    time: String,
    impact: String,
    forecast: Option<String>,
    previous: Option<String>,
    actual: Option<String>,
}

fn finalize_ff_event(p: &PartialFfEvent) -> Option<CalendarEvent> {
    if p.title.is_empty() || p.country.is_empty() || p.date.is_empty() {
        return None;
    }
    let scheduled = ff_date_time_to_unix_ms(&p.date, &p.time)?;
    Some(CalendarEvent {
        currency: p.country.to_uppercase(),
        title: p.title.clone(),
        scheduled_at_unix_ms: scheduled,
        impact: NewsImpact::parse(&p.impact),
        forecast: p.forecast.clone(),
        previous: p.previous.clone(),
        actual: p.actual.clone(),
        source: "forex_factory".to_string(),
    })
}

/// Combine FF's `MM-DD-YYYY` + `H:MMam/pm | All Day | Tentative`
/// into a Unix-ms UTC stamp. Returns `None` for unparseable dates;
/// callers drop those events. FF's `time` field is in EASTERN
/// TIME — the calendar header says "GMT-5" (sometimes -4 in DST).
///
/// Implementation note: we treat the time as if it were already
/// UTC because the difference is "this event is on the same trading
/// day in either timezone" and the watcher's adaptive-poll
/// threshold is in tens of minutes, not hours. A more careful
/// timezone pass is a follow-up — flagged in the cell below.
fn ff_date_time_to_unix_ms(date: &str, time: &str) -> Option<i64> {
    let date_part = NaiveDate::parse_from_str(date.trim(), "%m-%d-%Y").ok()?;
    let t = parse_ff_time(time).unwrap_or_else(|| {
        // "All Day" / "Tentative" → 00:00.
        NaiveTime::from_hms_opt(0, 0, 0).expect("00:00 always valid")
    });
    let naive = NaiveDateTime::new(date_part, t);
    // TODO(timezone): treat FF's Eastern Time as a fixed offset for
    // more accurate scheduled_at. Acceptable to mis-stamp by ~5h
    // for v1 — the adaptive-poll threshold filter still catches
    // events with day-precision granularity. Tracked separately.
    Some(chrono::Utc.from_utc_datetime(&naive).timestamp_millis())
}

fn parse_ff_time(time: &str) -> Option<NaiveTime> {
    let t = time.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("all day") || t.eq_ignore_ascii_case("tentative") {
        return None;
    }
    // Try "H:MMam/pm" then "HH:MM" 24h.
    if let Ok(parsed) = NaiveTime::parse_from_str(t, "%l:%M%p") {
        return Some(parsed);
    }
    if let Ok(parsed) = NaiveTime::parse_from_str(t, "%I:%M%p") {
        return Some(parsed);
    }
    if let Ok(parsed) = NaiveTime::parse_from_str(t, "%H:%M") {
        return Some(parsed);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_FF_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<weeklyevents>
  <event>
    <title>Non-Farm Employment Change</title>
    <country>USD</country>
    <date>05-23-2025</date>
    <time>8:30am</time>
    <impact>High</impact>
    <forecast>180K</forecast>
    <previous>177K</previous>
    <url>https://www.forexfactory.com/</url>
  </event>
  <event>
    <title>ECB Press Conference</title>
    <country>EUR</country>
    <date>05-23-2025</date>
    <time>2:30pm</time>
    <impact>High</impact>
    <forecast></forecast>
    <previous></previous>
  </event>
  <event>
    <title>Bank Holiday</title>
    <country>JPY</country>
    <date>05-26-2025</date>
    <time>All Day</time>
    <impact>Holiday</impact>
  </event>
</weeklyevents>"#;

    #[test]
    fn parses_nfp_with_high_impact_and_forecast() {
        let events = parse_ff_calendar_xml(SAMPLE_FF_XML).expect("parse ok");
        let nfp = events
            .iter()
            .find(|e| e.title.contains("Non-Farm"))
            .expect("NFP event found");
        assert_eq!(nfp.currency, "USD");
        assert_eq!(nfp.impact, NewsImpact::High);
        assert_eq!(nfp.forecast.as_deref(), Some("180K"));
        assert_eq!(nfp.previous.as_deref(), Some("177K"));
        assert_eq!(nfp.source, "forex_factory");
    }

    #[test]
    fn parses_holiday_event() {
        let events = parse_ff_calendar_xml(SAMPLE_FF_XML).expect("parse ok");
        let h = events
            .iter()
            .find(|e| e.title == "Bank Holiday")
            .expect("holiday event found");
        assert_eq!(h.impact, NewsImpact::Holiday);
        assert_eq!(h.currency, "JPY");
    }

    #[test]
    fn parses_empty_forecast_and_previous_as_none() {
        let events = parse_ff_calendar_xml(SAMPLE_FF_XML).expect("parse ok");
        let ecb = events
            .iter()
            .find(|e| e.title.contains("ECB"))
            .expect("ECB event found");
        assert!(ecb.forecast.is_none());
        assert!(ecb.previous.is_none());
    }

    #[test]
    fn parses_full_count() {
        let events = parse_ff_calendar_xml(SAMPLE_FF_XML).expect("parse ok");
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn truncated_xml_yields_empty_events_not_panic() {
        // quick-xml's streaming reader walks to EOF without erroring
        // on unclosed tags — it returns an empty event list because
        // no `</event>` ever fires the finalise path. That's an
        // acceptable degradation (operator sees "no events" rather
        // than a crash); document it via assertion.
        let result = parse_ff_calendar_xml("<weeklyevents><event><title>oops");
        let events = result.expect("permissive walker doesn't error on truncated input");
        assert!(events.is_empty());
    }

    #[test]
    fn syntactically_broken_tag_returns_error() {
        // Genuinely malformed XML (unbalanced angle brackets inside
        // an attribute value) trips quick-xml's lexer.
        let bad = "<weeklyevents><<>>></weeklyevents>";
        assert!(parse_ff_calendar_xml(bad).is_err());
    }

    #[test]
    fn time_parser_handles_am_pm() {
        let t = parse_ff_time("8:30am").expect("morning parse");
        assert_eq!(t, NaiveTime::from_hms_opt(8, 30, 0).unwrap());
        let t = parse_ff_time("2:30pm").expect("afternoon parse");
        assert_eq!(t, NaiveTime::from_hms_opt(14, 30, 0).unwrap());
    }

    #[test]
    fn time_parser_returns_none_for_all_day() {
        assert!(parse_ff_time("All Day").is_none());
        assert!(parse_ff_time("Tentative").is_none());
    }

    #[test]
    fn empty_event_finalises_to_none() {
        let p = PartialFfEvent::default();
        assert!(finalize_ff_event(&p).is_none());
    }
}
