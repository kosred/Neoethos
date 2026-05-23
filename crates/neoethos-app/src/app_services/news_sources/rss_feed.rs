//! Generic RSS feed news source.
//!
//! Powers the FXStreet / DailyFX / Investing.com headline streams
//! behind a single implementation. Each variant is a thin
//! constructor that fills in the feed URL + display name — the
//! parsing is shared.
//!
//! ## RSS shape
//!
//! ```xml
//! <rss version="2.0">
//!   <channel>
//!     <item>
//!       <title>FX update: dollar steady ahead of NFP</title>
//!       <link>https://...</link>
//!       <description>The dollar trades flat ...</description>
//!       <pubDate>Fri, 23 May 2025 08:00:00 GMT</pubDate>
//!     </item>
//!     ...
//!   </channel>
//! </rss>
//! ```
//!
//! We accept both `pubDate` (RFC 2822) and `dc:date` (ISO 8601) for
//! the timestamp. Feeds that publish HTML in `<description>` get
//! their tags stripped to a plain-text summary.

use super::cache;
use super::{Headline, NewsSource};
use anyhow::{Context, Result, anyhow};
use chrono::DateTime;

const FETCH_TIMEOUT_SECS: u64 = 10;
const USER_AGENT: &str = "NeoEthos-NewsAggregator/0.4";

pub struct RssFeedSource {
    id: &'static str,
    display_name: &'static str,
    url: String,
}

impl RssFeedSource {
    /// Build a generic RSS reader. The `url_override_env` is read
    /// once at construction — operators tweaking the feed URL
    /// without rebuilding can set the env var and restart.
    pub fn new(
        id: &'static str,
        display_name: &'static str,
        default_url: &'static str,
        url_override_env: &str,
    ) -> Self {
        let url = std::env::var(url_override_env).unwrap_or_else(|_| default_url.to_string());
        Self {
            id,
            display_name,
            url,
        }
    }

    pub fn fxstreet() -> Self {
        Self::new(
            "fxstreet",
            "FXStreet news RSS",
            "https://www.fxstreet.com/rss/news",
            "NEOETHOS_FXSTREET_RSS_URL",
        )
    }

    pub fn dailyfx() -> Self {
        Self::new(
            "dailyfx",
            "DailyFX market-news RSS",
            "https://www.dailyfx.com/feeds/market-news",
            "NEOETHOS_DAILYFX_RSS_URL",
        )
    }

    pub fn investing() -> Self {
        Self::new(
            "investing",
            "Investing.com news RSS",
            "https://www.investing.com/rss/news.rss",
            "NEOETHOS_INVESTING_RSS_URL",
        )
    }
}

impl NewsSource for RssFeedSource {
    fn id(&self) -> &'static str {
        self.id
    }

    fn display_name(&self) -> &'static str {
        self.display_name
    }

    fn fetch_headlines(&self) -> Result<Vec<Headline>> {
        if let Some(cached) = cache::get_headlines(self.id()) {
            return Ok(cached);
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
            .user_agent(USER_AGENT)
            .build()
            .context("failed to build HTTP client for RSS feed")?;
        let body = match client.get(&self.url).send().and_then(|r| r.text()) {
            Ok(b) => b,
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::news_sources::rss",
                    url = %self.url,
                    error = %err,
                    "failed to fetch RSS feed; serving stale cache if any"
                );
                return Ok(cache::get_headlines_stale(self.id()).unwrap_or_default());
            }
        };
        let headlines = parse_rss(&body, self.id)?;
        cache::put_headlines(self.id(), headlines.clone());
        Ok(headlines)
    }
}

/// Parse an RSS 2.0 feed into our canonical Headline shape. The
/// `source_id` is stamped on every row so the aggregator can audit
/// provenance.
pub fn parse_rss(body: &str, source_id: &str) -> Result<Vec<Headline>> {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    let mut reader = Reader::from_str(body);
    reader.config_mut().trim_text(true);

    let mut headlines = Vec::new();
    let mut buf = Vec::new();
    let mut in_item = false;
    let mut cur_field: Option<String> = None;
    let mut cur = PartialItem::default();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name_bytes = e.name().as_ref().to_vec();
                let name = std::str::from_utf8(&name_bytes).unwrap_or("").to_string();
                if name == "item" {
                    in_item = true;
                    cur = PartialItem::default();
                } else if in_item {
                    cur_field = Some(name);
                }
            }
            Ok(Event::Text(t)) => {
                if in_item
                    && let Some(field) = &cur_field
                {
                    let text = t.unescape().unwrap_or_default().to_string();
                    match field.as_str() {
                        "title" => cur.title.push_str(&text),
                        "link" => cur.link.push_str(&text),
                        "description" => cur.description.push_str(&text),
                        "pubDate" => cur.pub_date.push_str(&text),
                        // dc:date — namespace-prefixed; quick-xml gives
                        // us the full tag name as `dc:date`.
                        "dc:date" => cur.dc_date.push_str(&text),
                        _ => {}
                    }
                }
            }
            Ok(Event::CData(c)) => {
                // Many feeds put HTML descriptions inside CDATA. We
                // keep the raw bytes then strip on finalise.
                if in_item
                    && let Some(field) = &cur_field
                {
                    let text = std::str::from_utf8(c.as_ref()).unwrap_or("").to_string();
                    match field.as_str() {
                        "title" => cur.title.push_str(&text),
                        "description" => cur.description.push_str(&text),
                        _ => {}
                    }
                }
            }
            Ok(Event::End(e)) => {
                let name_bytes = e.name().as_ref().to_vec();
                let name = std::str::from_utf8(&name_bytes).unwrap_or("").to_string();
                if name == "item" {
                    in_item = false;
                    cur_field = None;
                    if let Some(h) = finalize_item(&cur, source_id) {
                        headlines.push(h);
                    }
                } else if in_item && Some(name) == cur_field {
                    cur_field = None;
                }
            }
            Ok(Event::Eof) => break,
            Err(err) => {
                return Err(anyhow!("RSS parse error from {source_id}: {err}"));
            }
            _ => {}
        }
        buf.clear();
    }
    Ok(headlines)
}

#[derive(Default)]
struct PartialItem {
    title: String,
    link: String,
    description: String,
    pub_date: String,
    dc_date: String,
}

fn finalize_item(p: &PartialItem, source_id: &str) -> Option<Headline> {
    if p.title.trim().is_empty() {
        return None;
    }
    let published_at = parse_rss_timestamp(&p.pub_date)
        .or_else(|| parse_rss_timestamp(&p.dc_date))
        .unwrap_or(0);
    Some(Headline {
        title: p.title.trim().to_string(),
        link: p.link.trim().to_string(),
        summary: strip_html_tags(&p.description).trim().to_string(),
        published_at_unix_ms: published_at,
        source: source_id.to_string(),
    })
}

/// Parse RFC 2822 (`Fri, 23 May 2025 08:00:00 GMT`) and ISO 8601
/// timestamps. Returns Unix milliseconds. None when neither shape
/// parses — the caller substitutes 0 so the headline still shows
/// up; the operator just sees "unknown time" downstream.
pub fn parse_rss_timestamp(raw: &str) -> Option<i64> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(dt) = DateTime::parse_from_rfc2822(s) {
        return Some(dt.timestamp_millis());
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.timestamp_millis());
    }
    None
}

/// Minimal HTML-tag stripper for RSS descriptions. Replaces tags
/// with a space (so adjacent words don't merge), then collapses
/// runs of whitespace.
pub fn strip_html_tags(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut inside = false;
    for ch in input.chars() {
        match ch {
            '<' => inside = true,
            '>' => {
                inside = false;
                out.push(' ');
            }
            c if !inside => out.push(c),
            _ => {}
        }
    }
    // Collapse multiple spaces.
    let mut prev_space = false;
    let mut compact = String::with_capacity(out.len());
    for ch in out.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                compact.push(' ');
            }
            prev_space = true;
        } else {
            compact.push(ch);
            prev_space = false;
        }
    }
    compact
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RSS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>FX news</title>
    <item>
      <title>Dollar steady ahead of NFP</title>
      <link>https://example.com/post-1</link>
      <description><![CDATA[The <b>dollar</b> trades flat ahead of <i>NFP</i>.]]></description>
      <pubDate>Fri, 23 May 2025 08:00:00 GMT</pubDate>
    </item>
    <item>
      <title>ECB hints at cut</title>
      <link>https://example.com/post-2</link>
      <description>President Lagarde signalled...</description>
      <pubDate>Thu, 22 May 2025 14:30:00 GMT</pubDate>
    </item>
  </channel>
</rss>"#;

    #[test]
    fn parses_two_items() {
        let h = parse_rss(SAMPLE_RSS, "fxstreet").expect("parse ok");
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn parses_title_link_published_at() {
        let h = parse_rss(SAMPLE_RSS, "fxstreet").expect("parse ok");
        assert_eq!(h[0].title, "Dollar steady ahead of NFP");
        assert_eq!(h[0].link, "https://example.com/post-1");
        assert!(h[0].published_at_unix_ms > 0);
        assert_eq!(h[0].source, "fxstreet");
    }

    #[test]
    fn strips_html_tags_from_description() {
        let h = parse_rss(SAMPLE_RSS, "fxstreet").expect("parse ok");
        // CDATA-wrapped HTML should be flattened.
        assert!(!h[0].summary.contains("<b>"));
        assert!(!h[0].summary.contains("</b>"));
        assert!(h[0].summary.contains("dollar"));
        assert!(h[0].summary.contains("NFP"));
    }

    #[test]
    fn collapses_whitespace_runs() {
        let s = strip_html_tags("<p>hello</p>  <span>world</span>");
        assert_eq!(s.trim(), "hello world");
    }

    #[test]
    fn parses_rfc2822_timestamp() {
        let ts = parse_rss_timestamp("Fri, 23 May 2025 08:00:00 GMT").expect("parse ok");
        // 2025-05-23T08:00:00Z → 1747987200000
        assert_eq!(ts, 1_747_987_200_000);
    }

    #[test]
    fn parses_rfc3339_timestamp() {
        let ts = parse_rss_timestamp("2025-05-23T08:00:00Z").expect("parse ok");
        assert_eq!(ts, 1_747_987_200_000);
    }

    #[test]
    fn returns_none_for_blank_timestamp() {
        assert!(parse_rss_timestamp("").is_none());
        assert!(parse_rss_timestamp("garbage").is_none());
    }

    #[test]
    fn item_with_blank_title_dropped() {
        let xml = r#"<rss><channel><item><title></title><link>x</link></item></channel></rss>"#;
        let h = parse_rss(xml, "fxstreet").expect("parse ok");
        assert!(h.is_empty());
    }
}
