//! Backend news-research service — the data source behind `GET /news/feed`.
//!
//! **Distribution-safe by design.** The pipeline has two stages and
//! neither requires anything installed on the end-user's machine beyond
//! the app itself plus a ChatGPT login:
//!
//!   1. **Fetch** — pull public, *no-API-key* financial RSS/Atom feeds
//!      (ForexLive, DailyFX, MarketWatch, …) server-side with `reqwest`
//!      and parse them with `feed-rs`. No credentials, no key, no CLI.
//!   2. **Summarise** — hand the collected headlines to the operator's
//!      Codex (ChatGPT subscription) through the *existing* direct-HTTP
//!      integration ([`neoethos_codex::CodexClient`]) — the very same
//!      OAuth path the AI Desk chat already uses. Works for ANY user who
//!      has signed in once; no `codex` CLI, MCP server, python or any
//!      third-party API key is involved.
//!
//! Every fallible step is best-effort and fails *soft*: an unreachable
//! feed is skipped and counted (surfaced in `notice`), and a Codex
//! outage simply yields headlines with no AI briefing rather than an
//! error page. The endpoint therefore never 500s on a transient network
//! blip — the UI always has *something* to render.
//!
//! Feed URLs are **operator config** (`NewsConfig.rss_feeds`, editable in
//! Settings → News); this module hardcodes none of them. The only
//! constant here is the internal fetch-coalescing cache window, which is
//! an implementation detail (it bounds how often we hit Codex), not a
//! trading parameter.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use chrono::Utc;
use neoethos_codex::{AuthStore, ChatCompletionRequest, ChatMessage, CodexClient};
use serde::Serialize;
use tokio::sync::Mutex;

/// Per-feed timeout for the RSS fetch. Generous enough for slow feeds,
/// short enough that one dead host can't stall the whole desk (fetches
/// run concurrently, so the wall-clock is the slowest *single* feed).
const FEED_TIMEOUT: Duration = Duration::from_secs(8);

/// Max entries we keep from any single feed before merging — stops one
/// chatty feed from drowning out the others.
const PER_FEED_CAP: usize = 12;

/// Max headlines in the merged, deduped, newest-first feed we return and
/// summarise. Keeps the Codex prompt bounded and the UI list scannable.
const TOTAL_CAP: usize = 30;

/// Headlines we actually paste into the Codex prompt. A briefing doesn't
/// improve past the freshest ~20 stories and the prompt stays cheap.
const SUMMARY_HEADLINES: usize = 20;

/// Fetch-coalescing window. Repeated UI polls / auto-scroll refreshes
/// inside this window reuse the cached feed instead of re-hitting the
/// RSS hosts and (more importantly) Codex. NOT a trading knob — purely
/// an internal rate-limit on outbound calls. A manual refresh bypasses
/// it via `build_feed` (see `?force=true` on the endpoint).
const CACHE_TTL: Duration = Duration::from_secs(600);

/// One headline from a feed.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewsItem {
    pub title: String,
    /// Canonical article URL — the UI's "open in browser" target.
    pub link: String,
    /// Human-readable feed name (the feed's own `<title>`, else its host).
    pub source: String,
    /// Publication time as Unix milliseconds (UTC); `None` when the feed
    /// omitted a date.
    pub published_ms: Option<i64>,
    /// The feed's own short blurb, stripped of HTML. Empty when absent.
    pub blurb: String,
}

/// The full payload `GET /news/feed` returns.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewsFeed {
    /// Merged, deduped, newest-first headlines.
    pub items: Vec<NewsItem>,
    /// Codex-written market briefing (summary + cautious read on risk).
    /// Empty when Codex isn't connected or there were no headlines.
    pub ai_summary: String,
    /// `true` when the Codex briefing succeeded. `false` → the UI shows a
    /// "Connect ChatGPT for an AI briefing" hint instead of a blank box.
    pub ai_available: bool,
    /// Server wall-clock when this feed was assembled (Unix ms, UTC).
    pub generated_at_ms: i64,
    /// Non-fatal diagnostics (e.g. "2 of 4 feeds were unreachable",
    /// "No news feeds configured"). Empty on a fully-clean fetch.
    pub notice: String,
}

/// Process-wide cache of the last successfully-built feed + when it was
/// built. `OnceLock<Mutex<...>>` so the first caller initialises it and
/// every later caller shares one coalescing window.
static CACHE: OnceLock<Mutex<Option<(NewsFeed, Instant)>>> = OnceLock::new();

fn cache() -> &'static Mutex<Option<(NewsFeed, Instant)>> {
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Build the feed, reusing a cached result if it's still inside
/// [`CACHE_TTL`]. This is the default path for UI polls.
pub async fn build_feed_cached(feeds: Vec<String>) -> NewsFeed {
    {
        let guard = cache().lock().await;
        if let Some((feed, built_at)) = guard.as_ref() {
            if built_at.elapsed() < CACHE_TTL {
                return feed.clone();
            }
        }
    }
    let fresh = build_feed(feeds).await;
    // Only cache a feed that actually carries headlines — caching an
    // empty (all-feeds-down) result would pin the desk blank for the
    // whole TTL even after the network recovers.
    if !fresh.items.is_empty() {
        let mut guard = cache().lock().await;
        *guard = Some((fresh.clone(), Instant::now()));
    }
    fresh
}

/// Build the feed unconditionally (skips the cache). Wired to the UI's
/// manual refresh button via `?force=true`.
pub async fn build_feed(feeds: Vec<String>) -> NewsFeed {
    let generated_at_ms = Utc::now().timestamp_millis();

    if feeds.is_empty() {
        return NewsFeed {
            items: Vec::new(),
            ai_summary: String::new(),
            ai_available: false,
            generated_at_ms,
            notice: "No news feeds configured — add RSS feeds in Settings → News.".to_string(),
        };
    }

    let total_feeds = feeds.len();
    let (mut items, failures) = fetch_all(feeds).await;

    // Dedup by canonical link (some feeds syndicate the same wire story),
    // then sort newest-first; undated items sink to the bottom.
    dedup_by_link(&mut items);
    items.sort_by(|a, b| b.published_ms.unwrap_or(0).cmp(&a.published_ms.unwrap_or(0)));
    items.truncate(TOTAL_CAP);

    let ai_summary = summarise(&items).await.unwrap_or_default();
    let ai_available = !ai_summary.is_empty();

    let notice = if items.is_empty() {
        if failures > 0 {
            format!("All {total_feeds} news feeds were unreachable — check your connection or feed URLs in Settings → News.")
        } else {
            "No headlines returned by the configured feeds.".to_string()
        }
    } else if failures > 0 {
        format!("{failures} of {total_feeds} feeds were unreachable; showing the rest.")
    } else {
        String::new()
    };

    NewsFeed {
        items,
        ai_summary,
        ai_available,
        generated_at_ms,
        notice,
    }
}

/// Fetch every feed concurrently. Returns the merged items plus the
/// count of feeds that failed (timeout, non-200, or parse error).
async fn fetch_all(feeds: Vec<String>) -> (Vec<NewsItem>, usize) {
    // One shared client (connection pool, gzip). A spoofed desktop UA
    // keeps the stricter feeds (some 403 a bare reqwest UA) happy.
    let client = reqwest::Client::builder()
        .timeout(FEED_TIMEOUT)
        .user_agent(concat!(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NeoEthos/",
            env!("CARGO_PKG_VERSION"),
            " news-desk"
        ))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let mut set = tokio::task::JoinSet::new();
    for url in feeds {
        let client = client.clone();
        set.spawn(async move {
            match fetch_one(&client, &url).await {
                Ok(items) => Ok(items),
                Err(err) => Err((url, err)),
            }
        });
    }

    let mut items = Vec::new();
    let mut failures = 0usize;
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(Ok(mut got)) => items.append(&mut got),
            Ok(Err((url, err))) => {
                failures += 1;
                tracing::warn!(
                    target: "neoethos_app::news",
                    url = %url,
                    error = %err,
                    "news feed fetch/parse failed — skipping",
                );
            }
            Err(join_err) => {
                // A fetch task panicked. Don't take the whole desk down.
                failures += 1;
                tracing::warn!(
                    target: "neoethos_app::news",
                    error = %join_err,
                    "news fetch task panicked — skipping",
                );
            }
        }
    }
    (items, failures)
}

/// Fetch + parse a single feed.
async fn fetch_one(client: &reqwest::Client, url: &str) -> anyhow::Result<Vec<NewsItem>> {
    let bytes = client
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;

    let parsed = feed_rs::parser::parse(&bytes[..])
        .map_err(|e| anyhow::anyhow!("not a valid RSS/Atom feed: {e}"))?;

    let source = parsed
        .title
        .as_ref()
        .map(|t| t.content.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| host_of(url));

    let mut out = Vec::new();
    for entry in parsed.entries.into_iter().take(PER_FEED_CAP) {
        let title = entry
            .title
            .as_ref()
            .map(|t| t.content.trim().to_string())
            .unwrap_or_default();
        if title.is_empty() {
            continue;
        }
        let link = entry
            .links
            .first()
            .map(|l| l.href.trim().to_string())
            .unwrap_or_default();
        let published_ms = entry
            .published
            .or(entry.updated)
            .map(|d| d.timestamp_millis());
        let blurb = entry
            .summary
            .as_ref()
            .map(|t| strip_html(&t.content))
            .unwrap_or_default();
        out.push(NewsItem {
            title,
            link,
            source: source.clone(),
            published_ms,
            blurb,
        });
    }
    Ok(out)
}

/// Ask Codex (the operator's ChatGPT subscription, via the existing
/// direct-HTTP integration) for a concise market briefing over the
/// collected headlines. Returns `None` when there's nothing to
/// summarise or Codex is unreachable/not-authenticated — both are
/// non-fatal (the UI just shows headlines without a briefing).
async fn summarise(items: &[NewsItem]) -> Option<String> {
    if items.is_empty() {
        return None;
    }

    let mut headlines = String::new();
    for (i, it) in items.iter().take(SUMMARY_HEADLINES).enumerate() {
        headlines.push_str(&format!("{}. [{}] {}\n", i + 1, it.source, it.title));
    }

    let prompt = format!(
        "Here are today's forex / financial-market headlines pulled from public RSS feeds:\n\n\
         {headlines}\n\
         Write a concise market briefing for an active forex trader:\n\
         1. Three to four sentences on the dominant themes and what is driving them.\n\
         2. A short, cautious read on possible USD and major-pair direction, naming the key risks on both sides.\n\
         Keep the whole thing under 180 words. Describe drivers, mechanics and risk only — do NOT give financial advice or specific buy/sell calls."
    );

    let store = AuthStore::at_default();
    let client = CodexClient::new(store);

    let mut request = ChatCompletionRequest::simple(&prompt);
    request.messages.insert(
        0,
        ChatMessage {
            role: "system".to_string(),
            content: "You are the NeoEthos news desk. You summarise market \
                      news objectively and always foreground risk. You never \
                      give financial advice or specific trade calls."
                .to_string(),
        },
    );

    match client.chat(request).await {
        Ok(resp) => resp
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content.trim().to_string())
            .filter(|s| !s.is_empty()),
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::news",
                error = %err,
                "Codex news briefing unavailable — serving headlines only",
            );
            None
        }
    }
}

/// Drop later items whose `link` we've already seen. Stable: keeps the
/// first occurrence (feeds are appended in completion order, which we
/// re-sort by date afterwards anyway).
fn dedup_by_link(items: &mut Vec<NewsItem>) {
    let mut seen = std::collections::HashSet::new();
    items.retain(|it| {
        // Items with no link can't be deduped meaningfully — keep them.
        if it.link.is_empty() {
            return true;
        }
        seen.insert(it.link.clone())
    });
}

/// Bare host of a URL (sans `www.`), for use as a feed label when the
/// feed itself didn't carry a title.
fn host_of(url: &str) -> String {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.trim_start_matches("www.").to_string()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "news".to_string())
}

/// Lightweight HTML-tag stripper + minimal entity decode for feed
/// blurbs. The result is rendered as PLAIN TEXT in Flutter (never as a
/// WebView), so this is for readability, not sanitisation.
fn strip_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    let decoded = out
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ");
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}
