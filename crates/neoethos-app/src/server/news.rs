//! `GET /news/feed` — the AI news desk.
//!
//! Returns current market headlines (from public no-API-key RSS feeds,
//! fetched server-side) together with a Codex-written market briefing
//! when the operator has connected their ChatGPT subscription. Needs no
//! API key, CLI, MCP server or python on the end-user's machine — see
//! [`crate::app_services::news_research`] for the distribution-safe
//! rationale.
//!
//! Defensive: a config-load failure falls back to the built-in default
//! feed list (via `Settings::default()`), and the service layer never
//! panics on a dead feed or a Codex outage, so this route always returns
//! a well-formed `200` payload the UI can render.

use axum::Json;
use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use serde::Deserialize;

use super::state::AppApiState;
use crate::app_services::news_research;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewsQuery {
    /// When `true`, bypass the fetch-coalescing cache and pull fresh.
    /// Wired to the UI's manual "refresh" button so the operator can
    /// force an update without waiting out the cache window.
    pub force: Option<bool>,
}

/// `GET /news/feed?force=` — headlines + Codex market briefing.
pub async fn feed(State(state): State<AppApiState>, Query(q): Query<NewsQuery>) -> Response {
    // Operator-configured feed URLs. A config-load failure falls back to
    // `Settings::default()`, whose `NewsConfig::default().rss_feeds` is
    // the single source of truth for the built-in defaults — we never
    // hardcode feed URLs in the handler.
    let feeds = Settings::from_yaml(state.config_path())
        .unwrap_or_default()
        .news
        .rss_feeds;

    let payload = if q.force.unwrap_or(false) {
        news_research::build_feed(feeds).await
    } else {
        news_research::build_feed_cached(feeds).await
    };

    Json(payload).into_response()
}
