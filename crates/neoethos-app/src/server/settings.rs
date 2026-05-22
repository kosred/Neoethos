//! `/settings` — app-wide non-risk settings (data dir, news, LLM model).
//!
//! Companion to `/risk`: returns the parts of `Settings` that operators
//! tweak from the Settings tab of the Flutter UI. Same Phase 1 shape:
//! read-only, single GET, returns the in-memory representation of
//! config.yaml.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;

use super::state::AppApiState;

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsDto {
    pub data_dir: String,
    pub news_calendar_enabled: bool,
    pub news_calendar_source: String,
    pub openai_model: String,
}

pub async fn settings(State(_state): State<AppApiState>) -> Response {
    let settings = match Settings::from_yaml("config.yaml") {
        Ok(s) => s,
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::server::settings",
                error = %err,
                "failed to load config.yaml for /settings endpoint"
            );
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "config.yaml not loadable",
                    "code": "config_load_failed",
                })),
            )
                .into_response();
        }
    };

    Json(SettingsDto {
        // `data_dir` lives on `SystemConfig`; `openai_model` lives on
        // `NewsConfig` (verified in `crates/neoethos-core/src/config.rs`).
        // Keep the JSON keys flat so the Flutter side doesn't have to
        // mirror the Rust nesting.
        data_dir: settings.system.data_dir.display().to_string(),
        news_calendar_enabled: settings.news.news_calendar_enabled,
        news_calendar_source: settings.news.news_calendar_source.clone(),
        openai_model: settings.news.openai_model.clone(),
    })
    .into_response()
}
