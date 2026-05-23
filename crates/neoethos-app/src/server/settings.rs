//! `/settings` — app-wide non-risk settings (data dir, news, LLM model).
//!
//! Companion to `/risk`: returns/writes the parts of `Settings` that
//! operators tweak from the Settings tab of the Flutter UI.
//!
//! - GET  → returns the in-memory representation of `config.yaml` as a
//!         flat `SettingsDto` (camelCase).
//! - POST → accepts a partial `SettingsUpdateDto`, merges into the
//!         existing `Settings`, and rewrites `config.yaml` via
//!         `Settings::save()`. Returns the post-merge view.
//!
//! Why merge instead of replace: the on-disk YAML carries ~200+
//! fields across `SystemConfig`, `RiskConfig`, `ModelsConfig`,
//! `NewsConfig`. The UI only exposes a handful — replacing the whole
//! file would silently zero out everything the UI doesn't show.
//! Merging keeps the unexposed knobs intact and only touches what the
//! operator actually edited.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use std::path::PathBuf;

use super::state::AppApiState;

/// Path to the canonical `config.yaml` this server reads + writes.
/// Hardcoded for now — the binary's CWD is the workspace root in dev
/// and the installer's `app_dir` in production. Pulling this into an
/// env var is part of the next-phase SoT refactor.
const CONFIG_PATH: &str = "config.yaml";

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsDto {
    pub data_dir: String,
    pub news_calendar_enabled: bool,
    pub news_calendar_source: String,
    pub openai_model: String,
}

/// Partial-update payload for `POST /settings`. All fields optional —
/// only the ones the caller sends get applied. Unsent fields keep
/// their on-disk value, which is the safe default when the UI ships
/// new controls in stages.
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsUpdateDto {
    pub data_dir: Option<String>,
    pub news_calendar_enabled: Option<bool>,
    pub news_calendar_source: Option<String>,
    pub openai_model: Option<String>,
}

pub async fn settings(State(_state): State<AppApiState>) -> Response {
    let settings = match Settings::from_yaml(CONFIG_PATH) {
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

    Json(dto_from_settings(&settings)).into_response()
}

/// POST /settings — merge-update + persist to config.yaml.
///
/// Validation rules:
/// - `data_dir` is trimmed; rejected if blank (we never want a
///   silently-empty path that breaks downstream readers).
/// - `news_calendar_source` is trimmed; rejected if blank (same reason).
/// - `openai_model` is trimmed but allowed blank (an empty model name
///   disables the LLM news pipeline — operator-intentional).
/// - Booleans pass straight through.
pub async fn update_settings(
    State(_state): State<AppApiState>,
    Json(payload): Json<SettingsUpdateDto>,
) -> Response {
    let mut settings = match Settings::from_yaml(CONFIG_PATH) {
        Ok(s) => s,
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::server::settings",
                error = %err,
                "failed to load config.yaml before POST /settings merge"
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

    if let Some(raw) = payload.data_dir {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "data_dir cannot be blank",
                    "code": "invalid_data_dir",
                })),
            )
                .into_response();
        }
        settings.system.data_dir = PathBuf::from(trimmed);
    }
    if let Some(b) = payload.news_calendar_enabled {
        settings.news.news_calendar_enabled = b;
    }
    if let Some(raw) = payload.news_calendar_source {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "news_calendar_source cannot be blank",
                    "code": "invalid_news_source",
                })),
            )
                .into_response();
        }
        settings.news.news_calendar_source = trimmed.to_string();
    }
    if let Some(raw) = payload.openai_model {
        // Blank is allowed — operator-intentional "disable LLM news".
        settings.news.openai_model = raw.trim().to_string();
    }

    if let Err(err) = settings.save(CONFIG_PATH) {
        tracing::error!(
            target: "neoethos_app::server::settings",
            error = %err,
            "failed to write config.yaml from POST /settings"
        );
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("failed to persist settings: {err}"),
                "code": "config_save_failed",
            })),
        )
            .into_response();
    }

    tracing::info!(
        target: "neoethos_app::server::settings",
        "config.yaml updated via POST /settings"
    );
    Json(dto_from_settings(&settings)).into_response()
}

fn dto_from_settings(settings: &Settings) -> SettingsDto {
    // `data_dir` lives on `SystemConfig`; `openai_model` lives on
    // `NewsConfig` (verified in `crates/neoethos-core/src/config.rs`).
    // Keep the JSON keys flat so the Flutter side doesn't have to
    // mirror the Rust nesting.
    SettingsDto {
        data_dir: settings.system.data_dir.display().to_string(),
        news_calendar_enabled: settings.news.news_calendar_enabled,
        news_calendar_source: settings.news.news_calendar_source.clone(),
        openai_model: settings.news.openai_model.clone(),
    }
}
