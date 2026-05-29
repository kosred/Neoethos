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
use neoethos_core::{NewsTradingMode, Settings};
use std::path::PathBuf;

use super::state::AppApiState;

/// Path to the canonical `config.yaml` this server reads + writes.
///
// **F-553 + F-576 closure (2026-05-25)**: the per-file `const CONFIG_PATH`
// was removed in favour of the process-wide install on
// `server::state::current_config_path()` so the operator's CLI
// `--config` flag propagates. Local helper keeps the call-sites
// readable without re-introducing the duplication.
fn config_path() -> std::path::PathBuf {
    super::state::current_config_path()
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsDto {
    pub data_dir: String,
    pub news_calendar_enabled: bool,
    pub news_calendar_source: String,
    pub openai_model: String,
    /// `block_on_news` | `allow_always` | `warn_only`. Controls how
    /// the gate behaves during the kill window around high-impact
    /// news events. See [`NewsTradingMode`].
    pub news_trading_mode: String,
    pub news_trading_mode_display_name: String,
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
    /// Snake_case id of a [`NewsTradingMode`] variant.
    pub news_trading_mode: Option<String>,
}

pub async fn settings(State(_state): State<AppApiState>) -> Response {
    let settings = match Settings::from_yaml(config_path()) {
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

/// `GET /settings/raw` — return the raw `config.yaml` contents so the
/// Flutter Settings screen can surface the full 200+ knob configuration
/// the typed `/settings` DTO can't enumerate (#193). The response is
/// `{"yaml": "<file contents>", "path": "<absolute path>"}`.
pub async fn settings_raw_yaml(State(_state): State<AppApiState>) -> Response {
    let path = config_path();
    let absolute = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
    match std::fs::read_to_string(&path) {
        Ok(yaml) => Json(serde_json::json!({
            "yaml": yaml,
            "path": absolute.display().to_string(),
        }))
        .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("failed to read {}: {err}", path.display()),
                "code": "config_read_failed",
            })),
        )
            .into_response(),
    }
}

/// Payload for `POST /settings/raw` — F-312, 2026-05-29.
#[derive(Debug, serde::Deserialize)]
pub struct RawYamlUpdate {
    /// Verbatim new contents of `config.yaml`. Must parse as a YAML
    /// mapping (the top-level structure expected by `Settings`).
    pub yaml: String,
}

/// `POST /settings/raw` — write the entire `config.yaml` verbatim.
///
/// Closes the F-312 silent-drop hole: the typed `POST /settings`
/// (`SettingsUpdateDto`) only knows about 5 fields out of 200+.
/// Operators editing GA / risk / model knobs via the Advanced Settings
/// raw-YAML editor previously saw "Saved." but their edits were
/// silently filtered out by the DTO's strict deserialization.
///
/// This endpoint:
///   1. Parses the submitted body as `serde_yaml_ng::Value` to confirm it
///      is well-formed and a top-level mapping (the shape `Settings`
///      expects). Reject 400 on parse failure with the parser's own
///      error message — much friendlier than letting `Settings` blow
///      up on the next discovery start.
///   2. Re-parses as `Settings` to enforce the typed schema (catches
///      missing required fields, type mismatches). Reject 400 on
///      schema failure with the typed-deserialize error.
///   3. Writes a timestamped backup of the current file alongside it
///      (`config.yaml.bak.<unix-ms>`). Pull-to-restore is then a
///      manual `Copy-Item` away — cheap insurance against a Save
///      button click that the operator regrets.
///   4. Writes the new YAML to the canonical config path atomically
///      (via `write_to_temp + rename`).
///
/// Returns `{ok: true, path: "...", backupPath: "..."}` on success.
pub async fn update_settings_raw_yaml(
    State(_state): State<AppApiState>,
    Json(payload): Json<RawYamlUpdate>,
) -> Response {
    // (1) YAML well-formedness check.
    let parsed_value: serde_yaml_ng::Value = match serde_yaml_ng::from_str(&payload.yaml) {
        Ok(v) => v,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("YAML parse error: {err}"),
                    "code": "yaml_parse_failed",
                })),
            )
                .into_response();
        }
    };
    if !matches!(parsed_value, serde_yaml_ng::Value::Mapping(_)) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "config.yaml must be a top-level YAML mapping \
                          (sections like `system:`, `risk:`, `models:` etc.)",
                "code": "yaml_not_a_mapping",
            })),
        )
            .into_response();
    }

    // (2) Typed schema check. This catches fat-finger field renames
    // before they reach the GA engine. Use the same deserializer that
    // `Settings::from_yaml` uses internally.
    if let Err(err) = serde_yaml_ng::from_str::<neoethos_core::Settings>(&payload.yaml) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("Schema error — your YAML parses but \
                                  doesn't match the Settings struct: {err}"),
                "code": "yaml_schema_failed",
                "hint": "Common causes: typo in a field name, wrong type \
                         (e.g. string where the schema expects a number), \
                         missing required section.",
            })),
        )
            .into_response();
    }

    // (3) Backup the existing file. We accept a missing source (e.g.
    // first save before any seed wrote) but log it so the operator
    // sees something happened.
    let path = config_path();
    let backup_path = match write_backup(&path) {
        Ok(Some(p)) => Some(p),
        Ok(None) => None,
        Err(err) => {
            // Don't block the write on backup failure — log + continue.
            tracing::warn!(
                target: "neoethos_app::server::settings",
                error = %err,
                "failed to write config.yaml backup before raw save \
                 (continuing with the write)"
            );
            None
        }
    };

    // (4) Atomic write via temp file + rename so a crash mid-write
    // can't truncate the live config.
    if let Err(err) = write_atomic(&path, &payload.yaml) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("failed to persist config.yaml: {err}"),
                "code": "config_save_failed",
            })),
        )
            .into_response();
    }

    tracing::info!(
        target: "neoethos_app::server::settings",
        path = %path.display(),
        bytes = payload.yaml.len(),
        "config.yaml updated via POST /settings/raw"
    );

    Json(serde_json::json!({
        "ok": true,
        "path": path.display().to_string(),
        "backupPath": backup_path.map(|p| p.display().to_string()),
        "bytesWritten": payload.yaml.len(),
    }))
    .into_response()
}

/// Write `<path>.bak.<unix-ms>` from the current contents of `path`.
/// Returns `Ok(None)` if the source file doesn't exist yet (first
/// write — nothing to back up). Returns `Ok(Some(backup_path))` on
/// success, `Err(...)` on actual I/O failure.
fn write_backup(path: &std::path::Path) -> std::io::Result<Option<PathBuf>> {
    if !path.exists() {
        return Ok(None);
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let backup = path.with_extension(format!(
        "yaml.bak.{}",
        stamp
    ));
    std::fs::copy(path, &backup)?;
    Ok(Some(backup))
}

/// Write `contents` to `path` atomically via temp-file + rename.
fn write_atomic(path: &std::path::Path, contents: &str) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent)?;
    let tmp = path.with_extension("yaml.tmp");
    std::fs::write(&tmp, contents)?;
    // On Windows, `rename` over an existing file fails — explicitly
    // remove the target first. The temp file stays as the
    // crash-recovery artefact if rename then fails.
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
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
    let mut settings = match Settings::from_yaml(config_path()) {
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
    if let Some(raw) = payload.news_trading_mode {
        let parsed = NewsTradingMode::parse(&raw).ok_or(()).map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "unknown news_trading_mode `{}`. Expected one of: \
                         block_on_news, allow_always, warn_only.",
                        raw
                    ),
                    "code": "invalid_news_trading_mode",
                })),
            )
        });
        match parsed {
            Ok(mode) => settings.news.news_trading_mode = mode,
            Err(resp) => return resp.into_response(),
        }
    }
    if let Err(err) = settings.save(config_path()) {
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
    let mode = settings.news.news_trading_mode;
    SettingsDto {
        data_dir: settings.system.data_dir.display().to_string(),
        news_calendar_enabled: settings.news.news_calendar_enabled,
        news_calendar_source: settings.news.news_calendar_source.clone(),
        openai_model: settings.news.openai_model.clone(),
        news_trading_mode: mode.as_str().to_string(),
        news_trading_mode_display_name: mode.display_name().to_string(),
    }
}
