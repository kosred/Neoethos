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

use super::errors::actionable_error;
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
    /// UI language code (`"en"` | `"el"`) — see `SystemConfig::ui_locale`.
    pub ui_locale: String,
    /// Top-level trading mode (`"risky"` | `"prop_firm"`) — see
    /// `SystemConfig::trading_mode`. Drives discovery + risk orientation.
    pub trading_mode: String,
    /// Compute device preference (`"auto"` | `"cpu"` | `"gpu"`) — see
    /// `SystemConfig::enable_gpu_preference`. `auto` picks the best device and,
    /// with the never-OOM auto-tuner, fits any card; `cpu` forces the CPU lane.
    pub compute_mode: String,
    /// Risky-Mode goal — see `SystemConfig::risky_*`. Start/target balances
    /// (account ccy) + horizon (days). The operator sets these and they
    /// pressure the Risky discovery search toward strategies that can hit the
    /// target in time.
    pub risky_start_balance: f64,
    pub risky_target_balance: f64,
    pub risky_horizon_days: u32,
    pub news_calendar_enabled: bool,
    pub news_calendar_source: String,
    /// `block_on_news` | `allow_always` | `warn_only`. Controls how
    /// the gate behaves during the kill window around high-impact
    /// news events. See [`NewsTradingMode`].
    pub news_trading_mode: String,
    pub news_trading_mode_display_name: String,
    // ── Discovery search budget/quality knobs (models.prop_search_*) ──
    // Surfaced (2026-06-01) so the UI/CLI can tune search depth — the
    // operator's L40 VPS vs local budget — without hand-editing raw YAML.
    pub search_population: usize,
    pub search_generations: usize,
    pub search_max_hours: f64,
    pub search_max_indicators: usize,
    pub search_portfolio_size: usize,
    pub search_corr_threshold: f64,
    pub search_max_rows: usize,
    // ── GA anti-stagnation knobs (models.discovery_runtime / models.search_runtime) ──
    // Surfaced (2026-06-28) so the operator can un-stick the search from Settings.
    pub prefilter_top_k: usize,
    pub convergence_patience: usize,
    pub stagnation_patience: usize,
    pub novelty_weight: f64,
    pub disable_smc_gate: bool,
    /// Portfolio-level concurrent-risk cap (balance fraction; 0 = disabled).
    pub max_portfolio_risk: f64,
}

/// Partial-update payload for `POST /settings`. All fields optional —
/// only the ones the caller sends get applied. Unsent fields keep
/// their on-disk value, which is the safe default when the UI ships
/// new controls in stages.
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsUpdateDto {
    pub data_dir: Option<String>,
    /// `"en"` | `"el"`. Unknown values are rejected (400) so a stale UI can't
    /// wedge an unsupported locale into config.yaml.
    pub ui_locale: Option<String>,
    /// `"risky"` | `"prop_firm"`. Unknown values are rejected (400).
    pub trading_mode: Option<String>,
    /// `"auto"` | `"cpu"` | `"gpu"`. Unknown values are rejected (400).
    pub compute_mode: Option<String>,
    pub risky_start_balance: Option<f64>,
    pub risky_target_balance: Option<f64>,
    pub risky_horizon_days: Option<u32>,
    pub news_calendar_enabled: Option<bool>,
    pub news_calendar_source: Option<String>,
    /// Snake_case id of a [`NewsTradingMode`] variant.
    pub news_trading_mode: Option<String>,
    // Discovery search knobs (models.prop_search_*) — all optional.
    pub search_population: Option<usize>,
    pub search_generations: Option<usize>,
    pub search_max_hours: Option<f64>,
    pub search_max_indicators: Option<usize>,
    pub search_portfolio_size: Option<usize>,
    pub search_corr_threshold: Option<f64>,
    pub search_max_rows: Option<usize>,
    // GA anti-stagnation knobs (models.discovery_runtime / models.search_runtime).
    pub prefilter_top_k: Option<usize>,
    pub convergence_patience: Option<usize>,
    pub stagnation_patience: Option<usize>,
    pub novelty_weight: Option<f64>,
    pub disable_smc_gate: Option<bool>,
    /// Risk fraction per trade (0..=max_risk_per_trade). Lets the operator set
    /// the sizing risk for the search/run directly (clamped on write).
    pub risk_per_trade: Option<f64>,
    /// Portfolio-level cap on TOTAL concurrent risk across all live engines
    /// (balance fraction; 0 disables). Clamped to [0, 0.5].
    pub max_portfolio_risk: Option<f64>,
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
        Err(err) => {
            let err = anyhow::anyhow!("{err}");
            actionable_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Could not read config.yaml. Make sure the app can read its data folder, \
                 then reload Settings.",
                &err,
            )
        }
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
        let err = anyhow::anyhow!("{err}");
        return actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Settings could not be saved. Close any editor that may have config.yaml open \
             and make sure the folder is writable, then try again.",
            &err,
        );
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
    if let Some(raw) = payload.ui_locale {
        let code = raw.trim().to_ascii_lowercase();
        if code != "en" && code != "el" {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "unknown ui_locale `{}`. Expected one of: en, el.",
                        raw
                    ),
                    "code": "invalid_ui_locale",
                })),
            )
                .into_response();
        }
        settings.system.ui_locale = code;
    }
    if let Some(raw) = payload.trading_mode {
        let mode = raw.trim().to_ascii_lowercase();
        if mode != "risky" && mode != "prop_firm" {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "unknown trading_mode `{}`. Expected one of: risky, prop_firm.",
                        raw
                    ),
                    "code": "invalid_trading_mode",
                })),
            )
                .into_response();
        }
        settings.system.trading_mode = mode;
    }
    if let Some(rpt) = payload.risk_per_trade {
        let cap = if settings.risk.max_risk_per_trade > 0.0 {
            settings.risk.max_risk_per_trade
        } else {
            0.10
        };
        settings.risk.risk_per_trade = rpt.clamp(0.0, cap);
    }
    if let Some(cap) = payload.max_portfolio_risk {
        // 0 disables; hard ceiling 50% — beyond that a "cap" is meaningless.
        settings.risk.max_portfolio_risk = cap.clamp(0.0, 0.5);
    }
    if let Some(raw) = payload.compute_mode {
        let mode = raw.trim().to_ascii_lowercase();
        if mode != "auto" && mode != "cpu" && mode != "gpu" {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "unknown compute_mode `{}`. Expected one of: auto, cpu, gpu.",
                        raw
                    ),
                    "code": "invalid_compute_mode",
                })),
            )
                .into_response();
        }
        settings.system.enable_gpu_preference = mode;
    }
    // Risky-Mode goal (positive values only; the search + projection read these).
    if let Some(v) = payload.risky_start_balance {
        if v > 0.0 {
            settings.system.risky_start_balance_usd = v;
        }
    }
    if let Some(v) = payload.risky_target_balance {
        if v > 0.0 {
            settings.system.risky_target_balance_usd = v;
        }
    }
    if let Some(v) = payload.risky_horizon_days {
        if v > 0 {
            settings.system.risky_horizon_days = v;
        }
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
    // ── Discovery search knobs (clamp to sane floors so a fat-fingered
    // 0 can't wedge the GA) ──────────────────────────────────────────
    if let Some(v) = payload.search_population {
        settings.models.prop_search_population = v.max(10);
    }
    if let Some(v) = payload.search_generations {
        settings.models.prop_search_generations = v.max(1);
    }
    if let Some(v) = payload.search_max_hours {
        // 0 = no time cap; otherwise clamp to a 30-day ceiling.
        settings.models.prop_search_max_hours = v.clamp(0.0, 720.0);
    }
    if let Some(v) = payload.search_max_indicators {
        // 0 = "use all features" (sentinel honoured downstream).
        settings.models.prop_search_max_indicators = v;
    }
    if let Some(v) = payload.search_portfolio_size {
        settings.models.prop_search_portfolio_size = v.max(1);
    }
    if let Some(v) = payload.search_corr_threshold {
        settings.models.prop_search_corr_threshold = v.clamp(0.0, 1.0);
    }
    if let Some(v) = payload.search_max_rows {
        settings.models.prop_search_max_rows = v; // 0 = full dataset
    }
    // ── GA anti-stagnation knobs (un-stick the search from the UI) ──────
    if let Some(v) = payload.prefilter_top_k {
        settings.models.discovery_runtime.prefilter_top_k = v.max(10);
    }
    if let Some(v) = payload.convergence_patience {
        settings.models.search_runtime.convergence_patience = v.max(10);
    }
    if let Some(v) = payload.stagnation_patience {
        settings.models.search_runtime.stagnation_patience = v.max(1);
    }
    if let Some(v) = payload.novelty_weight {
        settings.models.search_runtime.novelty_weight = v.clamp(0.0, 1.0);
    }
    if let Some(v) = payload.disable_smc_gate {
        settings.models.search_runtime.disable_smc_gate = v;
    }

    if let Err(err) = settings.save(config_path()) {
        tracing::error!(
            target: "neoethos_app::server::settings",
            error = %err,
            "failed to write config.yaml from POST /settings"
        );
        let err = anyhow::anyhow!("{err}");
        return actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Settings could not be saved. Close any editor that may have config.yaml open \
             and make sure the folder is writable, then try again.",
            &err,
        );
    }

    tracing::info!(
        target: "neoethos_app::server::settings",
        "config.yaml updated via POST /settings"
    );

    Json(dto_from_settings(&settings)).into_response()
}

fn dto_from_settings(settings: &Settings) -> SettingsDto {
    // `data_dir` lives on `SystemConfig`; the news fields live on
    // `NewsConfig` (verified in `crates/neoethos-core/src/config.rs`).
    // Keep the JSON keys flat so the Flutter side doesn't have to
    // mirror the Rust nesting.
    let mode = settings.news.news_trading_mode;
    SettingsDto {
        data_dir: settings.system.data_dir.display().to_string(),
        ui_locale: settings.system.ui_locale.clone(),
        trading_mode: settings.system.trading_mode.clone(),
        compute_mode: settings.system.enable_gpu_preference.clone(),
        risky_start_balance: settings.system.risky_start_balance_usd,
        risky_target_balance: settings.system.risky_target_balance_usd,
        risky_horizon_days: settings.system.risky_horizon_days,
        news_calendar_enabled: settings.news.news_calendar_enabled,
        news_calendar_source: settings.news.news_calendar_source.clone(),
        news_trading_mode: mode.as_str().to_string(),
        news_trading_mode_display_name: mode.display_name().to_string(),
        search_population: settings.models.prop_search_population,
        search_generations: settings.models.prop_search_generations,
        search_max_hours: settings.models.prop_search_max_hours,
        search_max_indicators: settings.models.prop_search_max_indicators,
        search_portfolio_size: settings.models.prop_search_portfolio_size,
        search_corr_threshold: settings.models.prop_search_corr_threshold,
        search_max_rows: settings.models.prop_search_max_rows,
        prefilter_top_k: settings.models.discovery_runtime.prefilter_top_k,
        convergence_patience: settings.models.search_runtime.convergence_patience,
        stagnation_patience: settings.models.search_runtime.stagnation_patience,
        novelty_weight: settings.models.search_runtime.novelty_weight,
        disable_smc_gate: settings.models.search_runtime.disable_smc_gate,
        max_portfolio_risk: settings.risk.max_portfolio_risk,
    }
}
