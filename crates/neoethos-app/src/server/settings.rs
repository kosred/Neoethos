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
    /// Top-level trading mode (`"risky"` | `"prop_firm"`) — see
    /// `SystemConfig::trading_mode`. Drives discovery + risk orientation.
    pub trading_mode: String,
    /// Raw `models.discovery_mode` — the power-user escape hatch, NOT the
    /// master switch. Only `"strict"` / `"legacy"` do anything here; every
    /// other value defers to `trading_mode`.
    ///
    /// Surfaced 2026-08-09 (#267) so the UI cannot render a mode the engine
    /// does not run. Before this, `GET /settings` returned `trading_mode`
    /// alone and the escape hatch was invisible to every client.
    pub discovery_mode: String,
    /// The mode the SEARCH ENGINE will actually run, resolved with the same
    /// precedence the engine uses (`neoethos_core::resolved_config`, pinned to
    /// `neoethos_search::discovery::resolve_discovery_mode` by the cross-crate
    /// test `display_mode_matches_the_engine_mode`). One of `"risky"` |
    /// `"prop_firm"` | `"strict"`.
    ///
    /// **This is the field a UI should display as "current mode".**
    /// `trading_mode` is what the operator asked for; this is what he gets.
    pub effective_discovery_mode: String,
    /// True when `effective_discovery_mode != trading_mode` — i.e. the escape
    /// hatch is overriding the master switch. A client that shows
    /// `trading_mode` without honouring this flag is lying to the operator.
    pub trading_mode_divergent: bool,
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
    /// Auto-cull retirement → automatic Discovery on the same symbol+TF to
    /// refill the gap (the retired strategy stays blacklisted forever).
    pub auto_rediscover_on_cull: bool,
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
    /// SEARCH-MORE knob: `true` + CUDA card raises the GA population to the
    /// card's fits ceiling (≤16384) at run start, loudly logged. Changes what
    /// is searched, not just how fast.
    pub search_population_auto: bool,
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
    /// LIVE ML gate (models.live_ml_gate): the trained ensemble scales
    /// per-trade risk on live entries (genes keep the direction; ML only
    /// shrinks size / skips on a hard regime+anomaly collapse).
    pub live_ml_gate: bool,
}

/// Partial-update payload for `POST /settings`. All fields optional —
/// only the ones the caller sends get applied. Unsent fields keep
/// their on-disk value, which is the safe default when the UI ships
/// new controls in stages.
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsUpdateDto {
    pub data_dir: Option<String>,
    /// `"risky"` | `"prop_firm"`. Unknown values are rejected (400).
    pub trading_mode: Option<String>,
    /// `"auto"` | `"cpu"` | `"gpu"`. Unknown values are rejected (400).
    pub compute_mode: Option<String>,
    pub risky_start_balance: Option<f64>,
    pub risky_target_balance: Option<f64>,
    pub risky_horizon_days: Option<u32>,
    pub auto_rediscover_on_cull: Option<bool>,
    pub news_calendar_enabled: Option<bool>,
    pub news_calendar_source: Option<String>,
    /// Snake_case id of a [`NewsTradingMode`] variant.
    pub news_trading_mode: Option<String>,
    // Discovery search knobs (models.prop_search_*) — all optional.
    pub search_population: Option<usize>,
    pub search_population_auto: Option<bool>,
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
    /// LIVE ML gate toggle (models.live_ml_gate).
    pub live_ml_gate: Option<bool>,
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
///   2b. **Checks the risk VALUES (#293, 2026-08-09).** Steps 1 and 2
///      validate shape only, so before this the endpoint accepted and
///      persisted `risk_per_trade: 50`, a disabled daily breaker, or a
///      total breaker below the daily one — `Settings::from_yaml`'s
///      `validate_safety_bounds` never ran on this path. Reject 400
///      listing every violation. This refuses saves that used to
///      succeed; that is the point.
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

    // (2) Typed schema check — **and, since 2026-08-10, the FULL loader.**
    //
    // W2-4 / A1. This step used to be shape-only, and `trailing_enabeld:`
    // sailed through it: serde ignored the unknown key, the file was written
    // verbatim, and the endpoint reported success for an edit that would never
    // take effect. That endpoint is the only route to 364 of the 390 knobs, so
    // "saved" meaning "silently discarded" was the single largest lie in the
    // settings surface.
    //
    // `Deserialize for Settings` is now hand-written and IS the loader
    // (`config.rs`, `mod load_seal`). There is no derived impl left to route
    // around, so this one call runs — on this payload, before anything is
    // written — the retired-key prune, the **unknown-key refusal**, the preset
    // re-derivation and the money-path reports. A misspelled key now returns
    // 400 naming the key.
    //
    // No app-side edit was needed to close W2-4: sealing the deserializer in
    // `neoethos-core` closed it here as a consequence. That is the point of
    // sealing it rather than fixing call sites one at a time.
    let candidate: neoethos_core::Settings =
        match serde_yaml_ng::from_str::<neoethos_core::Settings>(&payload.yaml) {
            Ok(s) => s,
            Err(err) => {
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
        };

    // (2b) **VALUE check — #293, added 2026-08-09.**
    //
    // ⚠ CORRECTED 2026-08-10 (A1): the sentence that used to open this block —
    // "Steps 1 and 2 validate SHAPE only; `Settings::from_yaml` additionally
    // runs `validate_safety_bounds`" — is no longer true of step 2, because
    // step 2 is now the loader itself. It is still true of the CONSEQUENCE,
    // and that is why this block stays: `validate_safety_bounds` **logs and
    // does not reject**, by design, because config consumers require a
    // non-fatal load. A save endpoint is the one place where refusing is
    // possible, so the same conditions it screams about are enforced here as a
    // hard 400. Historically, going straight from `serde_yaml_ng::from_str` to
    // the disk write meant a hand-edited `risk_per_trade: 50` (meaning 50%,
    // typed as 5000%) was accepted, written, and became the sizing input on
    // the next run.
    //
    // The thresholds are copied deliberately and named in the message; if
    // `config.rs` moves one, this list must move with it.
    //
    // **This REFUSES saves that were previously accepted.** The five conditions
    // are listed in the response so a refusal is never mysterious. The config
    // this machine runs (daily 0.08, total 0.14, risk_per_trade 0.03 — see
    // `docs/pending-edits-forbidden-territory.md` §7 for where 0.08/0.14 come
    // from) passes all five; this does not lock the operator out of his own
    // editor.
    {
        let r = &candidate.risk;
        let mut violations: Vec<String> = Vec::new();
        if !r.risk_per_trade.is_finite() || r.risk_per_trade < 0.0 || r.risk_per_trade > 1.0 {
            violations.push(format!(
                "risk.risk_per_trade = {} — must be a FRACTION in [0, 1]. 0.005 means 0.5%, \
                 not 0.5. A value above 1.0 sizes every position ~100x too big.",
                r.risk_per_trade
            ));
        }
        if !r.daily_drawdown_limit.is_finite()
            || r.daily_drawdown_limit <= 0.0
            || r.daily_drawdown_limit > 0.20
        {
            violations.push(format!(
                "risk.daily_drawdown_limit = {} — must be in (0, 0.20]. This is the daily \
                 loss at which the bot stops trading; 0 or negative disables the brake \
                 entirely and >0.20 exceeds every published prop-firm rule.",
                r.daily_drawdown_limit
            ));
        }
        if !r.total_drawdown_limit.is_finite() || r.total_drawdown_limit <= 0.0 {
            violations.push(format!(
                "risk.total_drawdown_limit = {} — must be greater than 0. This is the \
                 account-level breaker.",
                r.total_drawdown_limit
            ));
        }
        if r.total_drawdown_limit.is_finite()
            && r.daily_drawdown_limit.is_finite()
            && r.total_drawdown_limit <= r.daily_drawdown_limit
        {
            violations.push(format!(
                "risk.total_drawdown_limit ({}) must exceed risk.daily_drawdown_limit ({}) — \
                 otherwise the account breaker fires on the first bad day and the daily \
                 breaker can never fire at all.",
                r.total_drawdown_limit, r.daily_drawdown_limit
            ));
        }
        if r.total_drawdown_limit.is_finite() && r.total_drawdown_limit > 0.30 {
            violations.push(format!(
                "risk.total_drawdown_limit = {} — above 0.30 exceeds every published \
                 prop-firm rule; the breaker would arm after the account is already failed.",
                r.total_drawdown_limit
            ));
        }
        if !violations.is_empty() {
            tracing::error!(
                target: "neoethos_app::server::settings",
                violations = ?violations,
                "REFUSED POST /settings/raw — risk values outside safe bounds. Nothing was \
                 written and no backup was taken."
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "Refused: {} risk value(s) are outside safe bounds. Nothing was saved.",
                        violations.len()
                    ),
                    "code": "risk_values_out_of_bounds",
                    "violations": violations,
                    "hint": "Every risk figure is a FRACTION of the account, not a percent: \
                             0.03 = 3%. Fix the listed values and save again.",
                })),
            )
                .into_response();
        }
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

/// Write `contents` to `path` atomically via the M07 primitive
/// (temp + fsync + rename, per-path writer lock, Windows retry).
///
/// 2026-07-19 deep-audit fix: the old local implementation did
/// remove-then-rename on the claim that Windows can't rename over an
/// existing file — which is FALSE for Rust std (MoveFileExW with
/// REPLACE_EXISTING; empirically verified on this machine). Worse, a
/// crash between the remove and the rename left NO config.yaml at all —
/// the exact "app won't open" corruption class M07 was built to close.
fn write_atomic(path: &std::path::Path, contents: &str) -> anyhow::Result<()> {
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent)?;
    neoethos_core::storage::json::write_bytes_atomic(path, contents.as_bytes())
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
        settings.system.trading_mode = mode.clone();

        // ── #267 (§2.4 of docs/audit-status-2026-08-09.md) — RE-VERIFIED AND
        // REFUTED 2026-08-09. DO NOT "fix" this by adding
        // `settings.models.discovery_mode = mode`. ──────────────────────────
        // The audit says this handler writes `system.trading_mode` while
        // "discovery reads `models.discovery_mode`", so a Risky click yields a
        // PropFirm run. Read the engine before believing it:
        //   * `neoethos_search::discovery::resolve_discovery_mode`
        //     (`discovery.rs:5772`) takes BOTH strings, and it is
        //     `system.trading_mode` — the field written above — that selects
        //     Risky vs PropFirm.
        //   * `models.discovery_mode` is consulted for ONE purpose: the
        //     `"strict"` / `"legacy"` power-user escape hatch
        //     (`discovery_mode_from_config`, `discovery.rs:5755`). Every other
        //     value, including `"risky"`, falls through to PropFirm there —
        //     which is why assigning `"risky"` to it is at best a no-op.
        //   * The call site is `discovery.rs:843-846`
        //     (`DiscoveryConfig::from_settings`), and the cross-crate test
        //     `display_mode_matches_the_engine_mode`
        //     (`discovery_tests.rs:2523`) pins that precedence against the
        //     report's copy of it.
        // Writing the escape hatch from this handler would SILENTLY DESTROY a
        // `strict`/`legacy` setting on every mode click — a real regression
        // traded for an imaginary one.
        //
        // What IS real, and is what this block now closes: when the escape
        // hatch is armed, the engine runs Strict no matter which mode the
        // operator picks here, and the banner would still read what he clicked.
        // So we resolve the mode through the SAME resolver the report uses
        // (`neoethos_core::resolved_config`, kept in step with the engine by the
        // test above — no precedence logic is re-implemented here) and REFUSE
        // the save when the selection would not take effect. Nothing is written
        // in that case: the operator clears the hatch first, deliberately.
        let effective = neoethos_core::resolved_config::ResolvedConfig::from_settings(&settings)
            .search
            .mode;
        if effective != mode {
            tracing::error!(
                target: "neoethos_app::server::settings",
                requested = %mode,
                effective = %effective,
                discovery_mode = %settings.models.discovery_mode,
                "REFUSED trading_mode change — `models.discovery_mode` escape hatch \
                 overrides it, so the UI would show a mode the engine does not run"
            );
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": format!(
                        "Cannot select `{mode}`: `models.discovery_mode` is set to `{}`, \
                         which forces the search into `{effective}` regardless of the \
                         trading mode. Nothing was saved. Clear `models.discovery_mode` \
                         (set it to `prop_firm`) in Settings → Advanced → raw YAML first, \
                         then pick the mode again.",
                        settings.models.discovery_mode
                    ),
                    "code": "trading_mode_overridden_by_discovery_mode",
                    "requested": mode.clone(),
                    "effective": effective.clone(),
                    "discoveryMode": settings.models.discovery_mode.clone(),
                })),
            )
                .into_response();
        }
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
    if let Some(b) = payload.auto_rediscover_on_cull {
        settings.system.auto_rediscover_on_cull = b;
    }
    if let Some(b) = payload.news_calendar_enabled {
        settings.news.news_calendar_enabled = b;
    }
    if let Some(raw) = payload.news_calendar_source {
        // Only accept a provider the calendar fetcher can actually serve.
        // Previously ANY non-blank string was accepted, persisted and echoed
        // back while `news_calendar::fetch_calendar` fetched ForexFactory
        // regardless — so the operator could "switch provider", see it saved,
        // and receive ForexFactory data forever.
        match neoethos_core::config::validate_news_calendar_source(&raw) {
            Ok(id) => settings.news.news_calendar_source = id,
            Err(message) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": message,
                        "code": "invalid_news_source",
                        "supported": neoethos_core::config::SUPPORTED_NEWS_CALENDAR_SOURCES,
                    })),
                )
                    .into_response();
            }
        }
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
    if let Some(b) = payload.search_population_auto {
        settings.models.prop_search_population_auto = b;
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
    if let Some(v) = payload.live_ml_gate {
        settings.models.live_ml_gate = v;
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
    // #267 — resolve the mode the ENGINE runs through the same resolver the run
    // report uses, rather than echoing the master switch and hoping. See the
    // long note in `update_settings`: `models.discovery_mode` is an escape
    // hatch that can override `system.trading_mode`, and until today no client
    // could see that it was armed.
    let effective_discovery_mode =
        neoethos_core::resolved_config::ResolvedConfig::from_settings(settings)
            .search
            .mode;
    // `"growth"` is an accepted alias of `"risky"` on the engine side
    // (`discovery.rs:5779`), so normalise before comparing — otherwise a
    // perfectly consistent `trading_mode: growth` would be reported as a
    // divergence and the flag would train the reader to ignore it.
    let requested_mode = match settings
        .system
        .trading_mode
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "growth" => "risky".to_string(),
        other => other.to_string(),
    };
    let trading_mode_divergent = effective_discovery_mode != requested_mode;
    if trading_mode_divergent {
        tracing::error!(
            target: "neoethos_app::server::settings",
            trading_mode = %settings.system.trading_mode,
            discovery_mode = %settings.models.discovery_mode,
            effective = %effective_discovery_mode,
            "config.yaml selects one trading mode and the search will run another — \
             `models.discovery_mode` is overriding `system.trading_mode`. The UI is \
             being told both values plus `tradingModeDivergent: true`; every candidate \
             ranked under the effective mode was ranked under rules the operator did \
             not pick from the mode switch."
        );
    }
    SettingsDto {
        data_dir: settings.system.data_dir.display().to_string(),
        trading_mode: settings.system.trading_mode.clone(),
        discovery_mode: settings.models.discovery_mode.clone(),
        effective_discovery_mode,
        trading_mode_divergent,
        compute_mode: settings.system.enable_gpu_preference.clone(),
        risky_start_balance: settings.system.risky_start_balance_usd,
        risky_target_balance: settings.system.risky_target_balance_usd,
        risky_horizon_days: settings.system.risky_horizon_days,
        auto_rediscover_on_cull: settings.system.auto_rediscover_on_cull,
        news_calendar_enabled: settings.news.news_calendar_enabled,
        news_calendar_source: settings.news.news_calendar_source.clone(),
        news_trading_mode: mode.as_str().to_string(),
        news_trading_mode_display_name: mode.display_name().to_string(),
        search_population: settings.models.prop_search_population,
        search_population_auto: settings.models.prop_search_population_auto,
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
        live_ml_gate: settings.models.live_ml_gate,
    }
}
