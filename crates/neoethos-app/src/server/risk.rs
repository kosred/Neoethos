//! `/risk` — prop-firm-safe risk caps + active preset selector.
//!
//! - GET  `/risk`         → current risk caps + active preset.
//! - POST `/risk/preset`  → switch the active prop-firm preset (FTMO,
//!                          MyForexFunds, FundedNext, The5%ers, None).
//!                          Rewrites `config.yaml` so the change
//!                          survives restart. The next-launch
//!                          `RiskConfig::default()` will reseed the
//!                          numeric fields from the new preset, but
//!                          any operator overrides in `config.yaml`
//!                          win — preset values are seeds, not locks.
//!
//! Backend-side: `crates/neoethos-core/src/domain/prop_firm.rs` owns
//! the preset registry. This module just exposes the active selection
//! + numeric thresholds over HTTP.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_core::domain::prop_firm::{PropFirmConstraints, PropFirmPreset};

use super::state::AppApiState;

// F-553/F-576 closure (2026-05-25): the per-file `const CONFIG_PATH`
// was removed in favour of the process-wide install on
// `server::state::current_config_path()` so the operator's CLI
// `--config` flag propagates. Local helper keeps the call-sites
// readable without re-introducing the duplication.
fn config_path() -> std::path::PathBuf {
    super::state::current_config_path()
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RiskDto {
    pub risk_per_trade: f64,
    pub min_risk_per_trade: f64,
    pub max_risk_per_trade: f64,
    pub daily_drawdown_limit: f64,
    pub total_drawdown_limit: f64,
    pub max_lot_size: f64,
    pub require_stop_loss: bool,
    /// Currently active preset (snake_case identifier).
    pub preset: String,
    /// Title-Case display name for the active preset.
    pub preset_display_name: String,
    /// All known presets the UI can offer in a dropdown. Each item
    /// includes the firm's hard ceilings so the UI can show them
    /// inline without a follow-up request.
    pub available_presets: Vec<PresetSummaryDto>,
    /// Whether the prop-firm gate is currently armed. Mirrors
    /// `RiskConfig.prop_firm_rules` — false when preset is `none`.
    pub prop_firm_rules_enabled: bool,
    /// **F-231/F-501/F-630 (2026-05-25)** — Risky Mode kill-switch
    /// cooldown status. `None` = no kill on record OR cooldown
    /// already elapsed (Risky Mode armed or never killed). `Some(n)`
    /// = the operator-approved 24 h auto re-arm has `n` seconds
    /// remaining; UI renders "Risky Mode auto re-arming in 17h 23m"
    /// as a status banner.
    pub risky_mode_cooldown_remaining_secs: Option<u64>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PresetSummaryDto {
    pub id: String,
    pub display_name: String,
    pub max_daily_loss_pct: f32,
    pub max_overall_drawdown_pct: f32,
    pub challenge_profit_target_pct: f32,
    pub min_trading_days: u32,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PresetUpdateDto {
    pub preset: String,
}

pub async fn risk(State(_state): State<AppApiState>) -> Response {
    // config.yaml lives at the workspace root by default and remains
    // the single source of truth for backend risk settings.
    let settings = match Settings::from_yaml(config_path()) {
        Ok(s) => s,
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::server::risk",
                error = %err,
                "failed to load config.yaml for /risk endpoint"
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

/// POST /risk/preset — switch the active prop-firm preset and persist.
///
/// Side-effects:
///   1. Validates the preset name (must be a known variant).
///   2. Loads `config.yaml`, sets `risk.preset` to the new value,
///      writes back. Leaves every other field untouched — the
///      numeric thresholds the user previously customised stay put.
///   3. The next launch's `RiskConfig::default()` will reseed
///      preset-derived fields with the new preset's defaults, but
///      only for users who haven't overridden them in YAML.
///
/// Why we don't auto-reseed all numeric fields on preset switch:
/// the operator may have spent time tuning their per-trade risk for
/// their style. Surprising them by overwriting their tuned values is
/// worse than the alternative (the UI shows the new preset's
/// recommended thresholds inline, operator can opt-in to copy them).
pub async fn update_preset(
    State(_state): State<AppApiState>,
    Json(payload): Json<PresetUpdateDto>,
) -> Response {
    let preset = match PropFirmPreset::parse(&payload.preset) {
        Some(p) => p,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "unknown preset `{}`. Known presets: ftmo, myforexfunds, fundednext, the5ers, none.",
                        payload.preset
                    ),
                    "code": "unknown_preset",
                })),
            )
                .into_response();
        }
    };

    let mut settings = match Settings::from_yaml(config_path()) {
        Ok(s) => s,
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::server::risk",
                error = %err,
                "failed to load config.yaml for POST /risk/preset"
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

    settings.risk.preset = preset;
    // Flip the gate flag in sync with the preset choice. "None"
    // disables the prop-firm gate; every other preset enables it.
    settings.risk.prop_firm_rules = preset != PropFirmPreset::None;
    // Apply the preset's numeric drawdown / profit limits IMMEDIATELY so the UI
    // + risk gate reflect the switch (previously only the name was persisted, so
    // the daily/total-DD caps never moved — FundedNext looked identical to None).
    // Sizing knobs (max_lot_size / risk_per_trade) stay operator-set on purpose.
    let constraints = PropFirmConstraints::for_preset(preset);
    settings.risk.daily_drawdown_limit = constraints.max_daily_loss_pct as f64;
    settings.risk.total_drawdown_limit = constraints.max_overall_drawdown_pct as f64;
    settings.risk.monthly_profit_target_pct = constraints.min_monthly_net_profit_pct as f64;

    if let Err(err) = settings.save(config_path()) {
        tracing::error!(
            target: "neoethos_app::server::risk",
            error = %err,
            "failed to persist preset change to config.yaml"
        );
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("failed to persist preset: {err}"),
                "code": "config_save_failed",
            })),
        )
            .into_response();
    }

    tracing::info!(
        target: "neoethos_app::server::risk",
        preset = %preset.as_str(),
        "prop-firm preset updated via POST /risk/preset"
    );
    Json(dto_from_settings(&settings)).into_response()
}

fn dto_from_settings(settings: &Settings) -> RiskDto {
    let r = &settings.risk;
    let preset = r.preset;
    let available_presets: Vec<PresetSummaryDto> = PropFirmPreset::all()
        .iter()
        .map(|&p| {
            let c = PropFirmConstraints::for_preset(p);
            PresetSummaryDto {
                id: p.as_str().to_string(),
                display_name: p.display_name().to_string(),
                max_daily_loss_pct: c.max_daily_loss_pct,
                max_overall_drawdown_pct: c.max_overall_drawdown_pct,
                challenge_profit_target_pct: c.challenge_profit_target_pct,
                min_trading_days: c.min_trading_days,
            }
        })
        .collect();
    // **F-231/F-501/F-630 (2026-05-25)**: surface Risky Mode kill-
    // switch cooldown to the UI. We load the persisted state and ask
    // it how many seconds remain on the 24h auto re-arm cooldown.
    // Best-effort: any IO / parse error → `None` (no cooldown shown);
    // the UI handles `None` as "no kill on record".
    let risky_mode_cooldown_remaining_secs =
        crate::app_services::risky_mode_persistence::load_risky_mode_state()
            .ok()
            .flatten()
            .and_then(|state| state.cooldown_remaining_secs(neoethos_core::utils::now_unix_ms()));

    RiskDto {
        risk_per_trade: r.risk_per_trade,
        min_risk_per_trade: r.min_risk_per_trade,
        max_risk_per_trade: r.max_risk_per_trade,
        daily_drawdown_limit: r.daily_drawdown_limit,
        total_drawdown_limit: r.total_drawdown_limit,
        max_lot_size: r.max_lot_size,
        require_stop_loss: r.require_stop_loss,
        preset: preset.as_str().to_string(),
        preset_display_name: preset.display_name().to_string(),
        available_presets,
        prop_firm_rules_enabled: r.prop_firm_rules,
        risky_mode_cooldown_remaining_secs,
    }
}
