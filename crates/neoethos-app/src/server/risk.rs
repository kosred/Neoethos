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
//! **The drawdown seeds this endpoint writes are BUFFERED, never the firm's
//! raw published ceiling** (#213/#214, 2026-08-09) — see the block comment in
//! `update_preset`. They also only ever TIGHTEN: a preset switch will not widen
//! a breaker that is already narrower on disk, and refuses loudly when it would
//! have.
//!
//! Backend-side: `crates/neoethos-core/src/domain/prop_firm.rs` owns
//! the preset registry. This module just exposes the active selection
//! + numeric thresholds over HTTP.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_core::domain::prop_firm::{
    PropFirmConstraints, PropFirmPreset, PropFirmRuntimeDefaults,
};

use super::state::AppApiState;

/// Fraction of a firm's PUBLISHED overall-drawdown ceiling this app arms its
/// internal breaker at. **This is a MIRROR of `neoethos_core::config`'s
/// `RiskConfig::default()` (`config.rs:561`, `* 0.7`) — not an independent
/// policy.** The two must move together; if they diverge, a preset click and a
/// fresh `Settings::default()` produce different breakers for the same preset,
/// which is precisely the divergence #213/#214 recorded.
///
/// It lives here as a local constant only because `config.rs` exposes no
/// preset→buffered-risk helper to call. The proper end state is one function in
/// `neoethos-core` that both callers use; that edit is recorded in
/// `docs/pending-edits-forbidden-territory.md` (§7) because `config.rs` was
/// owned by another workflow when this landed.
const TOTAL_DRAWDOWN_BUFFER: f64 = 0.7;

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
    //
    // ── #213 / #214 / #269 (2026-08-09) — THIS WRITER PRODUCED THE TWO WRONG
    // NUMBERS SITTING IN `config.yaml` TODAY. ────────────────────────────────
    // Until this change it wrote the firm's **raw published ceiling** into the
    // breaker thresholds:
    //
    //     settings.risk.daily_drawdown_limit = constraints.max_daily_loss_pct as f64;
    //     settings.risk.total_drawdown_limit = constraints.max_overall_drawdown_pct as f64;
    //
    // while `RiskConfig::default()` seeds those same two fields from the
    // **buffered** values (`config.rs:558` and `:561`). The `f32`→`f64`
    // fingerprints `0.10000000149011612` / `0.20000000298023224` in
    // `config.yaml:158-159` are that write: an `f32` preset constant widened on
    // assignment. On a £1,000 own-money account the total breaker armed at £200
    // lost instead of £140 and the daily at £100 instead of £80 — i.e. at the
    // moment the rule is already broken rather than short of it. And one click
    // on any preset in the UI put it back every time, which is why correcting
    // the YAML alone would have been cosmetic.
    //
    // The two fields use **different mechanisms** and both are mirrored here on
    // purpose — collapsing them into one formula yields a THIRD set of numbers,
    // which is how the divergence survived review:
    //   * daily → `PropFirmRuntimeDefaults::daily_dd_stop_trading_pct`, a
    //     separately published stop-trading threshold (FTMO 0.040 against the
    //     firm's 0.05 ceiling; The5%ers 0.032 against 0.04; own money 0.08
    //     against 0.10). It is NOT `max_daily_loss_pct * 0.7`.
    //   * total → `max_overall_drawdown_pct * TOTAL_DRAWDOWN_BUFFER`, the same
    //     0.7 `config.rs:561` applies.
    let constraints = PropFirmConstraints::for_preset(preset);
    let runtime = PropFirmRuntimeDefaults::for_preset(preset);
    let seeded_daily = runtime.daily_dd_stop_trading_pct;
    let seeded_total = (constraints.max_overall_drawdown_pct as f64) * TOTAL_DRAWDOWN_BUFFER;

    // NEVER SILENTLY RAISE A LIMIT. A preset switch may TIGHTEN a breaker; it
    // may not widen one that is already in force. Without this, moving from
    // The5%ers (daily 0.032) to "none" (0.08) more than doubles the daily loss
    // the bot will trade through, from a UI click whose label says nothing about
    // drawdown. The tighter of {what is on disk, what the preset seeds} wins,
    // and a refused seed is logged at WARN with both numbers and the way to
    // widen it deliberately.
    let prev_daily = settings.risk.daily_drawdown_limit;
    let prev_total = settings.risk.total_drawdown_limit;
    let daily_in_force = prev_daily.is_finite() && prev_daily > 0.0;
    let total_in_force = prev_total.is_finite() && prev_total > 0.0;
    let next_daily = if daily_in_force {
        prev_daily.min(seeded_daily)
    } else {
        seeded_daily
    };
    let next_total = if total_in_force {
        prev_total.min(seeded_total)
    } else {
        seeded_total
    };

    tracing::info!(
        target: "neoethos_app::server::risk",
        preset = %preset.as_str(),
        // Cast to f64: `tracing::field::Value` is implemented for f64, not f32.
        firm_raw_daily = constraints.max_daily_loss_pct as f64,
        firm_raw_total = constraints.max_overall_drawdown_pct as f64,
        buffered_daily = seeded_daily,
        buffered_total = seeded_total,
        previous_daily = prev_daily,
        previous_total = prev_total,
        applied_daily = next_daily,
        applied_total = next_total,
        "preset drawdown seeds: internal breakers arm SHORT of the firm's published \
         ceilings (daily = the preset's stop-trading threshold, total = 70% of the \
         published overall cap). The bot stops trading before the firm would fail the \
         account, not at the same instant."
    );
    if daily_in_force && seeded_daily > prev_daily {
        tracing::warn!(
            target: "neoethos_app::server::risk",
            preset = %preset.as_str(),
            preset_would_set = seeded_daily,
            kept = prev_daily,
            "REFUSED to widen `risk.daily_drawdown_limit` on a preset switch — the \
             value already on disk is tighter and stays. To widen it deliberately, \
             edit `risk.daily_drawdown_limit` in config.yaml (Settings → Advanced → \
             raw YAML) and save."
        );
    }
    if total_in_force && seeded_total > prev_total {
        tracing::warn!(
            target: "neoethos_app::server::risk",
            preset = %preset.as_str(),
            preset_would_set = seeded_total,
            kept = prev_total,
            "REFUSED to widen `risk.total_drawdown_limit` on a preset switch — the \
             value already on disk is tighter and stays. To widen it deliberately, \
             edit `risk.total_drawdown_limit` in config.yaml (Settings → Advanced → \
             raw YAML) and save."
        );
    }

    settings.risk.daily_drawdown_limit = next_daily;
    settings.risk.total_drawdown_limit = next_total;
    // #294 — `monthly_profit_target_pct` is WrittenNeverRead: this line assigns
    // it and the only readers (`domain::risk::PropFirmRules` /
    // `RiskManager::monthly_profit_target_pct`) build from `PropFirmConstraints`
    // directly, never from `Settings`. DECISION TAKEN 2026-08-09: **keep the
    // write, do not wire a reader here.** Reasons, in order:
    //   1. The value is a firm-published constraint, not an operator preference,
    //      so seeding it from the preset matches `config.rs:531` exactly — the
    //      loader and this writer agree, which is the property #213 lost.
    //   2. The only candidate reader is `RiskManager` (`domain/risk.rs:332`),
    //      which has **no production constructor at all** (#137) and is under an
    //      explicit operator hold: wiring it changes live position sizing on a
    //      funded account. Giving this field a reader means wiring that, and
    //      that is not a code-tidying decision.
    // Consequence to be honest about: setting a monthly profit target from the
    // UI still changes nothing at runtime, and a preset switch still overwrites
    // whatever was there. The ledger entry in
    // `crates/neoethos-core/tests/config_has_recipient.rs:200-207` remains
    // accurate (`Inert::WrittenNeverRead`, OWNER: operator) and needs no edit.
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
    //
    // 2026-08-09 (W3): routed through the shared
    // `kill_switch_cooldown_remaining_secs()` helper, which is the SAME
    // function `live_trading::run` consults before every Risky-Mode entry. The
    // Risk screen's "Risky cooldown" row (`desktop/src/screens/Risk.tsx:54`)
    // therefore now describes a clock that a real code path starts and a real
    // code path enforces — before this, it rendered a mechanism nothing could
    // trigger.
    let risky_mode_cooldown_remaining_secs =
        crate::app_services::risky_mode_persistence::kill_switch_cooldown_remaining_secs();

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
