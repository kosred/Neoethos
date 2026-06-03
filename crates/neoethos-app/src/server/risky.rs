//! `GET /risky/scenarios` — Risky / Growth Mode time-to-target projection.
//!
//! Computes the SAME numbers the live Risky Mode engine uses
//! ([`neoethos_core::domain::risky_mode::RiskyModeManager::time_to_target_scenarios`])
//! so the Growth Mode card can render the engine's real projection instead
//! of hardcoded illustrative daily-growth rates. This is the "single
//! source of truth" fix the operator asked for: no projection numbers are
//! invented in the UI; they all come from the same Brownian-motion math
//! the autonomous trader is sized against.
//!
//! Pure, read-only math — no live trading, no persistence, no broker. The
//! manager is built with the autonomous-only contract pre-accepted SOLELY
//! so the projection methods are callable; nothing here can place an order.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use neoethos_core::domain::risky_mode::{
    DEFAULT_EXPECTED_REWARD_TO_RISK, DEFAULT_EXPECTED_WIN_RATE, DEFAULT_RISKY_TRADES_PER_DAY,
    DEFAULT_STARTING_CAPITAL_USD, DEFAULT_TARGET_CAPITAL_USD,
    RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION, RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION,
    RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION, RiskyModeConfig, RiskyModeManager, RiskyStage,
};

use super::errors::actionable_error;
use super::state::AppApiState;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RiskyScenarioQuery {
    /// Starting bankroll in USD. Defaults to the engine's canonical
    /// [`DEFAULT_STARTING_CAPITAL_USD`].
    pub starting_usd: Option<f64>,
    /// Target bankroll in USD. Defaults to a sensible multiple of the
    /// start (never invented in the UI).
    pub target_usd: Option<f64>,
    /// Per-trade risk fraction — clamped to the signed Risky band
    /// `[0.30, 0.50]`. The Growth card's three aggression chips map to
    /// the min / default / max of this band.
    pub risk_fraction: Option<f64>,
    /// Expected post-cost win rate `(0,1)`. Defaults to the engine's
    /// honest scalping baseline.
    pub win_rate: Option<f64>,
    /// Expected reward-to-risk per trade. Defaults to the engine value.
    pub reward_to_risk: Option<f64>,
    /// Expected scalping cadence (trades/day) for the days estimate.
    pub trades_per_day: Option<f64>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RiskyScenarioDto {
    // Echo the resolved inputs so the UI renders exactly what was computed.
    pub starting_usd: f64,
    pub target_usd: f64,
    pub risk_fraction: f64,
    pub win_rate: f64,
    pub reward_to_risk: f64,
    pub trades_per_day: f64,
    /// 10th-percentile ("lucky") days to target; `null` when the
    /// configured edge has non-positive expected log-growth.
    pub best_case_days: Option<u32>,
    /// Deterministic ("typical") days to target.
    pub expected_days: Option<u32>,
    /// 75th-percentile ("unlucky but successful") days to target.
    pub conservative_days: Option<u32>,
    /// Brownian-barrier ruin probability from the starting bankroll
    /// (0.0..=1.0). This is the REAL engine estimate — not a heuristic.
    pub ruin_probability: f64,
    /// Band edges so the UI renders the aggression chips / slider from
    /// the engine's canonical bounds rather than hardcoding them.
    pub risk_fraction_min: f64,
    pub risk_fraction_max: f64,
}

/// `GET /risky/scenarios?startingUsd&targetUsd&riskFraction&winRate&rewardToRisk&tradesPerDay`
pub async fn scenarios(
    State(_state): State<AppApiState>,
    Query(q): Query<RiskyScenarioQuery>,
) -> Response {
    let starting = q
        .starting_usd
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_STARTING_CAPITAL_USD);
    let target = q
        .target_usd
        .filter(|v| v.is_finite() && *v > starting)
        .unwrap_or_else(|| DEFAULT_TARGET_CAPITAL_USD.max(starting * 2.0));
    // Aggression is operator-chosen but HARD-clamped to the signed Risky
    // band — the projection can never imply a fraction outside [0.30, 0.50].
    let risk_fraction = q
        .risk_fraction
        .filter(|v| v.is_finite())
        .unwrap_or(RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION)
        .clamp(
            RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION,
            RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION,
        );
    let win_rate = q
        .win_rate
        .filter(|v| v.is_finite() && *v > 0.0 && *v < 1.0)
        .unwrap_or(DEFAULT_EXPECTED_WIN_RATE);
    let reward_to_risk = q
        .reward_to_risk
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_EXPECTED_REWARD_TO_RISK);
    let trades_per_day = q
        .trades_per_day
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_RISKY_TRADES_PER_DAY);

    // Single flat-fraction stage spanning [starting, target] so the
    // projection runs at exactly the chosen aggression. (The live engine
    // tapers the fraction across stages; the card lets the operator pick a
    // single point on the band, which is what this projection reflects.)
    let stages = vec![RiskyStage {
        stage_idx: 0,
        bankroll_lower_usd: starting,
        bankroll_upper_usd: target.max(starting + 1.0),
        risk_per_trade_fraction: risk_fraction,
        max_concurrent_positions: 1,
        max_pair_exposure_fraction: risk_fraction,
        daily_loss_cap_fraction: 0.80,
        weekly_drawdown_cap_fraction: 0.95,
    }];

    // Fill everything else from the engine defaults; only the projection
    // inputs are overridden. `autonomous_only_contract_accepted: true` is
    // required purely to construct the manager for the read-only math.
    let config = RiskyModeConfig {
        starting_capital_usd: starting,
        target_capital_usd: target,
        stages,
        autonomous_only_contract_accepted: true,
        expected_trades_per_day: trades_per_day,
        expected_win_rate: win_rate,
        expected_reward_to_risk: reward_to_risk,
        ..RiskyModeConfig::default()
    };

    let manager = match RiskyModeManager::new(config, starting) {
        Ok(m) => m,
        Err(e) => {
            let err = anyhow::anyhow!("{e}");
            return actionable_error(
                StatusCode::BAD_REQUEST,
                "Could not build the Risky Mode projection from those inputs. \
                 Check the starting/target balances and risk fraction.",
                &err,
            );
        }
    };

    let s = manager.time_to_target_scenarios();
    Json(RiskyScenarioDto {
        starting_usd: starting,
        target_usd: target,
        risk_fraction,
        win_rate,
        reward_to_risk,
        trades_per_day,
        best_case_days: s.best_case_days,
        expected_days: s.expected_days,
        conservative_days: s.conservative_days,
        ruin_probability: s.ruin_probability,
        risk_fraction_min: RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION,
        risk_fraction_max: RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION,
    })
    .into_response()
}
