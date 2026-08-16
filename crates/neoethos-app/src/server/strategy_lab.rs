//! `/strategy_lab/*` — Promotion Gate status + promote-to-live (F-330).
//!
//! The Strategy Lab pipeline is Discovery → Training → Validation →
//! **Promotion Gate**. Discovery already runs walk-forward validation
//! inside its 16-stage funnel, so the per-strategy metrics in
//! `model_targets.json` are ALREADY validated — the promotion gate
//! reads those directly rather than re-running a separate validation
//! job.
//!
//! Endpoints:
//!   - `GET  /strategy_lab/promotion?symbol=EURUSD&base_tf=M5`
//!       Evaluate the latest portfolio for that symbol/timeframe
//!       against the promotion gate and return the decision +
//!       per-criterion breakdown. Read-only.
//!   - `POST /strategy_lab/promote`  (body: {symbol, baseTf})
//!       If the gate passes, copy the trained artifacts from
//!       `models/<symbol>/<tf>/` to `live_models/<symbol>/<tf>/` so the
//!       auto-trade producer (which prefers `live_models/`) starts
//!       using them. Refuses with 412 when the gate fails.

use std::path::{Path, PathBuf};

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_core::domain::promotion_gate::{
    PromotionDecision, PromotionGateConfig, PromotionMetrics, aggregate_portfolio,
    evaluate_promotion,
};
use serde::Deserialize;

use crate::app_services::discovery::{ModelTargetsFile, model_targets_path_for};

use super::errors::{actionable_error, internal_panic};
use super::state::AppApiState;

/// Root dir for trained models (what Training writes).
const MODELS_DIR: &str = "models";
/// Root dir for promoted models (what live inference prefers).
pub const LIVE_MODELS_DIR: &str = "live_models";

#[derive(Debug, Deserialize)]
pub struct PromotionQuery {
    pub symbol: Option<String>,
    pub base_tf: Option<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromotionResponseDto {
    pub symbol: String,
    pub base_tf: String,
    pub portfolio_size: usize,
    /// Portfolio-aggregate metrics the gate evaluated (None when the
    /// portfolio is empty or no model_targets.json exists yet).
    pub aggregate: Option<PromotionMetrics>,
    pub decision: PromotionDecision,
    /// The thresholds in effect, echoed so the UI can render
    /// "Sharpe 1.4 ≥ 1.0 ✓" without a second round-trip.
    pub config: PromotionGateConfig,
}

// ─── GET /strategy_lab/promotion ───────────────────────────────────────────

pub async fn promotion_status(
    State(_state): State<AppApiState>,
    Query(q): Query<PromotionQuery>,
) -> Response {
    // 2026-06-04 PARITY: empty → resolved from config.yaml inside
    // evaluate_promotion_for (shared SystemConfig resolvers), not a hardcoded
    // "EURUSD"/"M5" that ignored the operator's configured symbol/base.
    let symbol = q.symbol.unwrap_or_default();
    let base_tf = q.base_tf.unwrap_or_default();

    let result =
        tokio::task::spawn_blocking(move || evaluate_promotion_for(&symbol, &base_tf)).await;
    match result {
        Ok(Ok(dto)) => Json(dto).into_response(),
        Ok(Err(err)) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not evaluate the promotion gate. Run Discovery first to produce a \
             portfolio for this symbol/timeframe, then retry.",
            &err,
        ),
        Err(join_err) => internal_panic("Evaluating the promotion gate", join_err),
    }
}

/// Load the gate config the operator actually configured.
///
/// **2026-08-04 fix — the gate ran, but with thresholds nobody could
/// reach.** This function used to be
/// `fn load_gate_config(_settings: &Settings) -> PromotionGateConfig {
/// PromotionGateConfig::default() }`: it took the operator's `Settings`,
/// discarded it, and returned the hardcoded moderate defaults. Both the
/// read-only `GET /strategy_lab/promotion` and the authoritative
/// `POST /strategy_lab/promote` route through here, so no portfolio had
/// ever been judged against an operator-set bar — the endpoint even
/// echoed the hardcoded config back to the UI as "the thresholds in
/// effect", which made the wrong thresholds look confirmed.
///
/// The old doc comment said "a future `ModelsConfig.promotion_gate`
/// field can override them here without touching the endpoint". That
/// field now exists and this is the read. `models.promotion_gate` is
/// `#[serde(default)]`, so a `config.yaml` without the key still yields
/// exactly the previous thresholds — see
/// `a_settings_without_the_key_reproduces_the_previous_thresholds`.
fn load_gate_config(settings: &Settings) -> PromotionGateConfig {
    settings.models.promotion_gate.clone()
}

fn evaluate_promotion_for(symbol: &str, base_tf: &str) -> anyhow::Result<PromotionResponseDto> {
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::Promotion)
        .map_err(anyhow::Error::new)?;

    let config_path = super::state::current_config_path();
    let settings = Settings::from_yaml(&config_path)
        .map_err(|e| anyhow::anyhow!("{} not loadable: {e}", config_path.display()))?;
    let gate_config = load_gate_config(&settings);
    // 2026-06-04 PARITY: an empty symbol/base (request omitted it) resolves to
    // the configured default via the SAME shared SystemConfig resolvers the CLI
    // and the discovery/training endpoints use — never a hardcoded EURUSD/M5.
    let symbol = if symbol.trim().is_empty() {
        settings.system.resolve_symbol()
    } else {
        symbol.trim().to_uppercase()
    };
    let base_tf = if base_tf.trim().is_empty() {
        settings.system.resolve_base_timeframe()
    } else {
        base_tf.trim().to_uppercase()
    };
    let data_root = settings.system.data_dir;
    let path = model_targets_path_for(&data_root, &symbol, &base_tf);

    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => {
            // No portfolio on disk yet — the pipeline hasn't produced
            // one for this symbol/timeframe. Report "not promoted" with
            // an actionable summary rather than erroring.
            return Ok(PromotionResponseDto {
                symbol: symbol.to_string(),
                base_tf: base_tf.to_string(),
                portfolio_size: 0,
                aggregate: None,
                decision: PromotionDecision {
                    promoted: false,
                    criteria: Vec::new(),
                    summary: format!(
                        "No model_targets.json for {symbol} {base_tf} — run \
                         Discovery first (Strategy Lab → Discovery)."
                    ),
                },
                config: gate_config,
            });
        }
    };

    let file: ModelTargetsFile = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("model_targets.json parse error: {e}"))?;

    let metrics: Vec<PromotionMetrics> = file
        .portfolio
        .iter()
        .map(|e| PromotionMetrics {
            sharpe: e.sharpe_ratio,
            win_rate: e.win_rate,
            profit_factor: e.profit_factor,
            max_drawdown_pct: e.max_drawdown_pct,
            trades: e.trades_count,
        })
        .collect();

    let aggregate = aggregate_portfolio(&metrics);
    let decision = match &aggregate {
        Some(agg) => evaluate_promotion(agg, &gate_config),
        None => PromotionDecision {
            promoted: false,
            criteria: Vec::new(),
            summary: "Portfolio is empty — nothing to promote.".to_string(),
        },
    };

    Ok(PromotionResponseDto {
        symbol: symbol.to_string(),
        base_tf: base_tf.to_string(),
        portfolio_size: file.portfolio.len(),
        aggregate,
        decision,
        config: gate_config,
    })
}

// ─── POST /strategy_lab/promote ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct PromoteBody {
    pub symbol: Option<String>,
    pub base_tf: Option<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromoteResponseDto {
    pub promoted: bool,
    pub symbol: String,
    pub base_tf: String,
    /// Where the artifacts were copied to when promoted.
    pub live_models_path: Option<String>,
    pub files_copied: usize,
    pub message: String,
}

pub async fn promote(
    State(_state): State<AppApiState>,
    Json(body): Json<PromoteBody>,
) -> Response {
    // 2026-06-04 PARITY: empty → resolved from config.yaml inside
    // evaluate_promotion_for, matching the discovery/training defaults.
    let symbol = body.symbol.unwrap_or_default();
    let base_tf = body.base_tf.unwrap_or_default();

    let result = tokio::task::spawn_blocking(move || promote_if_gated(&symbol, &base_tf)).await;
    match result {
        Ok(Ok(dto)) if dto.promoted => Json(dto).into_response(),
        Ok(Ok(dto)) => {
            // Gate failed — 412 Precondition Failed is the honest code:
            // the request was well-formed but the resource state
            // (portfolio quality) didn't meet the precondition.
            (StatusCode::PRECONDITION_FAILED, Json(dto)).into_response()
        }
        Ok(Err(err)) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Promotion failed. Make sure Discovery finished and the models folder has valid \
             artifacts for this symbol/timeframe.",
            &err,
        ),
        Err(join_err) => internal_panic("Promoting the strategy", join_err),
    }
}

fn promote_if_gated(symbol: &str, base_tf: &str) -> anyhow::Result<PromoteResponseDto> {
    // Re-evaluate the gate server-side — never trust a client claim
    // that the portfolio passed. This is the authoritative check.
    let status = evaluate_promotion_for(symbol, base_tf)?;
    // 2026-07-19 deep-audit fix: use the RESOLVED symbol/base_tf from the
    // gate evaluation everywhere below. The raw args may be EMPTY (the body
    // omitted them and the gate resolved config defaults internally) — the
    // old code then built `models/""/""` as the copy source, which is the
    // models ROOT: a passing gate would have copied the ENTIRE model store
    // flat into live_models/, a layout live inference cannot read.
    let symbol = status.symbol.clone();
    let base_tf = status.base_tf.clone();
    if !status.decision.promoted {
        return Ok(PromoteResponseDto {
            promoted: false,
            symbol,
            base_tf,
            live_models_path: None,
            files_copied: 0,
            message: format!("Promotion blocked: {}", status.decision.summary),
        });
    }

    // Gate passed — copy the trained artifacts to live_models/.
    let src = PathBuf::from(MODELS_DIR).join(&symbol).join(&base_tf);
    let dst = PathBuf::from(LIVE_MODELS_DIR).join(&symbol).join(&base_tf);
    if !src.exists() {
        return Ok(PromoteResponseDto {
            promoted: false,
            symbol,
            base_tf,
            live_models_path: None,
            files_copied: 0,
            message: format!(
                "Gate passed but no trained artifacts at {} — run Training first.",
                src.display()
            ),
        });
    }

    let files_copied = copy_dir_recursive(&src, &dst)
        .map_err(|e| anyhow::anyhow!("copy {} → {}: {e}", src.display(), dst.display()))?;

    tracing::info!(
        target: "neoethos_app::strategy_lab::promote",
        %symbol, %base_tf,
        files = files_copied,
        dst = %dst.display(),
        "promoted portfolio to live_models"
    );

    let message = format!("Promoted {symbol} {base_tf} to live trading ({files_copied} files).");
    Ok(PromoteResponseDto {
        promoted: true,
        symbol,
        base_tf,
        live_models_path: Some(dst.display().to_string()),
        files_copied,
        message,
    })
}

/// Recursively copy `src` into `dst`, creating `dst` and any parents.
/// Returns the number of files copied. Existing files at the
/// destination are overwritten (a fresh promotion replaces the prior
/// live model).
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<usize> {
    std::fs::create_dir_all(dst)?;
    let mut copied = 0;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copied += copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
            copied += 1;
        }
    }
    Ok(copied)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for the 2026-08-04 "no final recipient" fix. Before it,
    /// `load_gate_config` was
    /// `fn load_gate_config(_settings: &Settings) -> PromotionGateConfig {
    /// PromotionGateConfig::default() }` and this test could not fail no
    /// matter what the operator configured.
    #[test]
    fn load_gate_config_honours_the_operator_threshold() {
        let mut settings = Settings::default();
        settings.models.promotion_gate.min_sharpe = 2.5;
        settings.models.promotion_gate.min_trades = 500;
        settings.models.promotion_gate.max_drawdown_pct = 8.0;

        let cfg = load_gate_config(&settings);
        assert_eq!(cfg.min_sharpe, 2.5, "the gate must read the operator's file");
        assert_eq!(cfg.min_trades, 500);
        assert_eq!(cfg.max_drawdown_pct, 8.0);
    }

    /// A portfolio that clears the shipped default bar must be REJECTED
    /// once the operator tightens it — the decision, not just the echoed
    /// config, has to change. `POST /strategy_lab/promote` re-runs this
    /// same path server-side, so this is the copy-to-`live_models/` guard.
    #[test]
    fn a_tightened_gate_changes_the_promotion_decision() {
        let portfolio = PromotionMetrics {
            sharpe: 1.8,
            win_rate: 0.56,
            profit_factor: 1.6,
            max_drawdown_pct: 12.0,
            trades: 240,
        };

        let shipped = load_gate_config(&Settings::default());
        assert!(
            evaluate_promotion(&portfolio, &shipped).promoted,
            "this portfolio clears the shipped default bar"
        );

        let mut settings = Settings::default();
        settings.models.promotion_gate.min_sharpe = 2.5;
        let decision = evaluate_promotion(&portfolio, &load_gate_config(&settings));
        assert!(
            !decision.promoted,
            "an operator min_sharpe of 2.5 must block a 1.8-Sharpe portfolio: {}",
            decision.summary
        );
    }

    /// A `config.yaml` written before the knob existed must promote
    /// exactly what it promoted yesterday.
    #[test]
    fn a_settings_without_the_key_reproduces_the_previous_thresholds() {
        assert_eq!(
            load_gate_config(&Settings::default()),
            PromotionGateConfig::default(),
            "adding the knob must not move the bar for existing installs"
        );
    }
}
