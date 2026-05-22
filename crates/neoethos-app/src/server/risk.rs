//! `/risk` — current prop-firm-safe risk caps.
//!
//! Pulls from the loaded `Settings` (config.yaml) so the Flutter Risk
//! Settings screen shows the same numbers the Rust trading session
//! would enforce. POST support (operator edits caps from the UI) lands
//! in a follow-up — for now this is read-only.
//!
//! The Phase 1 design returns the canonical caps verbatim. Once a
//! `POST /risk` endpoint exists, the same DTO doubles as the request
//! body shape — keep them aligned.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;

use super::state::AppApiState;

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
}

pub async fn risk(State(_state): State<AppApiState>) -> Response {
    // config.yaml lives at the workspace root by default. The same
    // file the egui side reads — single source of truth.
    let settings = match Settings::from_yaml("config.yaml") {
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

    let r = &settings.risk;
    Json(RiskDto {
        risk_per_trade: r.risk_per_trade,
        min_risk_per_trade: r.min_risk_per_trade,
        max_risk_per_trade: r.max_risk_per_trade,
        daily_drawdown_limit: r.daily_drawdown_limit,
        total_drawdown_limit: r.total_drawdown_limit,
        max_lot_size: r.max_lot_size,
        require_stop_loss: r.require_stop_loss,
    })
    .into_response()
}
