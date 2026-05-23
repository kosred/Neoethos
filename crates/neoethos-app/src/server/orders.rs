//! POST /orders — submit a Market order to the broker.
//!
//! Money-critical. Server-side defence in depth:
//!   - volume_lots must be > 0 and finite (validated by helper)
//!   - At least one of stopLoss / takeProfit must be present (otherwise
//!     we refuse 400) — operator can override with `risky:true`, but
//!     the Flutter UI deliberately makes that hard to flip.
//!   - Broker enforces min_volume / max_volume / step_volume; we
//!     surface its rejection verbatim.
//!
//! Returns the cTrader ExecutionOutcome verbatim so the UI can show
//! order_id + fill price + side, or the broker's failure reason.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::app_services::broker_api::{
    OrderSide, cancel_order_blocking, close_position_blocking, submit_market_order_blocking,
};
use crate::app_services::ctrader_errors::translate_anyhow;

use super::state::AppApiState;

#[derive(Debug, serde::Deserialize)]
pub struct NewOrderBody {
    pub symbol: String,
    pub side: OrderSide,
    /// In lots (1.0 = standard lot, 0.01 = micro lot). Server converts
    /// to broker volume units via the resolved symbol's lot_size.
    #[serde(rename = "volumeLots")]
    pub volume_lots: f64,
    /// Pip distance from fill price; converted to cTrader's
    /// relative_stop_loss (1e-5 units) by the helper. Absolute prices
    /// are not accepted on Market orders.
    #[serde(rename = "stopLossPips")]
    pub stop_loss_pips: Option<f64>,
    #[serde(rename = "takeProfitPips")]
    pub take_profit_pips: Option<f64>,
    pub comment: Option<String>,
    /// Operator must opt in to send an order with no SL and no TP.
    /// Without this, the server refuses 400 — protects against
    /// fat-finger "what's the worst that can happen" trades.
    #[serde(default)]
    pub risky: bool,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewOrderResponseDto {
    pub status: String,
    pub account_id: i64,
    pub symbol_id: Option<i64>,
    pub order_id: Option<i64>,
    pub position_id: Option<i64>,
    pub deal_id: Option<i64>,
    pub trade_side: Option<String>,
    pub order_type: Option<String>,
    pub message: String,
}

pub async fn place(State(_state): State<AppApiState>, Json(body): Json<NewOrderBody>) -> Response {
    if body.stop_loss_pips.is_none() && body.take_profit_pips.is_none() && !body.risky {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "stopLossPips and takeProfitPips are both missing — \
                          set at least one, or pass risky:true to override",
            })),
        )
            .into_response();
    }

    let symbol = body.symbol.trim().to_uppercase();
    if symbol.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "symbol must be non-empty"})),
        )
            .into_response();
    }

    let side = body.side;
    let volume_lots = body.volume_lots;
    let sl = body.stop_loss_pips;
    let tp = body.take_profit_pips;
    let comment = body.comment;

    let result = tokio::task::spawn_blocking(move || {
        submit_market_order_blocking(&symbol, side, volume_lots, sl, tp, comment)
    })
    .await;

    outcome_to_response(result)
}

// ─── POST /positions/{id}/close ────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
pub struct ClosePositionBody {
    #[serde(rename = "positionId")]
    pub position_id: i64,
    /// Volume to close, in cTrader's centi-lot units. The Flutter UI
    /// passes the position's full volume to close it entirely; partial
    /// closes are also legal.
    pub volume: i64,
}

pub async fn close_position(
    State(_state): State<AppApiState>,
    Json(body): Json<ClosePositionBody>,
) -> Response {
    if body.position_id <= 0 || body.volume <= 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "positionId and volume must both be positive",
            })),
        )
            .into_response();
    }
    let position_id = body.position_id;
    let volume = body.volume;
    let result =
        tokio::task::spawn_blocking(move || close_position_blocking(position_id, volume)).await;
    outcome_to_response(result)
}

// ─── POST /orders/{id}/cancel ──────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
pub struct CancelOrderBody {
    #[serde(rename = "orderId")]
    pub order_id: i64,
}

pub async fn cancel_order(
    State(_state): State<AppApiState>,
    Json(body): Json<CancelOrderBody>,
) -> Response {
    if body.order_id <= 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "orderId must be positive"})),
        )
            .into_response();
    }
    let order_id = body.order_id;
    let result = tokio::task::spawn_blocking(move || cancel_order_blocking(order_id)).await;
    outcome_to_response(result)
}

/// Shared response shaper for place/close/cancel — they all come back
/// as `CTraderExecutionOutcome`.
fn outcome_to_response(
    result: Result<
        anyhow::Result<crate::app_services::ctrader_execution::CTraderExecutionOutcome>,
        tokio::task::JoinError,
    >,
) -> Response {
    match result {
        Ok(Ok(outcome)) => {
            let dto = NewOrderResponseDto {
                status: format!("{:?}", outcome.status),
                account_id: outcome.account_id,
                symbol_id: outcome.symbol_id,
                order_id: outcome.order_id,
                position_id: outcome.position_id,
                deal_id: outcome.deal_id,
                trade_side: outcome.trade_side.clone(),
                order_type: outcome.order_type.clone(),
                message: outcome.description.clone().unwrap_or_else(|| {
                    format!(
                        "{:?}: orderId={:?} positionId={:?}",
                        outcome.status, outcome.order_id, outcome.position_id
                    )
                }),
            };
            Json(dto).into_response()
        }
        Ok(Err(err)) => {
            // Decorate the BAD_GATEWAY with a cTrader-error translation
            // when one can be extracted. The Flutter side renders the
            // structured `translation` payload as a colored banner with
            // an optional Re-authenticate / Open Settings CTA, instead
            // of the raw "errorCode=CH_ACCESS_TOKEN_INVALID" string the
            // operator would otherwise see.
            let mut body = serde_json::json!({"error": err.to_string()});
            if let Some(t) = translate_anyhow(&err) {
                body["translation"] =
                    serde_json::to_value(&t).unwrap_or(serde_json::Value::Null);
            }
            (StatusCode::BAD_GATEWAY, Json(body)).into_response()
        }
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("execution task panicked: {join_err}"),
            })),
        )
            .into_response(),
    }
}
