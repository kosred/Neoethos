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
    OrderSide, amend_position_sltp_blocking, cancel_order_blocking, close_position_blocking,
    fetch_account_runtime_blocking, submit_market_order_blocking, submit_pending_order_blocking,
};
use crate::app_services::ctrader_errors::translate_anyhow;
use crate::app_services::ctrader_messages::CTraderOrderType;

use super::errors::internal_panic;
use super::state::AppApiState;

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
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

    // 2026-06-10: risky:true is a deliberate operator override (a naked
    // position is a valid manual choice), but it is also the single most
    // dangerous order shape — one adverse tick has no bracket to stop it.
    // Leave a loud, money-tagged audit-trail entry whenever one actually
    // goes out so it is never silent in the logs.
    if body.risky && body.stop_loss_pips.is_none() && body.take_profit_pips.is_none() {
        tracing::warn!(
            target: "neoethos_app::orders",
            %symbol,
            volume_lots = body.volume_lots,
            side = ?body.side,
            "placing a NAKED order (risky=true, no stop-loss and no take-profit) — \
             this position has no bracket protection"
        );
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

// ─── POST /orders/pending — place a conditional (limit/stop) order ──────────

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct NewPendingOrderBody {
    pub symbol: String,
    pub side: OrderSide,
    /// "limit" or "stop" (case-insensitive). Limit fills at the trigger or
    /// better; stop fills once price trades through the trigger.
    #[serde(rename = "orderType")]
    pub order_type: String,
    #[serde(rename = "volumeLots")]
    pub volume_lots: f64,
    /// Price at which the resting order becomes active. This is the "criteria"
    /// the user sets — the broker fills the order when the market reaches it.
    #[serde(rename = "triggerPrice")]
    pub trigger_price: f64,
    #[serde(rename = "stopLossPips")]
    pub stop_loss_pips: Option<f64>,
    #[serde(rename = "takeProfitPips")]
    pub take_profit_pips: Option<f64>,
    /// Optional Good-Till-Date expiry (Unix ms). Omitted → Good-Till-Cancel.
    #[serde(rename = "expiryUnixMs")]
    pub expiry_unix_ms: Option<i64>,
    pub comment: Option<String>,
}

pub async fn place_pending(
    State(_state): State<AppApiState>,
    Json(body): Json<NewPendingOrderBody>,
) -> Response {
    let symbol = body.symbol.trim().to_uppercase();
    if symbol.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "symbol must be non-empty"})),
        )
            .into_response();
    }
    let order_type = match body.order_type.trim().to_ascii_lowercase().as_str() {
        "limit" => CTraderOrderType::Limit,
        "stop" => CTraderOrderType::Stop,
        other => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("orderType must be 'limit' or 'stop' (got '{other}')"),
                })),
            )
                .into_response();
        }
    };
    if !(body.trigger_price.is_finite() && body.trigger_price > 0.0) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "triggerPrice must be a finite, positive price"})),
        )
            .into_response();
    }

    let side = body.side;
    let volume_lots = body.volume_lots;
    let trigger = body.trigger_price;
    let sl = body.stop_loss_pips;
    let tp = body.take_profit_pips;
    let expiry = body.expiry_unix_ms;
    let comment = body.comment;

    let result = tokio::task::spawn_blocking(move || {
        submit_pending_order_blocking(
            &symbol,
            side,
            order_type,
            volume_lots,
            trigger,
            sl,
            tp,
            expiry,
            comment,
        )
    })
    .await;

    outcome_to_response(result)
}

// ─── GET /orders/pending — list resting (limit/stop) orders ────────────────

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingOrderDto {
    pub order_id: i64,
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    /// Broker wire volume (base units) + best-effort lots (needs symbol metadata).
    pub volume: f64,
    pub volume_lots: Option<f64>,
    /// Whichever of limit/stop the order carries — the price that triggers it.
    pub trigger_price: Option<f64>,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub open_timestamp_ms: Option<i64>,
    pub comment: Option<String>,
}

pub async fn list_pending(State(state): State<AppApiState>) -> Response {
    let names = state.symbol_catalog_snapshot().await;
    let result = tokio::task::spawn_blocking(fetch_account_runtime_blocking).await;
    let snapshot = match result {
        Ok(Ok(s)) => s,
        Ok(Err(err)) => {
            let raw = err.to_string();
            if let Some(t) = translate_anyhow(&err) {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({"error": t.message, "detail": raw, "translation": t})),
                )
                    .into_response();
            }
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": "Could not reach cTrader to list pending orders. Check Broker Setup.",
                    "detail": raw,
                })),
            )
                .into_response();
        }
        Err(join_err) => return internal_panic("Listing pending orders", join_err),
    };

    let orders: Vec<PendingOrderDto> = snapshot
        .reconcile
        .pending_orders
        .into_iter()
        .map(|o| {
            let symbol = names
                .get(&o.symbol_id)
                .cloned()
                .unwrap_or_else(|| format!("sym#{}", o.symbol_id));
            let volume_lots = neoethos_core::symbol_metadata::resolve(&symbol)
                .filter(|m| m.contract_size.is_finite() && m.contract_size > 0.0)
                .map(|m| o.volume / m.contract_size);
            PendingOrderDto {
                order_id: o.order_id,
                symbol,
                side: o.trade_side,
                order_type: o.order_type,
                volume: o.volume,
                volume_lots,
                trigger_price: o.limit_price.or(o.stop_price),
                limit_price: o.limit_price,
                stop_price: o.stop_price,
                stop_loss: o.stop_loss,
                take_profit: o.take_profit,
                open_timestamp_ms: o.open_timestamp_ms,
                comment: o.comment,
            }
        })
        .collect();

    Json(orders).into_response()
}

// ─── POST /positions/{id}/close ────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
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
#[serde(deny_unknown_fields, rename_all = "camelCase")]
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

// ─── POST /positions/protection (modify an open position's SL/TP) ──────────

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct AmendPositionProtectionBody {
    #[serde(rename = "positionId")]
    pub position_id: i64,
    /// New ABSOLUTE stop-loss price. cTrader's position-amend is price-based
    /// (not pip-relative); omit to leave the existing stop untouched.
    #[serde(rename = "stopLossPrice")]
    pub stop_loss_price: Option<f64>,
    #[serde(rename = "takeProfitPrice")]
    pub take_profit_price: Option<f64>,
    /// Toggle the broker-side trailing-stop flag on the position's SL.
    #[serde(rename = "trailingStopLoss")]
    pub trailing_stop_loss: Option<bool>,
}

/// Modify an open position's stop-loss / take-profit (move to breakeven, trail
/// a winner, widen/tighten). Money-critical: at least one bracket must be
/// supplied and every supplied price must be finite and positive.
pub async fn amend_position_protection(
    State(_state): State<AppApiState>,
    Json(body): Json<AmendPositionProtectionBody>,
) -> Response {
    if body.position_id <= 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "positionId must be positive"})),
        )
            .into_response();
    }
    if body.stop_loss_price.is_none() && body.take_profit_price.is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "supply at least one of stopLossPrice / takeProfitPrice to amend",
            })),
        )
            .into_response();
    }
    for (label, price) in [
        ("stopLossPrice", body.stop_loss_price),
        ("takeProfitPrice", body.take_profit_price),
    ] {
        if let Some(p) = price
            && (!p.is_finite() || p <= 0.0)
        {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("{label} must be a finite, positive price"),
                })),
            )
                .into_response();
        }
    }
    let position_id = body.position_id;
    let sl = body.stop_loss_price;
    let tp = body.take_profit_price;
    let trailing = body.trailing_stop_loss;
    let result = tokio::task::spawn_blocking(move || {
        amend_position_sltp_blocking(position_id, sl, tp, trailing)
    })
    .await;
    outcome_to_response(result)
}

/// Shared response shaper for place/close/cancel/amend — they all come back
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
            let raw = err.to_string();
            if let Some(t) = translate_anyhow(&err) {
                let body = serde_json::json!({
                    "error": t.message,
                    "detail": raw,
                    "translation": t,
                });
                (StatusCode::BAD_GATEWAY, Json(body)).into_response()
            } else {
                let body = serde_json::json!({
                    "error": "Broker request failed — could not reach cTrader. Make sure \
                              you're authenticated (Broker Setup → Re-authenticate) and \
                              connected.",
                    "detail": raw,
                });
                (StatusCode::BAD_GATEWAY, Json(body)).into_response()
            }
        }
        Err(join_err) => internal_panic("Submitting the order", join_err),
    }
}
