//! POST /orders — submit a Market order to the broker.
//! POST /orders/pending — place a resting limit/stop order.
//!
//! Money-critical. Server-side defence in depth, applied IDENTICALLY to both
//! order-placing routes as of 2026-08-09 (#191 closed the asymmetry):
//!   - volume_lots must be > 0 and finite (validated by helper)
//!   - `risk.require_stop_loss` ON (the shipped default) ⇒ stopLossPips is
//!     MANDATORY and `risky:true` does not override it. Added 2026-08-09 (W1):
//!     the field was displayed in Settings and in `GET /risk` and enforced
//!     nowhere. See [`require_stop_loss_setting`].
//!   - `risk.require_stop_loss` OFF ⇒ at least one of stopLoss / takeProfit
//!     must be present (otherwise we refuse 400) — operator can override with
//!     `risky:true`. **Pendings had NO such rule at all until 2026-08-09**;
//!     they were the one route on which a bracketless order could be placed
//!     with nothing ever asked, and it filled unattended.
//!   - Broker enforces min_volume / max_volume / step_volume; we
//!     surface its rejection verbatim.
//!   - While the margin-call watchdog holds a halt
//!     (`app_services::margin_call`), BOTH routes are refused at the broker
//!     boundary (`broker_api::prepare_new_order`) because the broker says it
//!     is about to liquidate. Closing, cancelling and amending stay allowed.
//!
//! Deliberately NOT enforced here, by operator decision: order SIZE. There is
//! no `risk.max_lot_size` clamp, no `risk_per_trade` sizing and no daily-entry
//! slot consumption on this path. Manual trading respects the operator; the
//! autopilot (`app_services::live_trading`) is where those caps live.
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

/// `risk.require_stop_loss`, read fresh per order.
///
/// **2026-08-09 (W1, the smaller half).** This field was displayed in two
/// places — the Settings control `knob_catalog.rs:370` ("Require Stop-Loss on
/// every order · When on, the risk gate REJECTS any order without a
/// stop_loss") and the `GET /risk` DTO echo rendered at
/// `desktop/src/screens/Risk.tsx:54` — and enforced in exactly zero. The
/// `config_has_recipient` guard passed it on the strength of `RiskDto` having
/// a same-named field.
///
/// Two honest options existed: enforce it, or delete it from the Settings
/// screen. Enforcing is the smaller change (this function plus one condition in
/// each of the two handlers, ~20 lines) and it keeps a control the operator
/// deliberately set to `true`; deleting would have touched `knob_catalog.rs`,
/// `RiskDto`, `desktop/src/api.ts`, `Risk.tsx`, `RiskConfig`, `RiskConfig::default`
/// and both `config.yaml` files to remove a safety toggle nobody asked to lose.
///
/// This does NOT clamp order size. `volume_lots` still passes through
/// untouched: the operator has ruled that the manual path respects him, and
/// `risk.max_lot_size` is deliberately left as the autopilot's cap only.
///
/// Fails CLOSED on an unreadable config: `RiskConfig::default()` ships
/// `require_stop_loss: true` (`config.rs:481`) and so does the operator's
/// `config.yaml:162`, so "we could not read the setting" resolves to the value
/// both of them carry rather than to the permissive one.
fn require_stop_loss_setting() -> bool {
    match neoethos_core::Settings::from_yaml(&super::state::current_config_path()) {
        Ok(s) => s.risk.require_stop_loss,
        Err(e) => {
            tracing::warn!(
                target: "neoethos_app::orders",
                error = %e,
                "could not read risk.require_stop_loss — assuming ON (the shipped default)"
            );
            true
        }
    }
}

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
    // `risk.require_stop_loss` (W1): when ON, an order MUST carry a stop-loss
    // and `risky: true` does NOT override it — that is precisely what the
    // Settings control says it does. When OFF, the pre-existing rule below is
    // the only one, unchanged.
    if require_stop_loss_setting() && body.stop_loss_pips.is_none() {
        tracing::warn!(
            target: "neoethos_app::orders",
            symbol = %body.symbol,
            volume_lots = body.volume_lots,
            risky_override_attempted = body.risky,
            "order refused — risk.require_stop_loss is ON and stopLossPips is missing"
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "stopLossPips is required because risk.require_stop_loss is ON \
                          in Settings. risky:true does not override it. Set a stop-loss, \
                          or turn 'Require Stop-Loss on every order' off in Settings.",
            })),
        )
            .into_response();
    }
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
        // `None`: the operator's own manual order. Deliberately not bound to
        // any engine's admission decision — the operator ruled that manual
        // trading respects the operator.
        submit_market_order_blocking(&symbol, side, volume_lots, sl, tp, comment, None)
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
    /// Operator must opt in to leave a resting order with no SL and no TP.
    ///
    /// **2026-08-09 (#191).** Until today `NewPendingOrderBody` had no `risky`
    /// field and pendings had **no bracket rule at all** when
    /// `risk.require_stop_loss` was OFF — while the market path
    /// ([`NewOrderBody`]) has refused a bracketless order since 2026-06-10. So
    /// a resting limit order could fill into a naked position without the
    /// operator ever being asked to confirm that is what he wanted, and it
    /// filled minutes or hours after he set it, when he was not watching.
    ///
    /// Same semantics and same field name as the market path: `risky: true`
    /// permits a bracketless order, and it does NOT override
    /// `risk.require_stop_loss` when that setting is ON.
    ///
    /// Respects the operator's #190 ruling — this is a bracket ACKNOWLEDGEMENT
    /// only. No size clamp, no `risk_per_trade` sizing and no daily-slot
    /// consumption were added to this path.
    #[serde(default)]
    pub risky: bool,
}

pub async fn place_pending(
    State(_state): State<AppApiState>,
    Json(body): Json<NewPendingOrderBody>,
) -> Response {
    // Same authority as `place` (W1). `risky: true` does NOT override
    // `require_stop_loss` on either path.
    if require_stop_loss_setting() && body.stop_loss_pips.is_none() {
        tracing::warn!(
            target: "neoethos_app::orders",
            symbol = %body.symbol,
            volume_lots = body.volume_lots,
            order_type = %body.order_type,
            risky_override_attempted = body.risky,
            "pending order refused — risk.require_stop_loss is ON and stopLossPips is missing"
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "stopLossPips is required because risk.require_stop_loss is ON \
                          in Settings. risky:true does not override it. Set a stop-loss, \
                          or turn 'Require Stop-Loss on every order' off in Settings.",
            })),
        )
            .into_response();
    }
    // #191 (2026-08-09): the bracket rule pendings never had.
    //
    // With `require_stop_loss` OFF, a resting limit/stop order used to be
    // accepted with no SL, no TP and no acknowledgement of any kind — the
    // market path has refused exactly that shape since 2026-06-10. The
    // asymmetry mattered more, not less, on the pending path: the order fills
    // minutes or hours later, unattended, and lands as a naked position.
    //
    // This REFUSES a shape that was previously accepted. The operator can
    // still place it deliberately with `risky: true`, same as on the market
    // path.
    if body.stop_loss_pips.is_none() && body.take_profit_pips.is_none() && !body.risky {
        tracing::warn!(
            target: "neoethos_app::orders",
            symbol = %body.symbol,
            volume_lots = body.volume_lots,
            order_type = %body.order_type,
            "pending order refused — no stopLossPips and no takeProfitPips, and \
             risky:true was not supplied"
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "stopLossPips and takeProfitPips are both missing on a resting \
                          order — set at least one, or pass risky:true to override. A \
                          bracketless pending order fills unattended into a naked position.",
            })),
        )
            .into_response();
    }
    // Mirrors the market path's naked-order audit trail. A resting bracketless
    // order is the same exposure, delayed — never let it be silent in the logs.
    if body.risky && body.stop_loss_pips.is_none() && body.take_profit_pips.is_none() {
        tracing::warn!(
            target: "neoethos_app::orders",
            symbol = %body.symbol,
            volume_lots = body.volume_lots,
            order_type = %body.order_type,
            trigger_price = body.trigger_price,
            side = ?body.side,
            "placing a NAKED PENDING order (risky=true, no stop-loss and no take-profit) \
             — when this fills, the resulting position has no bracket protection"
        );
    }
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
            // Manual path — see the note in `place`.
            None,
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
        tokio::task::spawn_blocking(move || close_position_blocking(position_id, volume, None))
            .await;
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
        // Deliberately the UNBOUND variant (#199). This is the operator's own
        // manual amend and it is not the product of any engine's admission
        // decision, so there is no environment to hold it to — same rule as
        // the `None` passed on the manual `place` path above. Anything that
        // WAS admitted (the autopilot's trailing stop) must call
        // `amend_position_sltp_expecting` with the environment it was
        // admitted against.
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
