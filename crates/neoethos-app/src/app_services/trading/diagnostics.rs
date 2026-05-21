//! Formatting, journal/event helpers, and idempotent-retry synthesis.
//!
//! Pure functions extracted from trading.rs that produce human-readable
//! strings for the trade journal, the execution-surface diagnostics list,
//! and the app-section structured log. Also home to the reconcile-before-
//! retry idempotency helpers introduced in audit-fix F3.
//!
//! PRESERVED FIX (do not change without auditor sign-off):
//! - audit-fix F3: `extract_client_order_id_from_request` /
//!   `find_existing_client_order_id` / `synthesize_idempotent_retry_outcome`
//!   together implement the reconcile-before-retry path that prevents
//!   duplicate order submission when the broker rejects a token mid-flight.

use crate::app_record;
use crate::app_services::ctrader_account::{
    CTraderAccountRuntimeSnapshot, CTraderDealSnapshot, CTraderPendingOrderSnapshot,
    CTraderPositionSnapshot,
};
use crate::app_services::ctrader_execution::{
    CTraderExecutionOutcome, CTraderExecutionRequest, CTraderExecutionStatus,
};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    CTraderAmendOrderRequest, CTraderCancelOrderRequest, CTraderClosePositionRequest,
    CTraderNewOrderRequest, CTraderOrderTriggerMethod, CTraderOrderType, CTraderTimeInForce,
    CTraderTradeSide, build_amend_order_request, build_cancel_order_request,
    build_close_position_request, build_new_order_request,
};
use neoethos_core::logging::write_subsystem_record;
use neoethos_core::sectioned_log::SubsystemSection;
use tracing::error;

use super::risk_gate::units_to_ctrader_protocol_volume;

/// SECURITY (audit-fix F3): extract the `client_order_id` from a request
/// shape that may carry one. Only `NewOrder` carries this field — cancel
/// and close-position requests target an existing broker-side id, so they
/// are not at risk of duplicate submission in the same way.
pub(super) fn extract_client_order_id_from_request(
    request: &CTraderExecutionRequest,
) -> Option<String> {
    match request {
        CTraderExecutionRequest::NewOrder(order) => order.client_order_id.clone(),
        CTraderExecutionRequest::CancelOrder(_) | CTraderExecutionRequest::ClosePosition(_) => None,
    }
}

/// Inspect a fresh broker reconcile snapshot for a record that already
/// carries our `client_order_id`. Returns a short description of where it
/// was found (position vs. pending order) so the journal entry is useful.
pub(super) fn find_existing_client_order_id(
    reconcile: &crate::app_services::ctrader_account::CTraderReconcileSnapshot,
    client_order_id: &str,
) -> Option<String> {
    for position in &reconcile.positions {
        if position.client_order_id.as_deref() == Some(client_order_id) {
            return Some(format!("position #{}", position.position_id));
        }
    }
    for order in &reconcile.pending_orders {
        if order.client_order_id.as_deref() == Some(client_order_id) {
            return Some(format!("pending order #{}", order.order_id));
        }
    }
    None
}

/// When the reconcile-before-retry path proves the first attempt was
/// already accepted by the broker, fabricate an "Accepted" outcome that
/// quotes the broker-side record back to the caller. The retry path
/// must NOT reach the websocket again, so we synthesize a minimal
/// outcome from the reconcile snapshot.
pub(super) fn synthesize_idempotent_retry_outcome(
    reconcile: &crate::app_services::ctrader_account::CTraderReconcileSnapshot,
    client_order_id: &str,
) -> CTraderExecutionOutcome {
    let position = reconcile
        .positions
        .iter()
        .find(|p| p.client_order_id.as_deref() == Some(client_order_id));
    let order = reconcile
        .pending_orders
        .iter()
        .find(|o| o.client_order_id.as_deref() == Some(client_order_id));

    let (position_id, order_id, symbol_id, trade_side, lot_size, execution_price, timestamp_ms) =
        if let Some(p) = position {
            (
                Some(p.position_id),
                None,
                Some(p.symbol_id),
                Some(p.trade_side.clone()),
                Some(p.volume),
                p.price,
                p.open_timestamp_ms,
            )
        } else if let Some(o) = order {
            (
                None,
                Some(o.order_id),
                Some(o.symbol_id),
                Some(o.trade_side.clone()),
                Some(o.volume),
                o.limit_price.or(o.stop_price),
                o.open_timestamp_ms,
            )
        } else {
            (None, None, None, None, None, None, None)
        };

    CTraderExecutionOutcome {
        status: CTraderExecutionStatus::Accepted,
        account_id: reconcile.account_id,
        symbol_id,
        order_id,
        position_id,
        deal_id: None,
        trade_side,
        order_type: None,
        lot_size,
        requested_lot_size: lot_size,
        filled_lot_size: lot_size,
        execution_price,
        gross_profit: None,
        fee: None,
        swap: None,
        net_profit: None,
        timestamp_ms,
        error_code: None,
        description: Some(format!(
            "retry skipped: broker already had client_order_id={client_order_id}"
        )),
    }
}

pub(super) fn format_ctrader_terminal_info(
    trader: &crate::app_services::ctrader_account::CTraderTraderSnapshot,
    environment: CTraderEnvironment,
) -> String {
    let broker = trader.broker_name.as_deref().unwrap_or("cTrader Open API");
    format!(
        "{} · {} · account {} · balance {:.2}",
        broker,
        match environment {
            CTraderEnvironment::Live => "Live",
            CTraderEnvironment::Demo => "Demo",
        },
        trader.account_id,
        trader.balance
    )
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn format_ctrader_connect_error(err: &anyhow::Error) -> String {
    let message = err.to_string();
    if message.contains("restored token session") || message.contains("stored token bundle") {
        return "cTrader login required · restore or start auth first".to_string();
    }
    if message.contains("at least one discovered account") {
        return "cTrader account discovery required before connecting".to_string();
    }
    format!("cTrader connect failed: {message}")
}

pub(super) fn format_ctrader_position_line(position: &CTraderPositionSnapshot) -> String {
    let mut line = format!(
        "#{} · symbol {} {} {:.2}",
        position.position_id, position.symbol_id, position.trade_side, position.volume
    );
    if let Some(price) = position.price {
        line.push_str(&format!(" · open {:.5}", price));
    }
    if let Some(stop_loss) = position.stop_loss {
        line.push_str(&format!(" · sl {:.5}", stop_loss));
    }
    if let Some(take_profit) = position.take_profit {
        line.push_str(&format!(" · tp {:.5}", take_profit));
    }
    if let Some(swap) = position.swap {
        line.push_str(&format!(" · swap {:+.2}", swap));
    }
    if let Some(commission) = position.commission {
        line.push_str(&format!(" · fee {:+.2}", commission));
    }
    line.push_str(" · unrealized pnl unavailable");
    line
}

pub(super) fn format_ctrader_pending_order_line(order: &CTraderPendingOrderSnapshot) -> String {
    let mut line = format!(
        "#{} · symbol {} {} {} {:.2}",
        order.order_id, order.symbol_id, order.trade_side, order.order_type, order.volume
    );
    if let Some(limit_price) = order.limit_price {
        line.push_str(&format!(" @ {:.5}", limit_price));
    } else if let Some(stop_price) = order.stop_price {
        line.push_str(&format!(" @ {:.5}", stop_price));
    }
    if let Some(stop_loss) = order.stop_loss {
        line.push_str(&format!(" · sl {:.5}", stop_loss));
    }
    if let Some(take_profit) = order.take_profit {
        line.push_str(&format!(" · tp {:.5}", take_profit));
    }
    line
}

pub(super) fn format_ctrader_deal_line(deal: &CTraderDealSnapshot) -> String {
    let mut line = format!(
        "#{} · {} {} {:.2}",
        deal.deal_id, deal.deal_status, deal.trade_side, deal.filled_volume
    );
    if let Some(execution_price) = deal.execution_price {
        line.push_str(&format!(" @ {:.5}", execution_price));
    }
    if let Some(gross_profit) = deal.gross_profit {
        line.push_str(&format!(" · pnl {:+.2}", gross_profit));
    }
    if let Some(fee) = deal.fee {
        line.push_str(&format!(" · fee {:+.2}", fee));
    }
    if let Some(net_profit) = deal.net_profit {
        line.push_str(&format!(" · net {:+.2}", net_profit));
    }
    line
}

pub(super) fn format_ctrader_history_row(deal: &CTraderDealSnapshot) -> String {
    let mut line = format!(
        "{} · deal #{} · pos #{} · symbol {} {} {:.2}",
        format_timestamp_ms(deal.execution_timestamp_ms),
        deal.deal_id,
        deal.position_id,
        deal.symbol_id,
        deal.trade_side,
        deal.filled_volume
    );
    if let Some(entry_price) = deal.entry_price {
        line.push_str(&format!(" · entry {:.5}", entry_price));
    }
    if let Some(execution_price) = deal.execution_price {
        line.push_str(&format!(" · exit {:.5}", execution_price));
    }
    if let Some(gross_profit) = deal.gross_profit {
        line.push_str(&format!(" · gross {:+.2}", gross_profit));
    } else {
        line.push_str(" · gross n/a");
    }
    if let Some(fee) = deal.fee {
        line.push_str(&format!(" · fee {:+.2}", fee));
    } else {
        line.push_str(" · fee n/a");
    }
    if let Some(net_profit) = deal.net_profit {
        line.push_str(&format!(" · net {:+.2}", net_profit));
    } else {
        line.push_str(" · net n/a");
    }
    line
}

pub(super) fn append_ctrader_order_builder_diagnostics(
    diagnostics: &mut Vec<String>,
    runtime: &CTraderAccountRuntimeSnapshot,
) {
    let account_id = runtime.trader.account_id;
    let symbol_id = runtime
        .reconcile
        .positions
        .first()
        .map(|position| position.symbol_id)
        .or_else(|| {
            runtime
                .reconcile
                .pending_orders
                .first()
                .map(|order| order.symbol_id)
        });

    if let Some(symbol_id) = symbol_id {
        let seed_volume = runtime
            .reconcile
            .positions
            .first()
            .map(|position| position.volume)
            .or_else(|| {
                runtime
                    .reconcile
                    .pending_orders
                    .first()
                    .map(|order| order.volume)
            })
            .unwrap_or(1.0);
        // audit-fix F5: this is a diagnostics builder; if a volume cannot
        // be safely encoded we log it and skip the preview entry rather
        // than crashing the panel.
        match units_to_ctrader_protocol_volume(seed_volume) {
            Ok(protocol_volume) => {
                let request = build_new_order_request(
                    &CTraderNewOrderRequest {
                        account_id,
                        symbol_id,
                        order_type: CTraderOrderType::Market,
                        trade_side: CTraderTradeSide::Buy,
                        volume: protocol_volume,
                        limit_price: None,
                        stop_price: None,
                        time_in_force: Some(CTraderTimeInForce::ImmediateOrCancel),
                        expiration_timestamp_ms: None,
                        stop_loss: None,
                        take_profit: None,
                        comment: Some("preview".to_string()),
                        base_slippage_price: None,
                        slippage_in_points: Some(10),
                        label: Some("preview".to_string()),
                        position_id: None,
                        client_order_id: Some("preview-new".to_string()),
                        relative_stop_loss: None,
                        relative_take_profit: None,
                        guaranteed_stop_loss: Some(false),
                        trailing_stop_loss: Some(false),
                        stop_trigger_method: Some(CTraderOrderTriggerMethod::Trade),
                    },
                    "preview-new-order",
                );
                diagnostics.push(format!(
                    "New order builder ready: payload {}",
                    request.payload_type
                ));
            }
            Err(err) => {
                diagnostics.push(format!("New order builder skipped: {err}"));
            }
        }
    }

    if let Some(order) = runtime.reconcile.pending_orders.first() {
        let cancel_request = build_cancel_order_request(
            &CTraderCancelOrderRequest {
                account_id,
                order_id: order.order_id,
            },
            "preview-cancel-order",
        );
        // audit-fix F5: same guarded conversion for the amend preview.
        let amend_volume = units_to_ctrader_protocol_volume(order.volume).ok();
        let amend_request = build_amend_order_request(
            &CTraderAmendOrderRequest {
                account_id,
                order_id: order.order_id,
                volume: amend_volume,
                limit_price: order.limit_price,
                stop_price: order.stop_price,
                expiration_timestamp_ms: None,
                stop_loss: order.stop_loss,
                take_profit: order.take_profit,
                slippage_in_points: Some(10),
                relative_stop_loss: None,
                relative_take_profit: None,
                guaranteed_stop_loss: Some(false),
                trailing_stop_loss: Some(false),
                stop_trigger_method: Some(CTraderOrderTriggerMethod::Trade),
            },
            "preview-amend-order",
        );
        diagnostics.push(format!(
            "Pending-order builders ready: cancel payload {} · amend payload {}",
            cancel_request.payload_type, amend_request.payload_type
        ));
    }

    if let Some(position) = runtime.reconcile.positions.first() {
        // audit-fix F5: same guarded conversion for the close preview.
        let close_volume = match units_to_ctrader_protocol_volume(position.volume) {
            Ok(v) => v,
            Err(err) => {
                diagnostics.push(format!("Close-position builder skipped: {err}"));
                return;
            }
        };
        let close_request = build_close_position_request(
            &CTraderClosePositionRequest {
                account_id,
                position_id: position.position_id,
                volume: close_volume,
            },
            "preview-close-position",
        );
        diagnostics.push(format!(
            "Close-position builder ready: payload {}",
            close_request.payload_type
        ));
    }
}

pub(super) fn format_execution_journal_line(
    action: &str,
    outcome: &CTraderExecutionOutcome,
) -> String {
    let timestamp = outcome
        .timestamp_ms
        .map(format_timestamp_ms)
        .unwrap_or_else(|| "event-time-unavailable".to_string());
    let mut line = format!(
        "{} · {} · status {}",
        timestamp,
        action,
        match outcome.status {
            CTraderExecutionStatus::Accepted => "ACCEPTED",
            CTraderExecutionStatus::Filled => "FILLED",
            CTraderExecutionStatus::Replaced => "REPLACED",
            CTraderExecutionStatus::Cancelled => "CANCELLED",
            CTraderExecutionStatus::PartialFill => "PARTIAL_FILL",
            CTraderExecutionStatus::Failed => "FAILED",
        }
    );
    if let Some(symbol_id) = outcome.symbol_id {
        line.push_str(&format!(" · symbol {}", symbol_id));
    }
    if let Some(trade_side) = &outcome.trade_side {
        line.push_str(&format!(" · side {}", trade_side));
    }
    if let Some(lot_size) = outcome.lot_size {
        line.push_str(&format!(" · size {:.2}", lot_size));
    }
    if let Some(order_id) = outcome.order_id {
        line.push_str(&format!(" · order {}", order_id));
    }
    if let Some(position_id) = outcome.position_id {
        line.push_str(&format!(" · position {}", position_id));
    }
    if let Some(execution_price) = outcome.execution_price {
        line.push_str(&format!(" · price {:.5}", execution_price));
    }
    if let Some(gross_profit) = outcome.gross_profit {
        line.push_str(&format!(" · gross {:+.2}", gross_profit));
    }
    if let Some(fee) = outcome.fee {
        line.push_str(&format!(" · fee {:+.2}", fee));
    }
    if let Some(net_profit) = outcome.net_profit {
        line.push_str(&format!(" · net {:+.2}", net_profit));
    }
    if let Some(error_code) = &outcome.error_code {
        line.push_str(&format!(" · error {}", error_code));
    }
    if let Some(description) = &outcome.description {
        line.push_str(&format!(" · {}", description));
    }
    line
}

pub(super) fn format_execution_outcome_status(
    prefix: &str,
    outcome: &CTraderExecutionOutcome,
) -> String {
    let mut line = format!(
        "{} {}",
        prefix,
        match outcome.status {
            CTraderExecutionStatus::Accepted => "accepted",
            CTraderExecutionStatus::Filled => "filled",
            CTraderExecutionStatus::Replaced => "replaced",
            CTraderExecutionStatus::Cancelled => "cancelled",
            CTraderExecutionStatus::PartialFill => "partially filled",
            CTraderExecutionStatus::Failed => "failed",
        }
    );
    if let Some(net_profit) = outcome.net_profit {
        line.push_str(&format!(" · net {:+.2}", net_profit));
    }
    if let Some(error_code) = &outcome.error_code {
        line.push_str(&format!(" · error {}", error_code));
    }
    line
}

pub(super) fn non_empty_option(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

pub(super) fn format_timestamp_ms(timestamp_ms: i64) -> String {
    timestamp_ms.to_string()
}

pub(super) fn record_app_event(operation: &str, status: &str, message: impl Into<String>) {
    if let Err(err) = write_subsystem_record(
        SubsystemSection::App,
        app_record(operation, status, message),
    ) {
        error!("Failed to write APP section log: {}", err);
    }
}
