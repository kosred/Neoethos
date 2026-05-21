//! cTrader new / cancel / amend (close) order execution, idempotent retry
//! path, and order error handling.
//!
//! Carved out of `trading/mod.rs` (Batch 5 follow-up). This module owns:
//! - The public UI entry points (`execute_buy_order`, `execute_sell_order`,
//!   `cancel_selected_order`, `close_selected_position`).
//! - The internal pipeline (`execute_ctrader_order`,
//!   `execute_ctrader_request`, `build_ctrader_execution_runtime_request`,
//!   `build_ctrader_order_request`, `resolve_selected_ctrader_symbol`).
//! - The Smart-ATR sizing helper (`calculate_smart_atr_in_points`) and the
//!   pip-position lookup (`ctrader_symbol_pip_position`).
//! - The post-trade refresh hook (`refresh_ctrader_runtime_after_execution`),
//!   journaling (`append_trade_journal`), and the equity reader
//!   (`ctrader_account_equity`).
//!
//! PRESERVED FIXES (do not change without auditor sign-off):
//! - Batch 1 / audit-fix F3 (idempotent retry): `execute_ctrader_request`
//!   detects a broker-side token rejection
//!   (`CTRADER_TOKEN_EXPIRED_SENTINEL`), force-refreshes the OAuth bundle,
//!   then — before re-issuing the request — calls `load_ctrader_account_runtime`
//!   (a `ProtoOAReconcileReq`) and scans the response for our
//!   `client_order_id`. If the broker already has the order, we synthesize a
//!   success outcome instead of risking a duplicate fill. If reconcile
//!   itself fails we surface a hard error so the operator decides — we do
//!   NOT silently retry. See `docs/audits/research/ctrader_api_reference.md`
//!   §2 ("Idempotency strategy") for the protocol-level justification.
//! - The `client_order_id` is composed of side, symbol, unix seconds, and
//!   the process-wide monotonic counter (`next_client_order_seq`) so that
//!   replays of a single logical order keep the same id while distinct
//!   orders within the same wall-clock second still get unique ids. The
//!   atomic-counter rationale lives in `client_order.rs`.
//! - `execute_buy_order` / `execute_sell_order` / `execute_ctrader_order`
//!   run `prop_firm_pre_trade_check` BEFORE submitting the order. The gate
//!   itself (and its `pip_position` clamp + symbol-name hard fail) lives
//!   in `risk_gate.rs`; this module only routes the equity/pip-position
//!   parameters in.
//! - `cancel_selected_order` and `close_selected_position` hard-fail when
//!   the execution account id cannot be parsed, instead of defaulting to
//!   `0` and letting the broker resolve "account 0" to whichever account
//!   it considers default. Same reasoning at both call sites.
//! - `close_selected_position` uses `ctrader_protocol_volume_from_units`
//!   from `risk_gate.rs` (audit-fix F5) so a non-finite volume surfaces
//!   as an error rather than silently saturating to `i64::MAX`.

use super::{
    AppState, CTRADER_TOKEN_EXPIRED_SENTINEL, CTraderAccountRuntimeSnapshot,
    CTraderCancelOrderRequest, CTraderClosePositionRequest, CTraderExecutionOutcome,
    CTraderExecutionRequest, CTraderExecutionRuntimeRequest, CTraderExecutionStatus,
    CTraderNewOrderRequest, CTraderOrderType, CTraderSymbolLookupRequest, CTraderTimeInForce,
    CTraderTradeSide, TradingAdapter, TradingAdapterKind, TradingSession,
    ctrader_protocol_volume_from_units, current_unix_seconds, extract_client_order_id_from_request,
    find_existing_client_order_id, format_execution_journal_line, format_execution_outcome_status,
    next_client_order_seq, non_empty_option, prop_firm_pre_trade_check, record_app_event,
    resolve_symbol, synthesize_idempotent_retry_outcome,
    validate_and_convert_lot_size_to_ctrader_volume,
};
use std::time::Instant;

impl TradingSession {
    /// Submit a BUY order for the currently-selected symbol.
    ///
    /// The order type (Market / Limit / Stop) comes from
    /// `state.order_ticket.order_type` — this method does NOT hardcode
    /// Market. The legacy name `execute_buy_market` was renamed in v0.4 to
    /// remove that misconception (the inner builder
    /// `build_ctrader_order_request` has always read `order_type` and the
    /// corresponding `target_price` correctly).
    ///
    /// For Limit / Stop the `target_price` field of the order ticket is
    /// surfaced as `limit_price` / `stop_price` respectively in the cTrader
    /// `ProtoOANewOrderReq`.
    pub fn execute_buy_order(&mut self, state: &mut AppState) {
        self.execute_ctrader_order(state, CTraderTradeSide::Buy, super::OrderSource::Manual);
    }

    /// Submit a SELL order for the currently-selected symbol. See
    /// [`Self::execute_buy_order`] for the order-type dispatch contract.
    pub fn execute_sell_order(&mut self, state: &mut AppState) {
        self.execute_ctrader_order(state, CTraderTradeSide::Sell, super::OrderSource::Manual);
    }

    pub fn cancel_selected_order(&mut self, state: &mut AppState) {
        let Some(order_id) = state.order_ticket.selected_order_id.or_else(|| {
            self.connected_ctrader_runtime().and_then(|runtime| {
                runtime
                    .reconcile
                    .pending_orders
                    .first()
                    .map(|order| order.order_id)
            })
        }) else {
            let message = "No pending cTrader order is selected for cancellation.".to_string();
            state.status_msg = message.clone();
            self.append_trade_journal(message.clone());
            record_app_event("ctrader_cancel_order", "FAILED", message);
            return;
        };

        // HARD FAIL: silently defaulting account_id to 0 here would target
        // whichever account the broker resolves "0" to. Refuse the request
        // instead so the operator sees a real error.
        let account_id = match self
            .selected_ctrader_execution_account_id()
            .and_then(|id| id.parse::<i64>().ok())
        {
            Some(id) => id,
            None => {
                let message =
                    "cTrader order cancel rejected: no execution account selected/parseable"
                        .to_string();
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_cancel_order", "FAILED", message);
                return;
            }
        };
        // Server-side single-call drill-down (Appendix C item #5):
        // before cancelling, capture the order's full details +
        // child-deal chain via `ProtoOAOrderDetailsReq` so the journal
        // records exactly what the broker thought was the order at
        // cancel time. Replaces an N-call client-side scan of the
        // reconcile snapshot + a separate deal lookup. Best-effort —
        // failure is logged at `warn!` but does not block the cancel.
        match self.fetch_ctrader_order_details(order_id) {
            Ok(snapshot) => {
                tracing::debug!(
                    target: "neoethos_app::ctrader_history",
                    order_id,
                    deals = snapshot.deals.len(),
                    "pre-cancel drill-down: ProtoOAOrderDetailsReq returned"
                );
                self.append_trade_journal(format!(
                    "Order #{order_id} pre-cancel snapshot: type {} · {} · {:.2} · {} child deals",
                    snapshot.order.order_type,
                    snapshot.order.trade_side,
                    snapshot.order.volume,
                    snapshot.deals.len(),
                ));
            }
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::ctrader_history",
                    order_id,
                    error = %err,
                    "pre-cancel drill-down: ProtoOAOrderDetailsReq failed; \
                     continuing without journal enrichment"
                );
            }
        }
        match self.execute_ctrader_request(
            state,
            CTraderExecutionRequest::CancelOrder(CTraderCancelOrderRequest {
                account_id,
                order_id,
            }),
            format!("Cancel order #{order_id}"),
        ) {
            Ok(outcome) => {
                state.status_msg = format_execution_outcome_status("Cancelled order", &outcome);
                state.order_ticket.selected_order_id = Some(order_id);
            }
            Err(err) => {
                let message = format!("cTrader order cancel failed: {err}");
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_cancel_order", "FAILED", message);
            }
        }
    }

    pub fn close_selected_position(&mut self, state: &mut AppState) {
        let Some(position_id) = state.order_ticket.selected_position_id.or_else(|| {
            self.connected_ctrader_runtime().and_then(|runtime| {
                runtime
                    .reconcile
                    .positions
                    .first()
                    .map(|position| position.position_id)
            })
        }) else {
            let message = "No open cTrader position is selected for closing.".to_string();
            state.status_msg = message.clone();
            self.append_trade_journal(message.clone());
            record_app_event("ctrader_close_position", "FAILED", message);
            return;
        };

        let Some(volume) = self
            .connected_ctrader_runtime()
            .and_then(|runtime| {
                runtime
                    .reconcile
                    .positions
                    .iter()
                    .find(|position| position.position_id == position_id)
            })
            .map(|position| position.volume)
        else {
            let message =
                format!("Selected cTrader position #{position_id} is no longer available.");
            state.status_msg = message.clone();
            self.append_trade_journal(message.clone());
            record_app_event("ctrader_close_position", "FAILED", message);
            return;
        };

        // audit-fix F5: surface overflow at the caller rather than letting
        // the silent cast through.
        let protocol_volume = match ctrader_protocol_volume_from_units(volume) {
            Ok(v) => v,
            Err(err) => {
                let message = format!("cTrader close-position rejected: {err}");
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_close_position", "FAILED", message);
                return;
            }
        };
        // HARD FAIL: same reasoning as cancel_order — refusing to send a
        // close-position request without a parseable account id is safer
        // than letting the broker resolve account_id=0.
        let account_id = match self
            .selected_ctrader_execution_account_id()
            .and_then(|id| id.parse::<i64>().ok())
        {
            Some(id) => id,
            None => {
                let message =
                    "cTrader position close rejected: no execution account selected/parseable"
                        .to_string();
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_close_position", "FAILED", message);
                return;
            }
        };
        // Server-side single-call drill-down (Appendix C item #5):
        // before closing, capture the position's full order chain via
        // `ProtoOAOrderListByPositionIdReq` so the journal records what
        // happened over the position's life. Replaces an N-call
        // client-side scan of `ProtoOAOrderListReq` filtered by
        // `position_id`. Best-effort — failure is logged at `warn!`
        // but does not block the close.
        match self.fetch_ctrader_orders_for_position(position_id, None, None) {
            Ok(orders) => {
                tracing::debug!(
                    target: "neoethos_app::ctrader_history",
                    position_id,
                    orders = orders.len(),
                    "pre-close drill-down: ProtoOAOrderListByPositionIdReq returned"
                );
                self.append_trade_journal(format!(
                    "Position #{position_id} pre-close snapshot: {} orders linked over the \
                     position's life (single ProtoOAOrderListByPositionIdReq, no client-side filter)",
                    orders.len()
                ));
            }
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::ctrader_history",
                    position_id,
                    error = %err,
                    "pre-close drill-down: ProtoOAOrderListByPositionIdReq failed; \
                     continuing without journal enrichment"
                );
            }
        }
        match self.execute_ctrader_request(
            state,
            CTraderExecutionRequest::ClosePosition(CTraderClosePositionRequest {
                account_id,
                position_id,
                volume: protocol_volume,
            }),
            format!("Close position #{position_id}"),
        ) {
            Ok(outcome) => {
                state.status_msg = format_execution_outcome_status("Closed position", &outcome);
                state.order_ticket.selected_position_id = Some(position_id);
            }
            Err(err) => {
                let message = format!("cTrader position close failed: {err}");
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_close_position", "FAILED", message);
            }
        }
    }

    pub(super) fn execute_ctrader_order(
        &mut self,
        state: &mut AppState,
        side: CTraderTradeSide,
        source: super::OrderSource,
    ) {
        // T-Manual HALT (kill-switch §10.4 / Tier T4) sits ABOVE the
        // per-trade and per-day gates. Reject every new order while
        // the halt is in force; the operator must clear the sentinel
        // via the chrome banner before trading resumes.
        if self.is_halted() {
            let message =
                "T-Manual HALT in force: new orders are blocked until the operator clears the \
                 sentinel flag from the chrome banner."
                    .to_string();
            state.status_msg = message.clone();
            self.append_trade_journal(message.clone());
            record_app_event("trading_halt", "ORDER_BLOCKED", message);
            return;
        }
        // Autonomous-only contract gate (research §7.1 / operator
        // directive 2026-05-17). When Risky Mode is armed AND the
        // operator signed `autonomous_only_contract_accepted` in the
        // wizard, the manager OWNS every order — manual BUY/SELL
        // clicks are rejected at the source. AI signals from the
        // auto-trade dispatcher are allowed through this gate and
        // proceed to the rest of the Risky Mode tier hierarchy.
        // This sits ABOVE `check_trade_allowed` so the rejection
        // doesn't depend on having already built the order request
        // (which would itself burn cycles + might bubble up real
        // errors that confuse the journal).
        if matches!(source, super::OrderSource::Manual)
            && let Some(rm) = self.risky_mode_manager()
            && rm.rejects_manual_orders()
        {
            let tier = neoethos_core::KillSwitchTier::ManualOrderWhileAutonomousOnly;
            let message = format!(
                "Risky Mode kill switch tripped ({tier:?}): manual BUY/SELL is forbidden \
                 while the autonomous-only contract is armed (research §7.1). \
                 Disarm Risky Mode in the Autonomy wizard step or wait for the AI \
                 auto-trade dispatcher."
            );
            tracing::warn!(
                target: "neoethos_app::risky_mode",
                tier = ?tier,
                "Risky Mode rejected manual order via autonomous-only contract"
            );
            state.status_msg = message.clone();
            self.append_trade_journal(message.clone());
            record_app_event("risky_mode_gate", "MANUAL_BLOCKED", message);
            return;
        }
        match self.build_ctrader_order_request(state, side) {
            Ok(order_request) => {
                // Risky Mode gate (research §4–§5 + operator directive
                // §7.1). When Risky Mode is armed it sits IMMEDIATELY
                // above the prop-firm gate — both run, Risky Mode is
                // the strictly tighter outer layer (research §11.3,
                // operator directive 2026-05-15). The gate consults the
                // sticky manual / hardware halts, the per-day cap, the
                // per-stage retreat trigger, the per-month cap, and the
                // pre-send sanity ceiling (`presend_sanity_ceiling_fraction`
                // of bankroll, default 55 %). We pass the converted-
                // protocol-volume (in standard-lot cents) divided by
                // 100 as the notional-USD-at-risk proxy. f64 throughout
                // per operator directive §7.2.
                if let Some(rm) = self.risky_mode_manager() {
                    let size_usd = (order_request.volume as f64) / 100.0;
                    let sl_pips = order_request.relative_stop_loss.unwrap_or(0) as f64;
                    let tp_pips = order_request.relative_take_profit.unwrap_or(0) as f64;
                    if let Err(tier) = rm.check_trade_allowed(size_usd, sl_pips, tp_pips) {
                        let message = format!(
                            "Risky Mode kill switch tripped ({tier:?}): order rejected before prop-firm gate"
                        );
                        tracing::warn!(
                            target: "neoethos_app::risky_mode",
                            tier = ?tier,
                            size_usd,
                            sl_pips,
                            tp_pips,
                            "Risky Mode rejected order"
                        );
                        state.status_msg = message.clone();
                        self.append_trade_journal(message.clone());
                        record_app_event("risky_mode_gate", "BLOCKED", message);
                        return;
                    }
                }
                // Batch 14 authoritative-PnL path: fetch the broker's
                // server-side unrealized PnL for every open position
                // and use that as the equity input to the prop-firm
                // gate. On any failure (network, auth, parse) we fall
                // back to the local mark-to-market via
                // `ctrader_account_equity` and emit a structured
                // `warn!` so an operator can correlate. On a circuit-
                // breaker trip (broker vs local drift > 1 % of
                // notional) we BLOCK the order.
                let (account_equity, breaker) = self.ctrader_account_equity_authoritative();
                if let Some(super::PnLDriftCircuitBreaker::Tripped {
                    position_id,
                    broker_net,
                    local,
                    notional,
                    drift_fraction,
                    threshold_fraction,
                }) = breaker
                {
                    let message = format!(
                        "Prop-firm risk gate blocked: PnL drift circuit breaker tripped for \
                         position #{position_id} (broker_net={broker_net:.4} vs local={local:.4}, \
                         drift={:.4}% > threshold {:.4}% of notional {notional:.2}). \
                         New orders blocked until operator acknowledges via \
                         FOREX_BOT_PNL_CIRCUIT_BREAKER_FRACTION override or a fresh reconcile.",
                        drift_fraction * 100.0,
                        threshold_fraction * 100.0,
                    );
                    state.status_msg = message.clone();
                    self.append_trade_journal(message.clone());
                    record_app_event("prop_firm_risk_gate", "BLOCKED_CIRCUIT_BREAKER", message);
                    return;
                }
                // News-blackout gate — runs BEFORE the prop-firm check
                // because (a) it's the cheapest evaluation (string check,
                // no allocation, no equity math) and (b) a blackout
                // rejection is informationally distinct from a risk-limit
                // breach: the operator needs to know "news event in
                // progress" not "lot too big". Audit gap #2 / roadmap
                // §5.2 news-blackout pre-trade requirement.
                if state.llm_news_filter.is_blackout() {
                    let message = format!(
                        "News-blackout gate blocked: {}",
                        state.llm_news_filter.blackout_reason()
                    );
                    state.status_msg = message.clone();
                    self.append_trade_journal(message.clone());
                    record_app_event("news_blackout_gate", "BLOCKED", message);
                    return;
                }
                let pip_position = self
                    .ctrader_symbol_pip_position(&state.selected_pair)
                    .unwrap_or(4);
                // Note: pass a live mid-market price to the
                // risk gate so Market orders (which carry no limit/stop
                // price) still get a per-trade risk-size check. The gate
                // hard-fails if `stop_loss` is set but neither the order
                // nor this fallback provides an entry estimate — refusing
                // to bypass the risk-per-trade ceiling.
                let market_price_for_entry =
                    self.ctrader_live_mid_price_for_symbol(order_request.symbol_id);
                if let Err(err) = prop_firm_pre_trade_check(
                    &state.risk,
                    &order_request,
                    account_equity,
                    self.initial_equity.unwrap_or(account_equity),
                    self.day_start_equity.unwrap_or(account_equity),
                    pip_position,
                    &state.selected_pair,
                    market_price_for_entry,
                ) {
                    let message = format!("Prop-firm risk gate blocked: {err}");
                    state.status_msg = message.clone();
                    self.append_trade_journal(message.clone());
                    record_app_event("prop_firm_risk_gate", "BLOCKED", message);
                    return;
                }
                match self.execute_ctrader_request(
                    state,
                    CTraderExecutionRequest::NewOrder(Box::new(order_request)),
                    format!("{} {}", side.label(), state.selected_pair),
                ) {
                    Ok(outcome) => {
                        state.status_msg = format_execution_outcome_status(
                            &format!("{} {}", side.label(), state.selected_pair),
                            &outcome,
                        );
                    }
                    Err(err) => {
                        let message = format!("cTrader order failed: {err}");
                        state.status_msg = message.clone();
                        self.append_trade_journal(message.clone());
                        record_app_event("ctrader_order", "FAILED", message);
                    }
                }
            }
            Err(err) => {
                let message = format!("cTrader order ticket invalid: {err}");
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_market_order", "FAILED", message);
            }
        }
    }

    pub(super) fn execute_ctrader_request(
        &mut self,
        state: &mut AppState,
        request: CTraderExecutionRequest,
        operator_action: String,
    ) -> anyhow::Result<CTraderExecutionOutcome> {
        if self.configured_adapter != TradingAdapterKind::CTrader {
            return Err(anyhow::anyhow!(
                "cTrader execution is only available when the cTrader adapter is selected"
            ));
        }
        if !self.connected {
            return Err(anyhow::anyhow!("cTrader runtime is not connected"));
        }

        let runtime_request = self.build_ctrader_execution_runtime_request(request.clone())?;
        let outcome = match self.ctrader_execution_backend.execute(&runtime_request) {
            Ok(outcome) => outcome,
            Err(err) => {
                // D11: cTrader signalled an OAuth-token failure. Force-
                // refresh the bundle (bypassing the time-window check) and
                // retry once. If refresh or retry also fails, surface the
                // original error so the operator sees the broker message.
                if !err.to_string().contains(CTRADER_TOKEN_EXPIRED_SENTINEL) {
                    return Err(err);
                }
                let warn = format!(
                    "cTrader token rejected by broker — forcing OAuth refresh and retrying: {err}"
                );
                self.append_trade_journal(warn.clone());
                state.status_msg = warn.clone();
                record_app_event("ctrader_token_refresh", "FORCED", warn);
                if let Err(refresh_err) = self.force_refresh_ctrader_token_bundle() {
                    return Err(refresh_err.context(err));
                }

                // SECURITY (audit-fix F3): before resubmitting the order
                // under the refreshed token, ask the broker whether this
                // `client_order_id` is already present. The original
                // attempt may have been accepted by the broker before the
                // network connection died — in which case retrying would
                // double the position. If reconcile fails, we do NOT
                // retry: surface the error so the operator can decide.
                if let Some(client_order_id) = extract_client_order_id_from_request(&request) {
                    let reconcile = self.load_ctrader_account_runtime().map_err(|reconcile_err| {
                        anyhow::anyhow!(
                            "cTrader retry aborted: reconcile-before-retry failed and we cannot prove the previous \
                             attempt was not already accepted by the broker (client_order_id={client_order_id}). \
                             Original error: {err}. Reconcile error: {reconcile_err}"
                        )
                    })?;
                    if let Some(existing) =
                        find_existing_client_order_id(&reconcile.reconcile, &client_order_id)
                    {
                        let message = format!(
                            "cTrader retry skipped: broker already has client_order_id={client_order_id} ({existing}); \
                             treating as success to avoid duplicate order"
                        );
                        self.append_trade_journal(message.clone());
                        state.status_msg = message.clone();
                        record_app_event("ctrader_retry_duplicate_skipped", "SUCCESS", message);
                        return Ok(synthesize_idempotent_retry_outcome(
                            &reconcile.reconcile,
                            &client_order_id,
                        ));
                    }
                }

                let retry_request =
                    self.build_ctrader_execution_runtime_request(request.clone())?;
                self.ctrader_execution_backend.execute(&retry_request)?
            }
        };
        let journal_line = format_execution_journal_line(&operator_action, &outcome);
        self.append_trade_journal(journal_line.clone());
        record_app_event(
            "ctrader_order_execution",
            match outcome.status {
                CTraderExecutionStatus::Failed => "FAILED",
                CTraderExecutionStatus::Cancelled => "SUCCESS",
                CTraderExecutionStatus::Accepted
                | CTraderExecutionStatus::Filled
                | CTraderExecutionStatus::Replaced
                | CTraderExecutionStatus::PartialFill => "SUCCESS",
            },
            journal_line,
        );
        // Risky Mode bankroll feed (research §4.2). When the broker
        // returns a realised-PnL figure on a position close, feed it
        // into the Risky Mode manager so stage advancement /
        // retreat tracks the real bankroll. Only fires on close-
        // position outcomes (new-order fills emit `net_profit = 0`
        // before the broker has computed it; the close-position
        // path is the canonical realised-PnL emit-point). Both the
        // broker payload and the Risky Mode bankroll cursor are f64
        // per operator directive §7.2 — the earlier f32 cast was
        // retired with the 2026-05-17 rebaseline.
        if matches!(&request, CTraderExecutionRequest::ClosePosition(_))
            && matches!(
                outcome.status,
                CTraderExecutionStatus::Filled | CTraderExecutionStatus::PartialFill
            )
            && let Some(realized_pnl) = outcome.net_profit
            && let Some(rm) = self.risky_mode_manager_mut()
        {
            if realized_pnl.is_finite() {
                rm.record_trade_outcome(realized_pnl);
                tracing::info!(
                    target: "neoethos_app::risky_mode",
                    realized_pnl_usd = realized_pnl,
                    bankroll_usd = rm.current_bankroll_usd(),
                    stage_idx = rm.current_stage().stage_idx,
                    "Risky Mode bankroll updated from closed position"
                );
            }
        }
        if let Err(err) = self.refresh_ctrader_runtime_after_execution() {
            let message =
                format!("cTrader execution succeeded but runtime refresh degraded: {err}");
            self.append_trade_journal(message.clone());
            state.status_msg = message.clone();
            record_app_event("ctrader_order_execution_refresh", "DEGRADED", message);
        }
        // Server-side per-position drill-down (Appendix C item #5 in
        // `docs/audits/research/ctrader_api_full_reference.md`): when the
        // broker hands back a `position_id`, issue a single
        // `ProtoOAOrderListByPositionIdReq` so the trade journal can
        // record the full order chain attached to this position instead
        // of forcing the operator to mentally cross-reference the
        // reconcile snapshot. Failure is logged but does not affect the
        // success outcome — diagnostics enrichment is best-effort.
        if let Some(position_id) = outcome.position_id {
            match self.fetch_ctrader_orders_for_position(position_id, None, None) {
                Ok(orders) => {
                    tracing::debug!(
                        target: "neoethos_app::ctrader_history",
                        position_id,
                        orders = orders.len(),
                        "post-execution drill-down: ProtoOAOrderListByPositionIdReq returned"
                    );
                    if !orders.is_empty() {
                        self.append_trade_journal(format!(
                            "Position #{position_id} order chain: {} orders linked (single \
                             ProtoOAOrderListByPositionIdReq, no client-side filter)",
                            orders.len()
                        ));
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        target: "neoethos_app::ctrader_history",
                        position_id,
                        error = %err,
                        "post-execution drill-down: ProtoOAOrderListByPositionIdReq failed; \
                         continuing without journal enrichment"
                    );
                }
            }
        }
        self.execution_surface_cache = None;
        self.market_chart_cache = None;
        Ok(outcome)
    }

    pub(super) fn build_ctrader_execution_runtime_request(
        &mut self,
        request: CTraderExecutionRequest,
    ) -> anyhow::Result<CTraderExecutionRuntimeRequest> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader execution requires configured client_id and client_secret"
            ));
        }
        let access_token = self
            .ensure_fresh_ctrader_token_bundle("cTrader execution requires a stored token bundle")?
            .access_token;
        let account_id = self
            .selected_ctrader_execution_account_id()
            .ok_or_else(|| {
                anyhow::anyhow!("cTrader execution requires a selected discovered account")
            })?;
        Ok(CTraderExecutionRuntimeRequest {
            client_id,
            client_secret,
            access_token,
            environment: self.selected_ctrader_environment(),
            account_id,
            request,
        })
    }

    pub(super) fn calculate_smart_atr_in_points(
        &self,
        _state: &AppState,
        symbol_name: &str,
    ) -> Option<i64> {
        let cache_entry = self.market_chart_cache.as_ref()?;
        let chart = &cache_entry.snapshot;
        if chart.candles.len() < 14 {
            return None;
        }
        let candles = &chart.candles[chart.candles.len() - 14..];
        let mut tr_sum = 0.0;
        for i in 1..candles.len() {
            let current = &candles[i];
            let prev = &candles[i - 1];
            let hl = current.high - current.low;
            let hc = (current.high - prev.close).abs();
            let lc = (current.low - prev.close).abs();
            let tr = hl.max(hc).max(lc);
            tr_sum += tr;
        }
        let atr = tr_sum / 13.0; // simple average of the 13 computed TRs

        // Convert ATR price delta into points (pipettes)
        let pip_position = self.ctrader_symbol_pip_position(symbol_name).unwrap_or(4);
        let point_multiplier = 10f64.powi(pip_position + 1);

        let atr_points = atr * point_multiplier;
        Some(atr_points as i64)
    }

    pub(super) fn build_ctrader_order_request(
        &mut self,
        state: &AppState,
        side: CTraderTradeSide,
    ) -> anyhow::Result<CTraderNewOrderRequest> {
        let resolved = self.resolve_selected_ctrader_symbol(&state.selected_pair)?;
        let protocol_volume = validate_and_convert_lot_size_to_ctrader_volume(
            &state.order_ticket,
            state.risk.max_lot_size,
            &resolved.symbol,
        )?;

        let mut relative_stop_loss = None;
        let mut relative_take_profit = None;

        if state.order_ticket.smart_sl_enabled {
            if let Some(atr_points) =
                self.calculate_smart_atr_in_points(state, &state.selected_pair)
            {
                // Calculate based on dynamic volatility
                let sl_mult = 1.5;
                let tp_mult = sl_mult * state.order_ticket.smart_rr_ratio; // standard RR 2.0 -> SL=1.5x, TP=3.0x

                relative_stop_loss = Some((atr_points as f64 * sl_mult) as i64);
                relative_take_profit = Some((atr_points as f64 * tp_mult) as i64);

                tracing::info!(
                    "Smart SL applied: ATR={}pts, SL={:?}, TP={:?} (RR={})",
                    atr_points,
                    relative_stop_loss,
                    relative_take_profit,
                    state.order_ticket.smart_rr_ratio
                );
            } else {
                tracing::warn!(
                    "Smart SL requested but not enough trailing candles for ATR. Sending order without SL/TP bounds or falling back to defaults."
                );
            }
        }

        let order_type = match state.order_ticket.order_type {
            crate::app_state::OrderType::Market => CTraderOrderType::Market,
            crate::app_state::OrderType::Limit => CTraderOrderType::Limit,
            crate::app_state::OrderType::Stop => CTraderOrderType::Stop,
        };

        // Note: defence-in-depth against malformed
        // target_price. The UI clamps to `0.0..=1_000_000.0`, but state can
        // also be mutated programmatically (auto-trade producer, restored
        // wizard config, etc.). Reject explicitly if a Limit/Stop order
        // arrives with target_price <= 0 or non-finite — refusing to send a
        // garbage price to the broker is safer than letting the broker
        // reject it after the WSS round-trip.
        let target_price = state.order_ticket.target_price;
        if matches!(
            order_type,
            CTraderOrderType::Limit | CTraderOrderType::Stop
        ) && (!target_price.is_finite() || target_price <= 0.0)
        {
            anyhow::bail!(
                "Order ticket target_price is not a positive finite price ({target_price}); \
                 refusing to send a Limit/Stop order to the broker. Pick a real price in the \
                 execution panel."
            );
        }

        let (limit_price, stop_price) = match order_type {
            CTraderOrderType::Market => (None, None),
            CTraderOrderType::Limit => (Some(target_price), None),
            CTraderOrderType::Stop => (None, Some(target_price)),
            _ => (None, None),
        };

        Ok(CTraderNewOrderRequest {
            account_id: resolved.account_id,
            symbol_id: resolved.light_symbol.symbol_id,
            order_type,
            trade_side: side,
            volume: protocol_volume,
            limit_price,
            stop_price,
            time_in_force: Some(CTraderTimeInForce::ImmediateOrCancel),
            expiration_timestamp_ms: None,
            stop_loss: None, // We use relative points below
            take_profit: None,
            comment: non_empty_option(&state.order_ticket.comment),
            base_slippage_price: None,
            slippage_in_points: Some(state.order_ticket.slippage_in_points),
            label: non_empty_option(&state.order_ticket.label),
            position_id: None,
            client_order_id: Some(format!(
                "{}-{}-{}-{:x}",
                side.label().to_ascii_lowercase(),
                state.selected_pair.to_ascii_lowercase(),
                // DOCUMENTED-DEFAULT: timestamp is decorative; `next_client_order_seq`
                // is the actual uniqueness guarantee. A clock-before-epoch failure
                // would just yield "0-<seq>" which is still unique.
                current_unix_seconds().unwrap_or_default(),
                next_client_order_seq()
            )),
            relative_stop_loss,
            relative_take_profit,
            guaranteed_stop_loss: None,
            trailing_stop_loss: state.order_ticket.trailing_stop.then_some(true),
            stop_trigger_method: None,
        })
    }

    pub(super) fn resolve_selected_ctrader_symbol(
        &mut self,
        symbol_name: &str,
    ) -> anyhow::Result<crate::app_services::ctrader_data::CTraderResolvedSymbol> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader symbol resolution requires configured client_id and client_secret"
            ));
        }
        let access_token = self
            .ensure_fresh_ctrader_token_bundle(
                "cTrader symbol resolution requires a stored token bundle",
            )?
            .access_token;
        let account_id = self
            .selected_ctrader_execution_account_id()
            .ok_or_else(|| {
                anyhow::anyhow!("cTrader symbol resolution requires a selected discovered account")
            })?;
        resolve_symbol(&CTraderSymbolLookupRequest {
            client_id,
            client_secret,
            access_token,
            environment: self.selected_ctrader_environment(),
            account_id,
            symbol_name: symbol_name.to_string(),
        })
    }

    /// Live account equity = balance + sum of mark-to-market unrealized PnL.
    ///
    /// Critical for prop-firm rules: every published challenge measures
    /// drawdown by EQUITY, not balance, so an open losing position MUST be
    /// counted before the gate fires. `unrealized_pnl` is fed by the
    /// streaming subsystem (set to 0.0 until that wire is in); when 0.0
    /// while positions are open we surface a one-shot warning so the
    /// operator notices the missing live update.
    ///
    /// This is the LOCAL fallback. The Batch 14 prop-firm path calls
    /// [`Self::ctrader_account_equity_authoritative`] first and only
    /// drops here on a broker-side failure.
    pub(super) fn ctrader_account_equity(&self) -> f64 {
        let runtime = match self.connected_ctrader_runtime() {
            Some(r) => r,
            None => return 0.0,
        };
        let balance = runtime.trader.balance;
        let unrealized = runtime.trader.unrealized_pnl;
        if !runtime.reconcile.positions.is_empty() && unrealized == 0.0 {
            tracing::warn!(
                target: "neoethos_app::risk",
                positions = runtime.reconcile.positions.len(),
                "ctrader equity computed without unrealized PnL; daily-DD check is balance-only \
                 until the streaming subsystem populates trader.unrealized_pnl"
            );
        }
        balance + unrealized
    }

    /// Batch 14 authoritative equity reader.
    ///
    /// Issues `ProtoOAGetPositionUnrealizedPnLReq` (payload type 2187)
    /// against the live cTrader session and folds the broker's
    /// `netUnrealizedPnL` per position into the equity figure that the
    /// prop-firm gate consumes. Returns `(equity, circuit_breaker)`
    /// where:
    ///
    /// - `equity = trader.balance + Σ broker_net` on success.
    /// - `equity = self.ctrader_account_equity()` (local fallback) on
    ///   any failure path. The fallback case logs a
    ///   `warn!(target = "neoethos_app::risk")` line with the account id,
    ///   the open-position count, and the error reason — per the
    ///   operator's no-silent-fallback directive (2026-05-15), the
    ///   operator can decide from that line whether to keep trading.
    /// - `circuit_breaker` is `Some(state)` when the broker call
    ///   succeeded — caller MUST inspect for `Tripped { .. }` and
    ///   refuse to size new orders. `None` when the fallback path
    ///   was used (we cannot evaluate drift without a broker value).
    pub(super) fn ctrader_account_equity_authoritative(
        &mut self,
    ) -> (f64, Option<super::PnLDriftCircuitBreaker>) {
        // Compute the local equity once up-front so it is available as
        // the fallback denominator and also as the input to the
        // circuit-breaker comparison. Cheap: pure-balance + streaming
        // PnL sum, no network.
        let local_equity = self.ctrader_account_equity();

        // Pull every value we need out of the immutably-borrowed
        // runtime snapshot inside a dedicated scope so the borrow is
        // released before we ask `self` for the mutable token-refresh
        // path below. Without this scope the NLL analysis CAN extend
        // the immutable borrow past the `&mut self` call.
        let (account_id, open_position_ids, positions_snapshot, balance, position_count) = {
            let Some(runtime) = self.connected_ctrader_runtime() else {
                // Not connected — there is no broker to consult. Local
                // path returns 0.0 in this branch; the prop-firm gate
                // upstream interprets equity==0 as "no information" and
                // will already block on its own (day_start_equity > 0
                // check). No warn-line because not-connected is already
                // a higher-priority error surfaced elsewhere.
                return (local_equity, None);
            };
            let open_position_ids: Vec<i64> = runtime
                .reconcile
                .positions
                .iter()
                .map(|p| p.position_id)
                .collect();
            let position_count = open_position_ids.len();
            (
                runtime.trader.account_id,
                open_position_ids,
                runtime.reconcile.positions.clone(),
                runtime.trader.balance,
                position_count,
            )
        };

        // No open positions: equity == balance regardless of which
        // side we ask. Skip the network round-trip — saves latency on
        // the most common path and avoids the fallback warn-line
        // firing on a healthy session.
        if open_position_ids.is_empty() {
            return (balance, Some(super::PnLDriftCircuitBreaker::Ok));
        }

        // Gather auth + transport without mutating the trade journal
        // on the success path. Mirrors `build_ctrader_execution_runtime_request`
        // but does not need the slow `execute()` plumbing.
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            tracing::warn!(
                target: "neoethos_app::risk",
                account_id,
                position_count,
                "falling back to local unrealized PnL: cTrader client_id/client_secret not configured"
            );
            return (local_equity, None);
        }

        let access_token = match self.ensure_fresh_ctrader_token_bundle(
            "authoritative PnL fetch requires a stored token bundle",
        ) {
            Ok(bundle) => bundle.access_token,
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::risk",
                    account_id,
                    position_count,
                    error = %err,
                    "falling back to local unrealized PnL: token bundle unavailable"
                );
                return (local_equity, None);
            }
        };

        let environment = self.selected_ctrader_environment();
        let transport =
            crate::app_services::ctrader_messages::ProductionCTraderOpenApiTransport::new(
                environment.endpoint_host(),
            );
        let authoritative = match super::fetch_unrealized_pnl_for_all_positions(
            &transport,
            &client_id,
            &client_secret,
            &access_token,
            account_id,
            &open_position_ids,
        ) {
            Ok(snapshot) => snapshot,
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::risk",
                    account_id,
                    position_count,
                    error = %err,
                    "falling back to local unrealized PnL: ProtoOAGetPositionUnrealizedPnLReq failed"
                );
                return (local_equity, None);
            }
        };

        // Authoritative equity: balance + sum of broker net PnL. The
        // circuit breaker compares broker_net to local-per-position;
        // we hand it the same `local_pnl_for_position` closure the
        // audit path uses so the two stay consistent. Local PnL per
        // position is `unrealized_pnl / position_count` only as a
        // last-resort proxy — the real per-position value lives in
        // the streaming subsystem (Batch 7 wired
        // `trader.unrealized_pnl` as a single account-wide figure;
        // per-position breakdown is on the bot's roadmap). We keep
        // the breaker conservative: if per-position local PnL is not
        // available, the breaker is `Ok` (no drift signal possible).
        let breaker = super::evaluate_pnl_drift_circuit_breaker(
            &authoritative,
            &positions_snapshot,
            |_position| {
                // Per-position local PnL is not directly tracked yet
                // (see comment above). Returning `None` causes the
                // breaker to skip the comparison for that position
                // and emit a `debug!` line. We deliberately do NOT
                // synthesize a per-position estimate here — operator
                // directive: silent fallback masks payload problems.
                None
            },
        );

        let equity = balance + authoritative.total_net();
        (equity, Some(breaker))
    }

    /// Pip position (decimal places of one pip) for a forex symbol.
    ///
    /// The bot is FX-only — JPY pairs use 2 decimal pip notation, every
    /// other major/minor uses 4. We deliberately do NOT branch on metals or
    /// crypto here because the bot doesn't trade them; if an unknown symbol
    /// shape arrives, log a structured warn and default to 4 so operators
    /// can spot the mis-routed instrument instead of silently mispricing it.
    pub(super) fn ctrader_symbol_pip_position(&self, symbol: &str) -> Option<i32> {
        let normalized = symbol.to_ascii_uppercase();
        if normalized.contains("JPY") {
            return Some(2);
        }
        // Heuristic: real FX symbols are exactly 6 alphabetic characters
        // (EURUSD, GBPCHF, ...). Anything else is suspicious in a forex-only
        // bot — log a warn but still return a sane default so we don't crash.
        let looks_like_fx_pair =
            normalized.len() == 6 && normalized.chars().all(|c| c.is_ascii_alphabetic());
        if !looks_like_fx_pair {
            tracing::warn!(
                target: "neoethos_app::risk",
                symbol,
                "symbol does not look like a 6-letter FX pair; defaulting pip_position=4"
            );
        }
        Some(4)
    }

    pub(super) fn refresh_ctrader_runtime_after_execution(&mut self) -> anyhow::Result<()> {
        let runtime = self.load_ctrader_account_runtime()?;
        self.terminal_info = super::format_ctrader_terminal_info(
            &runtime.trader,
            self.selected_ctrader_environment(),
        );
        self.adapter = Some(TradingAdapter::CTrader(runtime));
        self.connected = true;
        self.ctrader_runtime_refreshed_at = Some(Instant::now());
        self.execution_surface_cache = None;
        Ok(())
    }

    pub(super) fn connected_ctrader_runtime(&self) -> Option<&CTraderAccountRuntimeSnapshot> {
        match &self.adapter {
            Some(TradingAdapter::CTrader(runtime)) if self.connected => Some(runtime),
            _ => None,
        }
    }

    pub(super) fn append_trade_journal(&mut self, line: String) {
        self.trade_journal.push(line);
        if self.trade_journal.len() > 16 {
            let overflow = self.trade_journal.len() - 16;
            self.trade_journal.drain(0..overflow);
        }
        self.execution_surface_cache = None;
    }
}
