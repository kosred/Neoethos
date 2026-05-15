//! cTrader new / cancel / amend (close) order execution, idempotent retry
//! path, and order error handling.
//!
//! Carved out of `trading/mod.rs` (Batch 5 follow-up). This module owns:
//! - The public UI entry points (`execute_buy_market`, `execute_sell_market`,
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
//! - `execute_buy_market` / `execute_sell_market` / `execute_ctrader_order`
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
    ctrader_protocol_volume_from_units, current_unix_seconds,
    extract_client_order_id_from_request, find_existing_client_order_id,
    format_execution_journal_line, format_execution_outcome_status, next_client_order_seq,
    non_empty_option, prop_firm_pre_trade_check, record_app_event, resolve_symbol,
    synthesize_idempotent_retry_outcome, validate_and_convert_lot_size_to_ctrader_volume,
};
use std::time::Instant;

impl TradingSession {
    pub fn execute_buy_market(&mut self, state: &mut AppState) {
        self.execute_ctrader_order(state, CTraderTradeSide::Buy);
    }

    pub fn execute_sell_market(&mut self, state: &mut AppState) {
        self.execute_ctrader_order(state, CTraderTradeSide::Sell);
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
    ) {
        match self.build_ctrader_order_request(state, side) {
            Ok(order_request) => {
                let account_equity = self.ctrader_account_equity();
                let pip_position = self
                    .ctrader_symbol_pip_position(&state.selected_pair)
                    .unwrap_or(4);
                if let Err(err) = prop_firm_pre_trade_check(
                    &state.risk,
                    &order_request,
                    account_equity,
                    self.initial_equity.unwrap_or(account_equity),
                    self.day_start_equity.unwrap_or(account_equity),
                    pip_position,
                    &state.selected_pair,
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
                if let Some(client_order_id) =
                    extract_client_order_id_from_request(&request)
                {
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
        if let Err(err) = self.refresh_ctrader_runtime_after_execution() {
            let message =
                format!("cTrader execution succeeded but runtime refresh degraded: {err}");
            self.append_trade_journal(message.clone());
            state.status_msg = message.clone();
            record_app_event("ctrader_order_execution_refresh", "DEGRADED", message);
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

        let (limit_price, stop_price) = match order_type {
            CTraderOrderType::Market => (None, None),
            CTraderOrderType::Limit => (Some(state.order_ticket.target_price), None),
            CTraderOrderType::Stop => (None, Some(state.order_ticket.target_price)),
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
    pub(super) fn ctrader_account_equity(&self) -> f64 {
        let runtime = match self.connected_ctrader_runtime() {
            Some(r) => r,
            None => return 0.0,
        };
        let balance = runtime.trader.balance;
        let unrealized = runtime.trader.unrealized_pnl;
        if !runtime.reconcile.positions.is_empty() && unrealized == 0.0 {
            tracing::warn!(
                target: "forex_app::risk",
                positions = runtime.reconcile.positions.len(),
                "ctrader equity computed without unrealized PnL; daily-DD check is balance-only \
                 until the streaming subsystem populates trader.unrealized_pnl"
            );
        }
        balance + unrealized
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
                target: "forex_app::risk",
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

