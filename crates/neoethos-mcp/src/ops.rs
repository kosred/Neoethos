//! One `Backend` method per MCP tool. The MCP layer (`server.rs`) is a thin
//! shim over these, so the tier-1 tests can exercise every behavior —
//! including the demo guard and the wire-unit conversions — directly against
//! a mock HTTP backend without driving the MCP protocol.
//!
//! Conventions:
//! - Backend wire bodies are built here (camelCase / snake_case per route);
//!   MCP-facing names stay snake_case in `params.rs`.
//! - Trade-affecting methods call `ensure_demo()` FIRST, then any reads they
//!   need, then `trade_post()` — one fresh proof per broker-affecting call.
//! - Validation failures return `ToolError` with an actionable message; no
//!   silent clamping beyond what the backend itself documents.

use serde_json::{Value, json};

use crate::backend::{Backend, ToolError};
use crate::params::*;

/// 1.0 lot = 10_000_000 broker wire cents (0.01-unit cents), per the
/// backend's `/broker/margin/expected` contract.
const WIRE_UNITS_PER_LOT: f64 = 10_000_000.0;

fn require_positive(name: &str, v: f64) -> Result<(), ToolError> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(ToolError(format!(
            "{name} must be a finite, positive number (got {v})"
        )))
    }
}

impl Backend {
    // ── System and health ─────────────────────────────────────────────────

    pub async fn op_system_health(&self) -> Result<Value, ToolError> {
        let started = std::time::Instant::now();
        let health = self.get_json("/healthz", &[]).await?;
        Ok(json!({
            "healthz": health,
            "roundTripMs": started.elapsed().as_millis() as u64,
        }))
    }

    pub async fn op_hardware_info(&self) -> Result<Value, ToolError> {
        self.get_json("/hardware", &[]).await
    }

    pub async fn op_engine_status(&self) -> Result<Value, ToolError> {
        self.get_json("/engines/status", &[]).await
    }

    pub async fn op_storage_paths(&self) -> Result<Value, ToolError> {
        self.get_json("/storage/paths", &[]).await
    }

    pub async fn op_diagnostics_report(&self) -> Result<Value, ToolError> {
        self.post_json(
            "/diagnostics/report",
            &json!({
                "user_description": "Generated via the Codex MCP control plane.",
                "category": "MCP diagnostics",
            }),
        )
        .await
    }

    // ── Account and broker state ──────────────────────────────────────────

    /// `/broker/status` + `/broker/accounts` merged — the agent-visible face
    /// of the demo guard. The status half is authoritative; if the accounts
    /// round-trip fails (broker offline), its error is surfaced LOUDLY in
    /// the payload instead of failing the whole tool, so the agent can still
    /// see connected/environment while diagnosing.
    pub async fn op_broker_status(&self) -> Result<Value, ToolError> {
        let status = self.get_json("/broker/status", &[]).await?;
        let accounts = self.get_json("/broker/accounts", &[]).await;
        let (accounts_value, accounts_error) = match accounts {
            Ok(v) => (v, Value::Null),
            Err(e) => (Value::Null, Value::String(e.0)),
        };
        Ok(json!({
            "status": status,
            "accounts": accounts_value,
            "accountsError": accounts_error,
        }))
    }

    pub async fn op_account_snapshot(&self, p: AccountSnapshotParams) -> Result<Value, ToolError> {
        if p.refresh.unwrap_or(false) {
            self.post_json("/account/snapshot/refresh", &json!({}))
                .await
        } else {
            self.get_json("/account/snapshot", &[]).await
        }
    }

    pub async fn op_list_positions(&self) -> Result<Value, ToolError> {
        let snapshot = self.get_json("/account/snapshot", &[]).await?;
        Ok(snapshot.get("positions").cloned().unwrap_or(json!([])))
    }

    pub async fn op_broker_symbols(&self) -> Result<Value, ToolError> {
        self.get_json("/broker/symbols", &[]).await
    }

    pub async fn op_order_history(&self, p: OrderHistoryParams) -> Result<Value, ToolError> {
        let mut query: Vec<(&str, String)> = Vec::new();
        if let Some(from) = p.from_ms {
            query.push(("from", from.to_string()));
        }
        if let Some(to) = p.to_ms {
            query.push(("to", to.to_string()));
        }
        self.get_json("/broker/orders/history", &query).await
    }

    pub async fn op_cashflow(&self) -> Result<Value, ToolError> {
        self.get_json("/broker/cashflow", &[]).await
    }

    pub async fn op_expected_margin(&self, p: ExpectedMarginParams) -> Result<Value, ToolError> {
        require_positive("volume_lots", p.volume_lots)?;
        let wanted = p.symbol.trim().to_uppercase();
        if wanted.is_empty() {
            return Err(ToolError("symbol must be non-empty".to_string()));
        }
        let catalog = self.get_json("/broker/symbols", &[]).await?;
        let symbol_id = catalog
            .get("symbols")
            .and_then(Value::as_array)
            .and_then(|list| {
                list.iter().find(|s| {
                    s.get("symbolName")
                        .and_then(Value::as_str)
                        .is_some_and(|n| n.eq_ignore_ascii_case(&wanted))
                })
            })
            .and_then(|s| s.get("symbolId"))
            .and_then(Value::as_i64)
            .ok_or_else(|| {
                ToolError(format!(
                    "symbol '{wanted}' not found in the broker symbol catalog — run \
                     broker_symbols to see what this account can trade"
                ))
            })?;
        let volume = (p.volume_lots * WIRE_UNITS_PER_LOT).round() as i64;
        self.get_json(
            "/broker/margin/expected",
            &[
                ("symbolId", symbol_id.to_string()),
                ("volume", volume.to_string()),
            ],
        )
        .await
    }

    // ── Market data ───────────────────────────────────────────────────────

    pub async fn op_get_chart(&self, p: GetChartParams) -> Result<Value, ToolError> {
        let mut query = vec![
            ("symbol", p.symbol.trim().to_uppercase()),
            ("timeframe", p.timeframe.trim().to_uppercase()),
        ];
        if let Some(limit) = p.limit {
            query.push(("limit", limit.to_string()));
        }
        self.get_json("/chart", &query).await
    }

    pub async fn op_get_indicator(&self, p: GetIndicatorParams) -> Result<Value, ToolError> {
        let mut query = vec![
            ("symbol", p.symbol.trim().to_uppercase()),
            ("timeframe", p.timeframe.trim().to_uppercase()),
            ("indicator", p.indicator.trim().to_lowercase()),
        ];
        for (name, v) in [
            ("period", p.period),
            ("std_dev", p.std_dev),
            ("fast", p.fast),
            ("slow", p.slow),
            ("signal", p.signal),
            ("k_period", p.k_period),
            ("k_slow", p.k_slow),
            ("d_period", p.d_period),
        ] {
            if let Some(v) = v {
                query.push((name, v.to_string()));
            }
        }
        if let Some(limit) = p.limit {
            query.push(("limit", limit.to_string()));
        }
        self.get_json("/indicators", &query).await
    }

    pub async fn op_live_spots(&self) -> Result<Value, ToolError> {
        self.get_json("/live/spots", &[]).await
    }

    pub async fn op_spread_stats(&self) -> Result<Value, ToolError> {
        self.get_json("/data/spread-stats", &[]).await
    }

    pub async fn op_news_briefing(&self, p: NewsBriefingParams) -> Result<Value, ToolError> {
        let query = if p.force.unwrap_or(false) {
            vec![("force", "true".to_string())]
        } else {
            Vec::new()
        };
        self.get_json("/news/feed", &query).await
    }

    pub async fn op_get_watchlist(&self) -> Result<Value, ToolError> {
        self.get_json("/watchlist", &[]).await
    }

    pub async fn op_set_watchlist(&self, p: SetWatchlistParams) -> Result<Value, ToolError> {
        self.post_json("/watchlist", &json!({ "symbols": p.symbols }))
            .await
    }

    // ── Trading — direct orders (demo-guarded) ────────────────────────────

    pub async fn op_open_position(&self, p: OpenPositionParams) -> Result<Value, ToolError> {
        let symbol = p.symbol.trim().to_uppercase();
        if symbol.is_empty() {
            return Err(ToolError("symbol must be non-empty".to_string()));
        }
        require_positive("volume_lots", p.volume_lots)?;
        require_positive("stop_loss_pips", p.stop_loss_pips)?;
        if let Some(tp) = p.take_profit_pips {
            require_positive("take_profit_pips", tp)?;
        }
        let proof = self.ensure_demo().await?;
        let mut body = json!({
            "symbol": symbol,
            "side": p.side.as_wire(),
            "volumeLots": p.volume_lots,
            "stopLossPips": p.stop_loss_pips,
            // Hardcoded, no parameter to change it: this control plane never
            // sends a naked order.
            "risky": false,
        });
        if let Some(tp) = p.take_profit_pips {
            body["takeProfitPips"] = json!(tp);
        }
        if let Some(comment) = p.comment {
            body["comment"] = json!(comment);
        }
        self.trade_post("/orders", &body, proof).await
    }

    pub async fn op_close_position(&self, p: ClosePositionParams) -> Result<Value, ToolError> {
        if p.position_id <= 0 {
            return Err(ToolError("position_id must be positive".to_string()));
        }
        let proof = self.ensure_demo().await?;

        // Read the position's current volume so the agent talks lots, never
        // wire units (design §2 "additional hard rails").
        let snapshot = self.get_json("/account/snapshot", &[]).await?;
        let positions = snapshot
            .get("positions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let position = positions
            .iter()
            .find(|pos| pos.get("positionId").and_then(Value::as_i64) == Some(p.position_id))
            .ok_or_else(|| {
                ToolError(format!(
                    "position {} not found in the account snapshot — run list_positions to \
                     see the open positions, or account_snapshot with refresh=true if the \
                     snapshot may be stale",
                    p.position_id
                ))
            })?;
        let full_units = position
            .get("volumeUnits")
            .and_then(Value::as_i64)
            .filter(|v| *v > 0)
            .ok_or_else(|| {
                ToolError(format!(
                    "position {} carries no positive volumeUnits in the snapshot — cannot \
                     close safely; refresh the snapshot and retry",
                    p.position_id
                ))
            })?;

        let volume_units = match p.volume_lots {
            None => full_units,
            Some(lots) => {
                require_positive("volume_lots", lots)?;
                let full_lots = position
                    .get("volumeLots")
                    .and_then(Value::as_f64)
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .ok_or_else(|| {
                        ToolError(format!(
                            "position {} has no volumeLots in the snapshot (symbol metadata \
                             missing), so a partial close in lots cannot be converted to \
                             wire units. Omit volume_lots to close the full position.",
                            p.position_id
                        ))
                    })?;
                let units = ((lots / full_lots) * full_units as f64).round() as i64;
                if units <= 0 {
                    return Err(ToolError(format!(
                        "volume_lots {lots} converts to zero wire units for position {} \
                         (open volume {full_lots} lots) — too small to close",
                        p.position_id
                    )));
                }
                if units > full_units {
                    return Err(ToolError(format!(
                        "volume_lots {lots} exceeds the position's open volume of \
                         {full_lots} lots — pass a smaller volume or omit volume_lots for \
                         a full close"
                    )));
                }
                units
            }
        };

        self.trade_post(
            "/positions/close",
            &json!({ "positionId": p.position_id, "volume": volume_units }),
            proof,
        )
        .await
    }

    pub async fn op_modify_position_protection(
        &self,
        p: ModifyPositionProtectionParams,
    ) -> Result<Value, ToolError> {
        if p.position_id <= 0 {
            return Err(ToolError("position_id must be positive".to_string()));
        }
        if p.stop_loss_price.is_none() && p.take_profit_price.is_none() && p.trailing_stop.is_none()
        {
            return Err(ToolError(
                "supply at least one of stop_loss_price / take_profit_price / trailing_stop"
                    .to_string(),
            ));
        }
        for (name, v) in [
            ("stop_loss_price", p.stop_loss_price),
            ("take_profit_price", p.take_profit_price),
        ] {
            if let Some(v) = v {
                require_positive(name, v)?;
            }
        }
        let proof = self.ensure_demo().await?;
        let mut body = json!({ "positionId": p.position_id });
        if let Some(sl) = p.stop_loss_price {
            body["stopLossPrice"] = json!(sl);
        }
        if let Some(tp) = p.take_profit_price {
            body["takeProfitPrice"] = json!(tp);
        }
        if let Some(trailing) = p.trailing_stop {
            body["trailingStopLoss"] = json!(trailing);
        }
        self.trade_post("/positions/protection", &body, proof).await
    }

    pub async fn op_place_pending_order(
        &self,
        p: PlacePendingOrderParams,
    ) -> Result<Value, ToolError> {
        let symbol = p.symbol.trim().to_uppercase();
        if symbol.is_empty() {
            return Err(ToolError("symbol must be non-empty".to_string()));
        }
        require_positive("volume_lots", p.volume_lots)?;
        require_positive("trigger_price", p.trigger_price)?;
        require_positive("stop_loss_pips", p.stop_loss_pips)?;
        if let Some(tp) = p.take_profit_pips {
            require_positive("take_profit_pips", tp)?;
        }
        let proof = self.ensure_demo().await?;
        let mut body = json!({
            "symbol": symbol,
            "side": p.side.as_wire(),
            "orderType": p.order_type.as_wire(),
            "volumeLots": p.volume_lots,
            "triggerPrice": p.trigger_price,
            "stopLossPips": p.stop_loss_pips,
        });
        if let Some(tp) = p.take_profit_pips {
            body["takeProfitPips"] = json!(tp);
        }
        if let Some(expiry) = p.expiry_unix_ms {
            body["expiryUnixMs"] = json!(expiry);
        }
        if let Some(comment) = p.comment {
            body["comment"] = json!(comment);
        }
        self.trade_post("/orders/pending", &body, proof).await
    }

    pub async fn op_list_pending_orders(&self) -> Result<Value, ToolError> {
        self.get_json("/orders/pending", &[]).await
    }

    pub async fn op_cancel_pending_order(
        &self,
        p: CancelPendingOrderParams,
    ) -> Result<Value, ToolError> {
        if p.order_id <= 0 {
            return Err(ToolError("order_id must be positive".to_string()));
        }
        let proof = self.ensure_demo().await?;
        self.trade_post("/orders/cancel", &json!({ "orderId": p.order_id }), proof)
            .await
    }

    // ── Trading — human-confirm queue ─────────────────────────────────────

    pub async fn op_list_pending_actions(&self) -> Result<Value, ToolError> {
        self.get_json("/actions/pending", &[]).await
    }

    pub async fn op_confirm_pending_action(
        &self,
        p: ConfirmPendingActionParams,
    ) -> Result<Value, ToolError> {
        let id = p.action_id.trim();
        if id.is_empty() {
            return Err(ToolError("action_id must be non-empty".to_string()));
        }
        // Confirm EXECUTES the queued broker call — demo-guarded.
        let proof = self.ensure_demo().await?;
        let body = match p.volume_units_override {
            Some(v) if v > 0 => json!({ "volumeUnitsOverride": v }),
            Some(v) => {
                return Err(ToolError(format!(
                    "volume_units_override must be positive (got {v})"
                )));
            }
            None => json!({}),
        };
        self.trade_post(&format!("/actions/{id}/confirm"), &body, proof)
            .await
    }

    pub async fn op_reject_pending_action(
        &self,
        p: RejectPendingActionParams,
    ) -> Result<Value, ToolError> {
        let id = p.action_id.trim();
        if id.is_empty() {
            return Err(ToolError("action_id must be non-empty".to_string()));
        }
        let body = match p.reason {
            Some(reason) => json!({ "reason": reason }),
            None => json!({}),
        };
        self.post_json(&format!("/actions/{id}/reject"), &body)
            .await
    }

    // ── Autopilot ─────────────────────────────────────────────────────────

    pub async fn op_autopilot_start(&self, p: AutopilotStartParams) -> Result<Value, ToolError> {
        if p.portfolio_paths.iter().all(|s| s.trim().is_empty()) {
            return Err(ToolError(
                "portfolio_paths must contain at least one portfolio artifact path — run \
                 list_portfolios to see what is available"
                    .to_string(),
            ));
        }
        require_positive("lot_size", p.lot_size)?;
        let proof = self.ensure_demo().await?;
        // Backend body is snake_case (StartLiveBody has no rename_all).
        let mut body = json!({
            "portfolio_paths": p.portfolio_paths,
            "lot_size": p.lot_size,
        });
        if let Some(v) = p.stop_loss_pips {
            body["stop_loss_pips"] = json!(v);
        }
        if let Some(v) = p.take_profit_pips {
            body["take_profit_pips"] = json!(v);
        }
        if let Some(v) = p.warmup_bars {
            body["warmup_bars"] = json!(v);
        }
        if let Some(v) = p.cull_after_consecutive_losses {
            body["cull_after_consecutive_losses"] = json!(v);
        }
        if let Some(v) = p.cull_min_win_rate_pct {
            body["cull_min_win_rate_pct"] = json!(v);
        }
        if let Some(v) = p.cull_window_trades {
            body["cull_window_trades"] = json!(v);
        }
        self.trade_post("/autonomous/start", &body, proof).await
    }

    /// GUARD-EXEMPT by design (§2): stopping executes no broker order — it
    /// halts engines and leaves positions untouched — and must remain
    /// available precisely when the guard is refusing.
    pub async fn op_autopilot_stop(&self) -> Result<Value, ToolError> {
        self.post_json("/autonomous/stop", &json!({})).await
    }

    pub async fn op_autopilot_status(&self) -> Result<Value, ToolError> {
        self.get_json("/autonomous/status", &[]).await
    }

    pub async fn op_demo_gate_status(&self, p: DemoGateStatusParams) -> Result<Value, ToolError> {
        self.get_json("/autonomous/gate", &[("portfolio", p.portfolio)])
            .await
    }

    pub async fn op_replay_backtest(&self, p: ReplayBacktestParams) -> Result<Value, ToolError> {
        let mut body = json!({});
        if let Some(symbol) = p.symbol {
            body["symbol"] = json!(symbol);
        }
        if let Some(base_tf) = p.base_tf {
            body["base_tf"] = json!(base_tf);
        }
        self.post_json("/autonomous/replay", &body).await
    }

    pub async fn op_parity_check(&self, p: ParityCheckParams) -> Result<Value, ToolError> {
        let mut query = vec![("portfolio", p.portfolio)];
        if let Some(v) = p.window {
            query.push(("window", v.to_string()));
        }
        if let Some(v) = p.reference {
            query.push(("reference", v.to_string()));
        }
        self.get_json("/autonomous/parity", &query).await
    }

    pub async fn op_tail_risk(&self, p: TailRiskParams) -> Result<Value, ToolError> {
        let mut query = vec![("portfolio", p.portfolio)];
        if let Some(v) = p.iterations {
            query.push(("iterations", v.to_string()));
        }
        if let Some(v) = p.risk {
            query.push(("risk", v.to_string()));
        }
        self.get_json("/autonomous/tailrisk", &query).await
    }

    pub async fn op_challenge_sim(&self, p: ChallengeSimParams) -> Result<Value, ToolError> {
        let mut query = vec![("portfolio", p.portfolio)];
        if let Some(v) = p.iterations {
            query.push(("iterations", v.to_string()));
        }
        self.get_json("/autonomous/challenge", &query).await
    }

    // ── Discovery and training (job pattern) ──────────────────────────────

    pub async fn op_discovery_start(&self, p: DiscoveryStartParams) -> Result<Value, ToolError> {
        // Backend StartJobBody is snake_case.
        let mut body = json!({});
        if let Some(v) = p.symbol {
            body["symbol"] = json!(v);
        }
        if let Some(v) = p.base_tf {
            body["base_tf"] = json!(v);
        }
        if let Some(v) = p.higher_tfs {
            body["higher_tfs"] = json!(v);
        }
        if let Some(v) = p.population {
            body["population"] = json!(v);
        }
        if let Some(v) = p.generations {
            body["generations"] = json!(v);
        }
        if let Some(v) = p.max_indicators {
            body["max_indicators"] = json!(v);
        }
        if let Some(v) = p.target_candidates {
            body["target_candidates"] = json!(v);
        }
        if let Some(v) = p.portfolio_size {
            body["portfolio_size"] = json!(v);
        }
        self.post_json("/engines/discovery/start", &body).await
    }

    pub async fn op_discovery_stop(&self) -> Result<Value, ToolError> {
        self.post_json("/engines/discovery/stop", &json!({})).await
    }

    pub async fn op_training_start(&self, p: TrainingStartParams) -> Result<Value, ToolError> {
        let mut body = json!({});
        if let Some(v) = p.symbol {
            body["symbol"] = json!(v);
        }
        if let Some(v) = p.base_tf {
            body["base_tf"] = json!(v);
        }
        if let Some(v) = p.higher_tfs {
            body["higher_tfs"] = json!(v);
        }
        self.post_json("/engines/training/start", &body).await
    }

    pub async fn op_training_stop(&self) -> Result<Value, ToolError> {
        self.post_json("/engines/training/stop", &json!({})).await
    }

    pub async fn op_experience_train(&self) -> Result<Value, ToolError> {
        self.post_json("/experience/train", &json!({})).await
    }

    // ── Strategy lab and portfolios ───────────────────────────────────────

    pub async fn op_promotion_gate_status(&self) -> Result<Value, ToolError> {
        self.get_json("/strategy_lab/promotion", &[]).await
    }

    pub async fn op_promote_strategies(
        &self,
        p: PromoteStrategiesParams,
    ) -> Result<Value, ToolError> {
        let proof = self.ensure_demo().await?;
        // Backend PromoteBody is camelCase with deny_unknown_fields.
        let mut body = json!({});
        if let Some(v) = p.symbol {
            body["symbol"] = json!(v);
        }
        if let Some(v) = p.base_tf {
            body["baseTf"] = json!(v);
        }
        self.trade_post("/strategy_lab/promote", &body, proof).await
    }

    pub async fn op_list_portfolios(&self) -> Result<Value, ToolError> {
        self.get_json("/portfolios/list", &[]).await
    }

    pub async fn op_list_strategies(&self) -> Result<Value, ToolError> {
        self.get_json("/strategy/list", &[]).await
    }

    pub async fn op_strategy_report(&self, p: StrategyReportParams) -> Result<Value, ToolError> {
        let (Some(dir), Some(base)) = (p.dir, p.base) else {
            return Err(ToolError(
                "strategy_report needs both `dir` and `base` — take them from a \
                 list_strategies row"
                    .to_string(),
            ));
        };
        self.get_json("/strategy/report", &[("dir", dir), ("base", base)])
            .await
    }

    pub async fn op_strategy_blacklist(&self) -> Result<Value, ToolError> {
        self.get_json("/strategy/blacklist", &[]).await
    }

    // ── Settings and risk ─────────────────────────────────────────────────

    pub async fn op_get_settings(&self) -> Result<Value, ToolError> {
        self.get_json("/settings", &[]).await
    }

    pub async fn op_update_settings(&self, p: UpdateSettingsParams) -> Result<Value, ToolError> {
        // Typed fields only, no passthrough (design §1.4 / §3 #49). The
        // backend DTO is camelCase.
        let mut body = json!({});
        if let Some(v) = p.trading_mode {
            body["tradingMode"] = json!(v);
        }
        if let Some(v) = p.compute_mode {
            body["computeMode"] = json!(v);
        }
        if let Some(v) = p.risk_per_trade {
            body["riskPerTrade"] = json!(v);
        }
        if let Some(v) = p.max_portfolio_risk {
            body["maxPortfolioRisk"] = json!(v);
        }
        if let Some(v) = p.search_population {
            body["searchPopulation"] = json!(v);
        }
        if let Some(v) = p.search_generations {
            body["searchGenerations"] = json!(v);
        }
        if let Some(v) = p.search_max_hours {
            body["searchMaxHours"] = json!(v);
        }
        if let Some(v) = p.search_max_indicators {
            body["searchMaxIndicators"] = json!(v);
        }
        if let Some(v) = p.search_portfolio_size {
            body["searchPortfolioSize"] = json!(v);
        }
        if let Some(v) = p.search_corr_threshold {
            body["searchCorrThreshold"] = json!(v);
        }
        if let Some(v) = p.search_max_rows {
            body["searchMaxRows"] = json!(v);
        }
        if let Some(v) = p.prefilter_top_k {
            body["prefilterTopK"] = json!(v);
        }
        if let Some(v) = p.convergence_patience {
            body["convergencePatience"] = json!(v);
        }
        if let Some(v) = p.stagnation_patience {
            body["stagnationPatience"] = json!(v);
        }
        if let Some(v) = p.novelty_weight {
            body["noveltyWeight"] = json!(v);
        }
        if let Some(v) = p.disable_smc_gate {
            body["disableSmcGate"] = json!(v);
        }
        if let Some(v) = p.live_ml_gate {
            body["liveMlGate"] = json!(v);
        }
        if let Some(v) = p.risky_start_balance {
            body["riskyStartBalance"] = json!(v);
        }
        if let Some(v) = p.risky_target_balance {
            body["riskyTargetBalance"] = json!(v);
        }
        if let Some(v) = p.risky_horizon_days {
            body["riskyHorizonDays"] = json!(v);
        }
        if let Some(v) = p.news_calendar_enabled {
            body["newsCalendarEnabled"] = json!(v);
        }
        if let Some(v) = p.news_calendar_source {
            body["newsCalendarSource"] = json!(v);
        }
        if let Some(v) = p.news_trading_mode {
            body["newsTradingMode"] = json!(v);
        }
        let Some(obj) = body.as_object() else {
            return Err(ToolError(
                "internal: settings body was not an object".to_string(),
            ));
        };
        if obj.is_empty() {
            return Err(ToolError(
                "update_settings was called with no fields — pass at least one knob (see \
                 knob_catalog for what each one does)"
                    .to_string(),
            ));
        }
        self.post_json("/settings", &body).await
    }

    pub async fn op_knob_catalog(&self) -> Result<Value, ToolError> {
        let catalog = self.get_json("/settings/knob-catalog", &[]).await?;
        let presets = self.get_json("/settings/presets", &[]).await?;
        Ok(json!({ "knobCatalog": catalog, "presets": presets }))
    }

    pub async fn op_get_risk(&self) -> Result<Value, ToolError> {
        self.get_json("/risk", &[]).await
    }

    pub async fn op_apply_risk_preset(&self, p: ApplyRiskPresetParams) -> Result<Value, ToolError> {
        let preset = p.preset.trim().to_lowercase();
        if preset.is_empty() {
            return Err(ToolError(
                "preset must be one of: ftmo, myforexfunds, fundednext, the5ers, none".to_string(),
            ));
        }
        let proof = self.ensure_demo().await?;
        self.trade_post("/risk/preset", &json!({ "preset": preset }), proof)
            .await
    }

    pub async fn op_risky_scenarios(&self, p: RiskyScenariosParams) -> Result<Value, ToolError> {
        // Backend RiskyScenarioQuery is camelCase.
        let mut query: Vec<(&str, String)> = Vec::new();
        for (name, v) in [
            ("startingUsd", p.starting_usd),
            ("targetUsd", p.target_usd),
            ("riskFraction", p.risk_fraction),
            ("winRate", p.win_rate),
            ("rewardToRisk", p.reward_to_risk),
        ] {
            if let Some(v) = v {
                query.push((name, v.to_string()));
            }
        }
        self.get_json("/risky/scenarios", &query).await
    }

    // ── Journal and analytics ─────────────────────────────────────────────

    pub async fn op_journal_trades(&self, p: JournalTradesParams) -> Result<Value, ToolError> {
        // Backend JournalQuery is camelCase.
        let mut query: Vec<(&str, String)> = Vec::new();
        if let Some(v) = p.from_ms {
            query.push(("fromMs", v.to_string()));
        }
        if let Some(v) = p.to_ms {
            query.push(("toMs", v.to_string()));
        }
        if let Some(v) = p.limit {
            query.push(("limit", v.to_string()));
        }
        self.get_json("/journal/trades", &query).await
    }

    pub async fn op_journal_stats(&self) -> Result<Value, ToolError> {
        self.get_json("/journal/stats", &[]).await
    }

    pub async fn op_journal_analytics(&self) -> Result<Value, ToolError> {
        self.get_json("/journal/analytics", &[]).await
    }

    // ── Data management ───────────────────────────────────────────────────

    pub async fn op_data_coverage(&self) -> Result<Value, ToolError> {
        self.get_json("/data/bootstrap", &[]).await
    }

    pub async fn op_fetch_history(&self, p: FetchHistoryParams) -> Result<Value, ToolError> {
        let symbol = p.symbol.trim().to_uppercase();
        let timeframe = p.timeframe.trim().to_uppercase();
        if symbol.is_empty() || timeframe.is_empty() {
            return Err(ToolError(
                "symbol and timeframe must be non-empty".to_string(),
            ));
        }
        let mut body = json!({
            "symbol": symbol,
            "timeframe": timeframe,
            "fromMs": p.from_ms,
        });
        if let Some(to) = p.to_ms {
            body["toMs"] = json!(to);
        }
        self.post_json("/data/fetch", &body).await
    }

    pub async fn op_import_data(&self, p: ImportDataParams) -> Result<Value, ToolError> {
        let source_path = p.source_path.trim().to_string();
        let symbol = p.symbol.trim().to_uppercase();
        let timeframe = p.timeframe.trim().to_uppercase();
        if source_path.is_empty() || symbol.is_empty() || timeframe.is_empty() {
            return Err(ToolError(
                "source_path, symbol, and timeframe must all be non-empty".to_string(),
            ));
        }
        self.post_json(
            "/data/import",
            &json!({
                "sourcePath": source_path,
                "symbol": symbol,
                "timeframe": timeframe,
            }),
        )
        .await
    }
}
