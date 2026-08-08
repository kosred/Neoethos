//! The MCP tool surface: 62 tools over the NeoEthos backend HTTP API.
//!
//! Every handler is a thin shim over a `Backend::op_*` method (`ops.rs`) —
//! all behavior (validation, demo guard, wire conversion) lives there where
//! the tier-1 tests can reach it without driving the MCP protocol.
//!
//! Annotation policy (design §3): `read_only_hint=true` for pure reads;
//! `destructive_hint=true` for the demo-guarded trade set AND for
//! `update_settings` (honest UX hint — knobs alter live sizing — though it
//! executes no trade and is not guarded); `open_world_hint=true` only on
//! `news_briefing` (outbound RSS fetch); `idempotent_hint=true` on the stop
//! tools. Annotations are UX hints for Codex approval routing, never
//! security — the in-handler guard holds even under approval_mode=auto.

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};

use crate::backend::{Backend, ToolError};
use crate::params::*;

/// Render an op result as an MCP tool result: success carries the backend's
/// JSON verbatim (pretty-printed), failure carries the error message as
/// `isError: true` content the agent reads directly.
fn done(result: Result<serde_json::Value, ToolError>) -> CallToolResult {
    match result {
        Ok(value) => {
            let text = serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
            CallToolResult::success(vec![ContentBlock::text(text)])
        }
        Err(err) => CallToolResult::error(vec![ContentBlock::text(err.0)]),
    }
}

/// The Codex control plane: an MCP server (stdio) wrapping the localhost
/// NeoEthos backend API, with a fail-closed demo-only guard on every
/// trade-affecting tool.
pub struct ControlPlane {
    backend: Backend,
    tool_router: ToolRouter<Self>,
}

impl ControlPlane {
    pub fn new(base_url: &str, token: Option<String>) -> anyhow::Result<Self> {
        Ok(Self {
            backend: Backend::new(base_url, token)?,
            tool_router: Self::tool_router(),
        })
    }

    pub fn backend(&self) -> &Backend {
        &self.backend
    }
}

#[tool_router(vis = "pub")]
impl ControlPlane {
    // ── System and health ─────────────────────────────────────────────────

    #[tool(
        description = "Backend liveness check: GET /healthz plus the HTTP round-trip latency in \
                       milliseconds. Run this first if any other tool reports the backend as \
                       unreachable.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn system_health(&self) -> CallToolResult {
        done(self.backend.op_system_health().await)
    }

    #[tool(
        description = "Hardware inventory of the machine running the backend: CPU, RAM, GPUs and \
                       the compute lane the engines will use.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn hardware_info(&self) -> CallToolResult {
        done(self.backend.op_hardware_info().await)
    }

    #[tool(
        description = "Live state of the discovery/training/auto-trader engines with progress \
                       percent and counters. This is the POLL half of the job pattern: call \
                       discovery_start or training_start, then poll here until the state leaves \
                       'Running'.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn engine_status(&self) -> CallToolResult {
        done(self.backend.op_engine_status().await)
    }

    #[tool(
        description = "Where the backend stores data, models, journal and config on disk.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn storage_paths(&self) -> CallToolResult {
        done(self.backend.op_storage_paths().await)
    }

    #[tool(
        description = "Generate a diagnostics ZIP (logs + config) on the operator's desktop and \
                       return its path. Writes a file; touches nothing else.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    pub async fn diagnostics_report(&self) -> CallToolResult {
        done(self.backend.op_diagnostics_report().await)
    }

    // ── Account and broker state ──────────────────────────────────────────

    #[tool(
        description = "Broker adapter status merged with the OAuth account list: environment \
                       (Demo/Live), connected flag, active account id, and per-account isLive. \
                       This is the agent-visible face of the demo guard — trade tools execute \
                       only when this reports connected=true, environment=Demo and \
                       isLive=false for the active account.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn broker_status(&self) -> CallToolResult {
        done(self.backend.op_broker_status().await)
    }

    #[tool(
        description = "Account snapshot: balance, equity, margins, currency and all open \
                       positions. Pass refresh=true to force a broker round-trip first \
                       (read-only, no trade side effects).",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn account_snapshot(
        &self,
        Parameters(p): Parameters<AccountSnapshotParams>,
    ) -> CallToolResult {
        done(self.backend.op_account_snapshot(p).await)
    }

    #[tool(
        description = "The open positions array from the account snapshot: positionId, symbol, \
                       side, volume in lots AND broker wire units, entry/SL/TP prices, PnL in \
                       pips and account currency.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn list_positions(&self) -> CallToolResult {
        done(self.backend.op_list_positions().await)
    }

    #[tool(
        description = "Every symbol this broker account can trade, with ids, enabled flags and \
                       asset classes.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn broker_symbols(&self) -> CallToolResult {
        done(self.backend.op_broker_symbols().await)
    }

    #[tool(
        description = "The canonical timeframe labels this broker accepts (M1..MN1). Call this \
                       instead of guessing the timeframe string for get_chart / get_indicator.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn broker_timeframes(&self) -> CallToolResult {
        done(self.backend.op_broker_timeframes().await)
    }

    #[tool(
        description = "Scroll-back page of chart candles STRICTLY OLDER than before_ms (Unix ms) \
                       — the pagination companion to get_chart, which returns only the trailing \
                       window. Page back by passing the oldest candle's time as the next before_ms.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn chart_history(
        &self,
        Parameters(p): Parameters<ChartHistoryParams>,
    ) -> CallToolResult {
        done(self.backend.op_chart_history(p).await)
    }

    #[tool(
        description = "What the model swarm currently knows: the discovery-targets scan and \
                       intelligence summary the desktop surfaces.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn intelligence(&self) -> CallToolResult {
        done(self.backend.op_intelligence().await)
    }

    #[tool(
        description = "Broker-side order history for a time window (default: the last 7 days, \
                       the broker's maximum window). Times are Unix milliseconds.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn order_history(
        &self,
        Parameters(p): Parameters<OrderHistoryParams>,
    ) -> CallToolResult {
        done(self.backend.op_order_history(p).await)
    }

    #[tool(
        description = "Deposit/withdrawal cash-flow history from the broker (last 7 days by \
                       default).",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn cashflow(&self) -> CallToolResult {
        done(self.backend.op_cashflow().await)
    }

    #[tool(
        description = "Expected margin the broker would require for a hypothetical volume of a \
                       symbol. Pure quote — no order is placed.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn expected_margin(
        &self,
        Parameters(p): Parameters<ExpectedMarginParams>,
    ) -> CallToolResult {
        done(self.backend.op_expected_margin(p).await)
    }

    // ── Market data ───────────────────────────────────────────────────────

    #[tool(
        description = "OHLCV candles for a symbol/timeframe (broker-live when connected, disk \
                       cache otherwise; the `source` field says which).",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn get_chart(&self, Parameters(p): Parameters<GetChartParams>) -> CallToolResult {
        done(self.backend.op_get_chart(p).await)
    }

    #[tool(
        description = "Computed technical-indicator series over the chart data: sma, ema, rsi, \
                       macd, bollinger_bands, atr, stoch, adx, vwap.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn get_indicator(
        &self,
        Parameters(p): Parameters<GetIndicatorParams>,
    ) -> CallToolResult {
        done(self.backend.op_get_indicator(p).await)
    }

    #[tool(
        description = "Latest live bid/ask spot ticks for the watchlist symbols (sub-2s \
                       freshness for active majors while the broker session is up).",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn live_spots(&self) -> CallToolResult {
        done(self.backend.op_live_spots().await)
    }

    #[tool(
        description = "Session-aware spread statistics recorded from the broker's own ticks — \
                       what the spread actually costs per symbol and session.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn spread_stats(&self) -> CallToolResult {
        done(self.backend.op_spread_stats().await)
    }

    #[tool(
        description = "Financial news headlines fetched server-side from public RSS feeds, plus \
                       the market briefing when available. force=true bypasses the cache.",
        annotations(read_only_hint = true, open_world_hint = true)
    )]
    pub async fn news_briefing(
        &self,
        Parameters(p): Parameters<NewsBriefingParams>,
    ) -> CallToolResult {
        done(self.backend.op_news_briefing(p).await)
    }

    #[tool(
        description = "The saved Market Watch watchlist symbols.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn get_watchlist(&self) -> CallToolResult {
        done(self.backend.op_get_watchlist().await)
    }

    #[tool(
        description = "Replace the Market Watch watchlist. The live spot stream re-subscribes \
                       within ~5 s. No trade side effects.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    pub async fn set_watchlist(
        &self,
        Parameters(p): Parameters<SetWatchlistParams>,
    ) -> CallToolResult {
        done(self.backend.op_set_watchlist(p).await)
    }

    // ── Trading — direct orders (ALL demo-guarded) ────────────────────────

    #[tool(
        description = "Open a market position on the DEMO account. stop_loss_pips is REQUIRED — \
                       this control plane never places a naked order (risky:false is hardcoded). \
                       The server re-verifies the active account is a demo account on every call \
                       and refuses otherwise.",
        annotations(
            read_only_hint = false,
            destructive_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn open_position(
        &self,
        Parameters(p): Parameters<OpenPositionParams>,
    ) -> CallToolResult {
        done(self.backend.op_open_position(p).await)
    }

    #[tool(
        description = "Close an open position on the DEMO account, fully (omit volume_lots) or \
                       partially. The server reads the position's current volume and converts \
                       lots to broker wire units itself. Demo-guarded on every call.",
        annotations(
            read_only_hint = false,
            destructive_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn close_position(
        &self,
        Parameters(p): Parameters<ClosePositionParams>,
    ) -> CallToolResult {
        done(self.backend.op_close_position(p).await)
    }

    #[tool(
        description = "Amend an open DEMO position's protection: absolute stop-loss price, \
                       absolute take-profit price, and/or the trailing-stop flag (at least one \
                       required). Demo-guarded on every call.",
        annotations(
            read_only_hint = false,
            destructive_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn modify_position_protection(
        &self,
        Parameters(p): Parameters<ModifyPositionProtectionParams>,
    ) -> CallToolResult {
        done(self.backend.op_modify_position_protection(p).await)
    }

    #[tool(
        description = "Place a resting limit/stop order on the DEMO account that fills when \
                       price reaches trigger_price. stop_loss_pips is REQUIRED. Good-Till-Cancel \
                       unless expiry_unix_ms is set. Demo-guarded on every call.",
        annotations(
            read_only_hint = false,
            destructive_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn place_pending_order(
        &self,
        Parameters(p): Parameters<PlacePendingOrderParams>,
    ) -> CallToolResult {
        done(self.backend.op_place_pending_order(p).await)
    }

    #[tool(
        description = "List the broker's resting (limit/stop) orders with ids, triggers and \
                       protection levels.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn list_pending_orders(&self) -> CallToolResult {
        done(self.backend.op_list_pending_orders().await)
    }

    #[tool(
        description = "Cancel a resting order on the DEMO account by order_id. Demo-guarded on \
                       every call.",
        annotations(
            read_only_hint = false,
            destructive_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn cancel_pending_order(
        &self,
        Parameters(p): Parameters<CancelPendingOrderParams>,
    ) -> CallToolResult {
        done(self.backend.op_cancel_pending_order(p).await)
    }

    // ── Trading — human-confirm queue ─────────────────────────────────────

    #[tool(
        description = "The trade-management confirmation queue: proposed actions awaiting \
                       confirm/reject, plus recent history.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn list_pending_actions(&self) -> CallToolResult {
        done(self.backend.op_list_pending_actions().await)
    }

    #[tool(
        description = "Confirm a queued action — this EXECUTES the underlying broker call on the \
                       DEMO account (e.g. the queued close). Demo-guarded on every call.",
        annotations(
            read_only_hint = false,
            destructive_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn confirm_pending_action(
        &self,
        Parameters(p): Parameters<ConfirmPendingActionParams>,
    ) -> CallToolResult {
        done(self.backend.op_confirm_pending_action(p).await)
    }

    #[tool(
        description = "Reject a queued action with an optional reason. Audit-only — no broker \
                       side effects.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    pub async fn reject_pending_action(
        &self,
        Parameters(p): Parameters<RejectPendingActionParams>,
    ) -> CallToolResult {
        done(self.backend.op_reject_pending_action(p).await)
    }

    // ── Autopilot ─────────────────────────────────────────────────────────

    #[tool(
        description = "Start the autonomous trader on one or more discovered portfolios — it \
                       will open/close DEMO positions on its own until stopped. Demo-guarded on \
                       every call. Poll autopilot_status; stop with autopilot_stop.",
        annotations(
            read_only_hint = false,
            destructive_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn autopilot_start(
        &self,
        Parameters(p): Parameters<AutopilotStartParams>,
    ) -> CallToolResult {
        done(self.backend.op_autopilot_start(p).await)
    }

    #[tool(
        description = "Stop ALL running autopilot engines. Executes no broker order and leaves \
                       positions untouched — deliberately NOT demo-guarded so trading can always \
                       be halted, even while the guard is refusing. Idempotent.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn autopilot_stop(&self) -> CallToolResult {
        done(self.backend.op_autopilot_stop().await)
    }

    #[tool(
        description = "Live status of every running autopilot engine: running flag, per-engine \
                       trade counts, last signal, cull state.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn autopilot_status(&self) -> CallToolResult {
        done(self.backend.op_autopilot_status().await)
    }

    #[tool(
        description = "Demo forward-test gate verdict for a portfolio: eligibility, criteria \
                       breakdown, and whether the gate would be enforced (Live) or is \
                       informational (Demo).",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn demo_gate_status(
        &self,
        Parameters(p): Parameters<DemoGateStatusParams>,
    ) -> CallToolResult {
        done(self.backend.op_demo_gate_status(p).await)
    }

    #[tool(
        description = "Offline dry-run of the autonomous trader over on-disk history — the SAME \
                       engine and stats as the CLI trader-replay. ZERO broker calls. Long \
                       synchronous call: can take minutes on deep history.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn replay_backtest(
        &self,
        Parameters(p): Parameters<ReplayBacktestParams>,
    ) -> CallToolResult {
        done(self.backend.op_replay_backtest(p).await)
    }

    #[tool(
        description = "Live-vs-backtest parity harness: does the live-style bar window produce \
                       the SAME signals as a long-history computation for this portfolio? FAIL \
                       means live trading cannot match the validated backtest.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn parity_check(
        &self,
        Parameters(p): Parameters<ParityCheckParams>,
    ) -> CallToolResult {
        done(self.backend.op_parity_check(p).await)
    }

    #[tool(
        description = "Monte-Carlo tail risk of a portfolio's logged trade sequence: max-drawdown \
                       distribution (p95) and ruin probability at the current risk sizing.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn tail_risk(&self, Parameters(p): Parameters<TailRiskParams>) -> CallToolResult {
        done(self.backend.op_tail_risk(p).await)
    }

    #[tool(
        description = "First-passage Monte-Carlo of a prop-firm challenge for a portfolio: pass \
                       and funded probability per risk size, and the challenge-optimal sizing.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn challenge_sim(
        &self,
        Parameters(p): Parameters<ChallengeSimParams>,
    ) -> CallToolResult {
        done(self.backend.op_challenge_sim(p).await)
    }

    // ── Discovery and training (job pattern) ──────────────────────────────

    #[tool(
        description = "Start a strategy-discovery (GA search) job. Returns as soon as the \
                       backend accepts it — poll engine_status for progress, stop with \
                       discovery_stop. Omitted fields resolve from config.yaml exactly like the \
                       CLI. 409 if a discovery is already running.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    pub async fn discovery_start(
        &self,
        Parameters(p): Parameters<DiscoveryStartParams>,
    ) -> CallToolResult {
        done(self.backend.op_discovery_start(p).await)
    }

    #[tool(
        description = "Request cancellation of the running discovery job. Idempotent — returns \
                       running=false when nothing was running.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn discovery_stop(&self) -> CallToolResult {
        done(self.backend.op_discovery_stop().await)
    }

    #[tool(
        description = "Start a model-training job for a symbol/timeframe (needs a finished \
                       discovery's model_targets.json). Job pattern: poll engine_status, stop \
                       with training_stop. On success, discovery auto-chains here by itself.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    pub async fn training_start(
        &self,
        Parameters(p): Parameters<TrainingStartParams>,
    ) -> CallToolResult {
        done(self.backend.op_training_start(p).await)
    }

    #[tool(
        description = "Request cancellation of the running training job. Idempotent.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn training_stop(&self) -> CallToolResult {
        done(self.backend.op_training_stop().await)
    }

    #[tool(
        description = "Offline learning pass over the recorded live experience. Writes models \
                       and a report; never touches live trading.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    pub async fn experience_train(&self) -> CallToolResult {
        done(self.backend.op_experience_train().await)
    }

    // ── Strategy lab and portfolios ───────────────────────────────────────

    #[tool(
        description = "Promotion-gate verdict for the latest portfolio of the configured \
                       symbol/timeframe: aggregate metrics vs the operator's thresholds, \
                       criterion by criterion.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn promotion_gate_status(&self) -> CallToolResult {
        done(self.backend.op_promotion_gate_status().await)
    }

    #[tool(
        description = "Promote the trained strategies to live_models/ IF the promotion gate \
                       passes — this changes what the autopilot trades next. The gate is \
                       re-evaluated server-side; a failing gate refuses (HTTP 412). \
                       Demo-guarded on every call.",
        annotations(
            read_only_hint = false,
            destructive_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn promote_strategies(
        &self,
        Parameters(p): Parameters<PromoteStrategiesParams>,
    ) -> CallToolResult {
        done(self.backend.op_promote_strategies(p).await)
    }

    #[tool(
        description = "Discovered portfolio artifacts on disk — the inputs autopilot_start and \
                       the analysis tools (tail_risk, parity_check, challenge_sim) take.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn list_portfolios(&self) -> CallToolResult {
        done(self.backend.op_list_portfolios().await)
    }

    #[tool(
        description = "Per-strategy list with trades, verdicts and report locations (dir/base \
                       pairs for strategy_report).",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn list_strategies(&self) -> CallToolResult {
        done(self.backend.op_list_strategies().await)
    }

    #[tool(
        description = "Monthly/yearly performance report for one strategy (dir + base from \
                       list_strategies).",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn strategy_report(
        &self,
        Parameters(p): Parameters<StrategyReportParams>,
    ) -> CallToolResult {
        done(self.backend.op_strategy_report(p).await)
    }

    #[tool(
        description = "The permanent auto-cull blacklist: retired strategies that can never be \
                       traded again.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn strategy_blacklist(&self) -> CallToolResult {
        done(self.backend.op_strategy_blacklist().await)
    }

    // ── Settings and risk ─────────────────────────────────────────────────

    #[tool(
        description = "The typed app settings: trading mode, compute mode, search knobs, risky \
                       goals, news configuration.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn get_settings(&self) -> CallToolResult {
        done(self.backend.op_get_settings().await)
    }

    #[tool(
        description = "Partial typed settings update — only the fields you pass change; \
                       everything else keeps its on-disk value. No raw-YAML passthrough. \
                       Marked destructive as an honest hint: several knobs (risk_per_trade, \
                       max_portfolio_risk, trading_mode) alter live position sizing. Read \
                       knob_catalog first.",
        annotations(
            read_only_hint = false,
            destructive_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn update_settings(
        &self,
        Parameters(p): Parameters<UpdateSettingsParams>,
    ) -> CallToolResult {
        done(self.backend.op_update_settings(p).await)
    }

    #[tool(
        description = "Machine-readable catalog of every runtime knob (help text, ranges, \
                       defaults) plus the available presets — read this before touching \
                       update_settings.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn knob_catalog(&self) -> CallToolResult {
        done(self.backend.op_knob_catalog().await)
    }

    #[tool(
        description = "Current risk configuration: per-trade risk, drawdown limits, lot caps, \
                       active prop-firm preset and the available presets.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn get_risk(&self) -> CallToolResult {
        done(self.backend.op_get_risk().await)
    }

    #[tool(
        description = "Switch the active prop-firm risk preset (ftmo, myforexfunds, fundednext, \
                       the5ers, none) — REWRITES the live sizing configuration the running \
                       engines use. Demo-guarded on every call.",
        annotations(
            read_only_hint = false,
            destructive_hint = true,
            open_world_hint = false
        )
    )]
    pub async fn apply_risk_preset(
        &self,
        Parameters(p): Parameters<ApplyRiskPresetParams>,
    ) -> CallToolResult {
        done(self.backend.op_apply_risk_preset(p).await)
    }

    #[tool(
        description = "Risky/Growth-mode time-to-target projection computed by the live engine's \
                       own math: expected days to target and ruin probability for a given edge.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn risky_scenarios(
        &self,
        Parameters(p): Parameters<RiskyScenariosParams>,
    ) -> CallToolResult {
        done(self.backend.op_risky_scenarios(p).await)
    }

    // ── Journal and analytics ─────────────────────────────────────────────

    #[tool(
        description = "Closed trades of the ACTIVE account from the journal store, most-recent \
                       first (default limit 500). Times are Unix milliseconds.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn journal_trades(
        &self,
        Parameters(p): Parameters<JournalTradesParams>,
    ) -> CallToolResult {
        done(self.backend.op_journal_trades(p).await)
    }

    #[tool(
        description = "Computed performance stats over the journal: win rate, profit factor, \
                       expectancy, drawdown, equity curve.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn journal_stats(&self) -> CallToolResult {
        done(self.backend.op_journal_stats().await)
    }

    #[tool(
        description = "Per-trade analytics in comparable units: pips, R-multiples and MFE/MAE \
                       excursion recovered from the price series, grouped by symbol, hour, \
                       weekday and direction — locates WHERE the account makes or loses money.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn journal_analytics(&self) -> CallToolResult {
        done(self.backend.op_journal_analytics().await)
    }

    // ── Data management ───────────────────────────────────────────────────

    #[tool(
        description = "On-disk market-data inventory: data dir, symbols present, file count, \
                       last touch time.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    pub async fn data_coverage(&self) -> CallToolResult {
        done(self.backend.op_data_coverage().await)
    }

    #[tool(
        description = "Download historical bars from the broker into the on-disk store for a \
                       symbol/timeframe/window. Writes to disk; can run long for deep windows.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    pub async fn fetch_history(
        &self,
        Parameters(p): Parameters<FetchHistoryParams>,
    ) -> CallToolResult {
        done(self.backend.op_fetch_history(p).await)
    }

    #[tool(
        description = "Import a local CSV/TSV/Parquet/JSON/JSONL/Arrow file into the canonical \
                       on-disk market-data layout for a symbol/timeframe.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    pub async fn import_data(&self, Parameters(p): Parameters<ImportDataParams>) -> CallToolResult {
        done(self.backend.op_import_data(p).await)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ControlPlane {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("neoethos", env!("CARGO_PKG_VERSION"))
                    .with_title("NeoEthos Codex control plane"),
            )
            .with_instructions(
                "Full operational control of the NeoEthos trading backend: account state, \
                 charts, journal analytics, discovery/training jobs, autopilot, settings, and \
                 DEMO trade execution. Every trade-affecting tool re-verifies on each call \
                 that the active broker account is a demo account and refuses otherwise \
                 (fail-closed) — no tool on this server can ever execute against live money, \
                 and the tools that could switch the account or environment are deliberately \
                 not exposed. Long jobs follow start/poll/stop: discovery_start or \
                 training_start, then poll engine_status. The backend must be running \
                 (desktop app or 'neoethos-app --server'); every tool fails loudly with an \
                 actionable message when it is not.",
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn done_marks_errors_and_passes_messages_verbatim() {
        let err = done(Err(ToolError("exact message".to_string())));
        assert_eq!(err.is_error, Some(true));
        let text = err.content[0].as_text().expect("text content").text.clone();
        assert_eq!(text, "exact message");

        let ok = done(Ok(serde_json::json!({"a": 1})));
        assert_eq!(ok.is_error, Some(false));
    }
}
