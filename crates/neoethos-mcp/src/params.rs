//! Typed parameter structs for every MCP tool. Field docs become JSON-schema
//! descriptions (schemars), which is what Codex shows the model — keep them
//! precise. All names are snake_case on the MCP wire; the backend's
//! camelCase conversion happens inside `ops.rs`, never here.

use schemars::JsonSchema;
use serde::Deserialize;

/// "buy" | "sell" — matches the backend's `OrderSide` wire form exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum TradeSide {
    Buy,
    Sell,
}

impl TradeSide {
    pub fn as_wire(&self) -> &'static str {
        match self {
            TradeSide::Buy => "buy",
            TradeSide::Sell => "sell",
        }
    }
}

/// "limit" | "stop" — pending-order trigger semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum PendingOrderType {
    /// Fills at the trigger price or better.
    Limit,
    /// Fills once price trades through the trigger.
    Stop,
}

impl PendingOrderType {
    pub fn as_wire(&self) -> &'static str {
        match self {
            PendingOrderType::Limit => "limit",
            PendingOrderType::Stop => "stop",
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AccountSnapshotParams {
    /// When true, force a broker round-trip refresh before returning the
    /// snapshot (no trade side effects). Default: serve the cached snapshot.
    pub refresh: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OrderHistoryParams {
    /// Unix-millis inclusive lower bound. Default: 7 days before `to_ms`.
    pub from_ms: Option<i64>,
    /// Unix-millis upper bound. Default: now.
    pub to_ms: Option<i64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExpectedMarginParams {
    /// Symbol name, e.g. "EURUSD". Resolved to the broker symbol id via the
    /// symbol catalog.
    pub symbol: String,
    /// Volume in lots (1.0 = standard lot, 0.01 = micro lot).
    pub volume_lots: f64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GetChartParams {
    /// Symbol name, e.g. "EURUSD".
    pub symbol: String,
    /// Canonical timeframe label, e.g. "M5", "H1".
    pub timeframe: String,
    /// Trailing candles to return (default 500, max 2000).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChartHistoryParams {
    /// Symbol name, e.g. "EURUSD". Defaults to the backend's active chart symbol.
    pub symbol: Option<String>,
    /// Canonical timeframe label, e.g. "M5", "H1". Defaults to the active timeframe.
    pub timeframe: Option<String>,
    /// Cursor: return candles STRICTLY OLDER than this Unix-ms timestamp — the
    /// time of the oldest candle you currently hold. Page back by passing the
    /// oldest returned candle's time here.
    pub before_ms: i64,
    /// Candles to return in this page (backend default/cap apply).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GetIndicatorParams {
    /// Symbol name, e.g. "EURUSD".
    pub symbol: String,
    /// Canonical timeframe label, e.g. "M5", "H1".
    pub timeframe: String,
    /// One of: sma, ema, rsi, macd, bollinger_bands, atr, stoch, adx, vwap.
    pub indicator: String,
    /// Period for single-period indicators (sma/ema/rsi/atr/adx).
    pub period: Option<f64>,
    /// Bollinger Bands standard-deviation multiplier.
    pub std_dev: Option<f64>,
    /// MACD fast period (default 12).
    pub fast: Option<f64>,
    /// MACD slow period (default 26).
    pub slow: Option<f64>,
    /// MACD signal period (default 9).
    pub signal: Option<f64>,
    /// Stochastic %K period (default 14).
    pub k_period: Option<f64>,
    /// Stochastic %K smoothing (default 3).
    pub k_slow: Option<f64>,
    /// Stochastic %D period (default 3).
    pub d_period: Option<f64>,
    /// Trailing candles to compute over (default 200, max 2000).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NewsBriefingParams {
    /// When true, bypass the fetch-coalescing cache and pull fresh.
    pub force: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetWatchlistParams {
    /// The full replacement Market Watch set, e.g. ["EURUSD", "XAUUSD"].
    pub symbols: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenPositionParams {
    /// Symbol name, e.g. "EURUSD".
    pub symbol: String,
    /// "buy" or "sell".
    pub side: TradeSide,
    /// Volume in lots (1.0 = standard lot, 0.01 = micro lot).
    pub volume_lots: f64,
    /// REQUIRED stop-loss distance in pips from the fill price. This control
    /// plane never places a position without a stop loss (stricter than the
    /// backend, which allows naked orders with risky:true — a flag this
    /// server hardcodes to false and exposes no parameter for).
    pub stop_loss_pips: f64,
    /// Optional take-profit distance in pips from the fill price.
    pub take_profit_pips: Option<f64>,
    /// Optional free-form order comment (shows up in the broker journal).
    pub comment: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ClosePositionParams {
    /// The broker position id (from list_positions).
    pub position_id: i64,
    /// Volume to close, in lots. Omit to close the position entirely. The
    /// server converts lots to the broker's wire units from the position's
    /// current snapshot volume — never pass wire units here.
    pub volume_lots: Option<f64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModifyPositionProtectionParams {
    /// The broker position id (from list_positions).
    pub position_id: i64,
    /// New ABSOLUTE stop-loss price (not pips). Omit to leave unchanged.
    pub stop_loss_price: Option<f64>,
    /// New ABSOLUTE take-profit price (not pips). Omit to leave unchanged.
    pub take_profit_price: Option<f64>,
    /// Toggle the broker-side trailing-stop flag on the position's SL.
    pub trailing_stop: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlacePendingOrderParams {
    /// Symbol name, e.g. "EURUSD".
    pub symbol: String,
    /// "buy" or "sell".
    pub side: TradeSide,
    /// "limit" (fills at trigger or better) or "stop" (fills through trigger).
    pub order_type: PendingOrderType,
    /// Volume in lots.
    pub volume_lots: f64,
    /// Price at which the resting order becomes active.
    pub trigger_price: f64,
    /// REQUIRED stop-loss distance in pips from the eventual fill price.
    pub stop_loss_pips: f64,
    /// Optional take-profit distance in pips.
    pub take_profit_pips: Option<f64>,
    /// Optional Good-Till-Date expiry (Unix ms). Omitted = Good-Till-Cancel.
    pub expiry_unix_ms: Option<i64>,
    /// Optional free-form order comment.
    pub comment: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CancelPendingOrderParams {
    /// The resting order id (from list_pending_orders).
    pub order_id: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConfirmPendingActionParams {
    /// The queued action id (from list_pending_actions).
    pub action_id: String,
    /// Optional broker wire-unit volume override (partial close of a
    /// full-close proposal). Wire units, exactly as the queue stores them.
    pub volume_units_override: Option<i64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RejectPendingActionParams {
    /// The queued action id (from list_pending_actions).
    pub action_id: String,
    /// Optional free-form reason, persisted in the audit row.
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AutopilotStartParams {
    /// Portfolio artifact paths to trade (from list_portfolios). Each starts
    /// its own concurrent engine.
    pub portfolio_paths: Vec<String>,
    /// Lot size per trade for every started engine.
    pub lot_size: f64,
    /// Optional stop-loss pips override for every engine.
    pub stop_loss_pips: Option<f64>,
    /// Optional take-profit pips override for every engine.
    pub take_profit_pips: Option<f64>,
    /// Warmup bars before the engine starts signalling (backend default when
    /// omitted).
    pub warmup_bars: Option<usize>,
    /// Auto-cull: retire a strategy after this many consecutive losing
    /// trades (0 disables; backend default when omitted).
    pub cull_after_consecutive_losses: Option<u32>,
    /// Auto-cull: retire when the rolling win rate drops below this percent
    /// (0 disables; backend default when omitted).
    pub cull_min_win_rate_pct: Option<f64>,
    /// Rolling window (closed trades) for the win-rate cull.
    pub cull_window_trades: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DemoGateStatusParams {
    /// Portfolio artifact path (from list_portfolios).
    pub portfolio: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReplayBacktestParams {
    /// Symbol to replay. Omit to resolve from config.yaml like the CLI.
    pub symbol: Option<String>,
    /// Base timeframe. Omit to resolve from config.yaml like the CLI.
    pub base_tf: Option<String>,
    /// Path to a `*.live_portfolio.json` (from `list_portfolios`). Supplied →
    /// the REAL discovered genes are replayed, exactly as `trader-replay
    /// --portfolio` does. Omitted → a 3-bar momentum STUB that is not any
    /// strategy this product produced; the result says so in
    /// `fidelity_warnings`. Added 2026-08-10 (#229) — until then no front-end
    /// but the CLI could reach the real genes.
    pub portfolio_path: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ParityCheckParams {
    /// Portfolio artifact path (from list_portfolios).
    pub portfolio: String,
    /// Live-style window size (defaults to the live engine's warmup).
    pub window: Option<usize>,
    /// Long-history reference size (default 3000).
    pub reference: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TailRiskParams {
    /// Portfolio artifact path (from list_portfolios).
    pub portfolio: String,
    /// Monte-Carlo iterations (backend default 2000, clamped 200..20000).
    pub iterations: Option<usize>,
    /// Risk-fraction override (defaults to risk.risk_per_trade from config).
    pub risk: Option<f64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChallengeSimParams {
    /// Portfolio artifact path (from list_portfolios).
    pub portfolio: String,
    /// Monte-Carlo attempts per cell (backend default 2000).
    pub iterations: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryStartParams {
    /// Symbol to search. Omit to resolve from config.yaml like the CLI.
    pub symbol: Option<String>,
    /// Base timeframe. Omit to resolve from config.yaml like the CLI.
    pub base_tf: Option<String>,
    /// Higher-timeframe context ladder, e.g. ["M15","H1"]. Omit to use the
    /// operator-configured ladder.
    pub higher_tfs: Option<Vec<String>>,
    /// GA population override.
    pub population: Option<usize>,
    /// GA generations override.
    pub generations: Option<usize>,
    /// Max indicators per gene override.
    pub max_indicators: Option<usize>,
    /// Target candidate count override.
    pub target_candidates: Option<usize>,
    /// Portfolio size override.
    pub portfolio_size: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TrainingStartParams {
    /// Symbol to train. Omit to resolve from config.yaml like the CLI.
    pub symbol: Option<String>,
    /// Base timeframe. Omit to resolve from config.yaml like the CLI.
    pub base_tf: Option<String>,
    /// Higher-timeframe context (accepted for parity; training currently
    /// derives its own).
    pub higher_tfs: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PromoteStrategiesParams {
    /// Symbol. Omit to resolve from config.yaml like the desktop UI.
    pub symbol: Option<String>,
    /// Base timeframe. Omit to resolve from config.yaml.
    pub base_tf: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StrategyReportParams {
    /// Strategy cache dir (an `auto_loop*` name from list_strategies).
    pub dir: Option<String>,
    /// Report base name within the dir (from list_strategies).
    pub base: Option<String>,
}

/// Partial, TYPED mirror of the backend `POST /settings` DTO. No passthrough
/// for unknown keys — anything not listed here cannot be changed from this
/// control plane (raw config.yaml writes are deliberately not exposed).
#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UpdateSettingsParams {
    /// "risky" | "prop_firm".
    pub trading_mode: Option<String>,
    /// "auto" | "cpu" | "gpu".
    pub compute_mode: Option<String>,
    /// Risk fraction per trade (clamped server-side to the configured cap).
    pub risk_per_trade: Option<f64>,
    /// Portfolio-level cap on TOTAL concurrent risk (balance fraction;
    /// 0 disables; clamped to [0, 0.5]).
    pub max_portfolio_risk: Option<f64>,
    /// Discovery GA population.
    pub search_population: Option<usize>,
    /// Discovery GA generations.
    pub search_generations: Option<usize>,
    /// Discovery wall-clock cap in hours (0 = no cap).
    pub search_max_hours: Option<f64>,
    /// Max indicators per gene (0 = use all features).
    pub search_max_indicators: Option<usize>,
    /// Discovery portfolio size.
    pub search_portfolio_size: Option<usize>,
    /// Portfolio correlation threshold (0..1).
    pub search_corr_threshold: Option<f64>,
    /// Max data rows for the search (0 = full dataset).
    pub search_max_rows: Option<usize>,
    /// GA prefilter top-K.
    pub prefilter_top_k: Option<usize>,
    /// GA convergence patience (generations).
    pub convergence_patience: Option<usize>,
    /// GA stagnation patience (generations).
    pub stagnation_patience: Option<usize>,
    /// GA novelty weight (0..1).
    pub novelty_weight: Option<f64>,
    /// Disable the SMC gate in the search.
    pub disable_smc_gate: Option<bool>,
    /// LIVE ML gate: the trained ensemble scales per-trade risk on live
    /// entries (genes keep the direction).
    pub live_ml_gate: Option<bool>,
    /// Live blend floor (0..1): the smallest fraction of its size a gene entry
    /// may be shrunk to by a lukewarm ensemble. Must be >= `blend_veto_below`.
    pub blend_gate_floor: Option<f64>,
    /// Live blend veto (0..1): effective multiplier below which the bar is
    /// SKIPPED rather than sized to the floor.
    pub blend_veto_below: Option<f64>,
    /// Risky-Mode starting balance (account currency, > 0).
    pub risky_start_balance: Option<f64>,
    /// Risky-Mode target balance (account currency, > 0).
    pub risky_target_balance: Option<f64>,
    /// Risky-Mode horizon in days (> 0).
    pub risky_horizon_days: Option<u32>,
    /// Enable/disable the economic news calendar.
    pub news_calendar_enabled: Option<bool>,
    /// News calendar provider id (validated server-side).
    pub news_calendar_source: Option<String>,
    /// "block_on_news" | "allow_always" | "warn_only".
    pub news_trading_mode: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ApplyRiskPresetParams {
    /// One of: ftmo, myforexfunds, fundednext, the5ers, none.
    pub preset: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RiskyScenariosParams {
    /// Starting bankroll in USD (engine default when omitted).
    pub starting_usd: Option<f64>,
    /// Target bankroll in USD (engine default when omitted).
    pub target_usd: Option<f64>,
    /// Per-trade risk fraction (clamped to the engine's Risky band).
    pub risk_fraction: Option<f64>,
    /// Expected post-cost win rate in (0,1).
    pub win_rate: Option<f64>,
    /// Expected reward-to-risk per trade.
    pub reward_to_risk: Option<f64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JournalTradesParams {
    /// Unix-millis inclusive lower bound.
    pub from_ms: Option<i64>,
    /// Unix-millis upper bound.
    pub to_ms: Option<i64>,
    /// Max closed trades to return, most-recent first (default 500).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FetchHistoryParams {
    /// Symbol name, e.g. "EURUSD".
    pub symbol: String,
    /// Canonical timeframe label, e.g. "M5".
    pub timeframe: String,
    /// Unix-millis inclusive lower bound of the download window.
    pub from_ms: i64,
    /// Unix-millis exclusive upper bound. Omit = now.
    pub to_ms: Option<i64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportDataParams {
    /// Absolute path to the CSV/TSV/Parquet/JSON/JSONL/Arrow file to ingest.
    pub source_path: String,
    /// Symbol the data belongs to, e.g. "EURUSD".
    pub symbol: String,
    /// Timeframe of the bars, e.g. "M5".
    pub timeframe: String,
}
