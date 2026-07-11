//! Live trading value-types extracted from the retired legacy egui
//! `trading/` module (v0.4.36 egui-modernization).
//!
//! The old `app_services::trading` module was the egui-era
//! `TradingSession` surface — a stateful session struct plus ~10
//! carved-out submodules (`session`, `orders`, `market_data`,
//! `risk_gate`, `auto_trade`, `auto_trade_producer`, `snapshots`,
//! `diagnostics`, `background`, `ensemble_predictor_adapter`). After the
//! Flutter migration the production HTTP server drives cTrader directly
//! through `broker_api` + `ctrader_execution` (see `broker_api.rs`:
//! *"without going through `TradingSession`"*), so `TradingSession` and
//! its entire submodule tree became dead code, exercised only by the
//! `--api-test` smoke harness and `trading_tests.rs`. Both were removed.
//!
//! These four plain value-types survived the removal because production
//! code still references them:
//!
//! - [`TradingAdapterKind`] — broker capability flags read by
//!   `broker_config` (and, in time, the Flutter order/position screens).
//! - [`MarketChartSnapshot`] — payload type of the live
//!   [`crate::app_services::ServiceEvent::ChartDataUpdated`] variant.
//! - [`ChartCandle`] / [`ChartOverlay`] — the candle + overlay rows that
//!   make up a [`MarketChartSnapshot`].
//!
//! They carry no behaviour beyond the small capability matrix on
//! `TradingAdapterKind`, so they live here as a leaf module with no
//! dependencies on the rest of `app_services`.

/// Which broker integration backs a trading adapter. Drives the
/// capability flags below. The cTrader Open API is the only wired
/// backend (the broker-agnostic direction is MCP bridges — see `mcp/`).
/// The capability-flag abstraction stays so the UI never grows
/// `== "cTrader"` checks that would lock out a future adapter.
// The variant is never *constructed* and `as_str` is never *called* in live
// code today — their only consumers are the broker-readiness banner helpers in
// `broker_config`, which are themselves Phase 2-5-pending. The type is still
// referenced in signatures and the capability-flag methods below stay live.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingAdapterKind {
    CTrader,
}

impl TradingAdapterKind {
    #[allow(dead_code)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CTrader => "cTrader",
        }
    }

    /// True if the broker adapter implements `Cancel Pending Order` /
    /// `Close Open Position` round-trips. Capability flags are the right
    /// abstraction here — the UI must never gate these on
    /// `adapter_name == "cTrader"`, which would permanently lock out any
    /// future adapter whose execution backend handled the operations.
    #[allow(dead_code)]
    pub fn supports_order_cancellation(self) -> bool {
        match self {
            Self::CTrader => true,
        }
    }

    /// True if the broker adapter implements `Close Open Position`.
    /// Separated from `supports_order_cancellation` because the two
    /// capabilities CAN diverge per broker (e.g. an adapter that
    /// supports cancelling resting orders but only flattens positions
    /// via a counter-trade rather than a dedicated Close call).
    #[allow(dead_code)]
    pub fn supports_position_close(self) -> bool {
        match self {
            Self::CTrader => true,
        }
    }
}

/// A single OHLCV candle in a [`MarketChartSnapshot`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChartCandle {
    pub timestamp: Option<i64>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// A marker drawn over a chart candle (e.g. a bot-decision fill).
#[derive(Debug, Clone, PartialEq)]
pub struct ChartOverlay {
    pub label: String,
    pub candle_index: usize,
    pub price: f64,
}

/// Snapshot of a symbol's chart state — candles, overlay markers and the
/// derived price/headline metadata. Payload of the live
/// [`crate::app_services::ServiceEvent::ChartDataUpdated`] variant.
#[derive(Debug, Clone, PartialEq)]
pub struct MarketChartSnapshot {
    pub symbol: String,
    pub timeframe: String,
    pub available_timeframes: Vec<String>,
    pub candles: Vec<ChartCandle>,
    pub overlays: Vec<ChartOverlay>,
    pub price_min: f64,
    pub price_max: f64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub price_change_pct: Option<f64>,
    pub headline: String,
    pub overlay_status: String,
    pub warnings: Vec<String>,
}
