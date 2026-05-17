//! DXtrade broker integration — trait surface + fail-loud stubs.
//!
//! Phase D3. Defines the production contract for DXtrade-broker
//! support so the Flutter API layer (and the existing
//! `TradingSession`) can route to a real DXtrade backend once the
//! subphases land. Today, all production stubs return
//! [`anyhow::Error`] with a clear "not yet implemented" message
//! naming the specific subphase that fills it in.
//!
//! ## Why DXtrade matters
//!
//! cTrader's Open API is excellent for OAuth-supported brokers but
//! a large slice of the retail forex market is on platforms that
//! either use DXtrade's own REST/WebSocket API or only expose
//! MT4/MT5 (Phase D3.5 — hidden behind a DXtrade-shaped facade
//! per the operator's 2026-05-18 directive: "the end user will
//! never see that is using mt, or mt5 but only our bot for
//! everything").
//!
//! ## Subphase roadmap
//!
//! - **D3.1** `DxTradeAuthBackend`: account-credentials login
//!   (username + password OR account-id + investor-password
//!   depending on broker), session-token refresh.
//! - **D3.2** `DxTradeOrderBackend`: market / limit / stop order
//!   submission, modify, cancel.
//! - **D3.3** `DxTradeStreamingBackend`: live quote tape +
//!   trendbar stream over WebSocket.
//! - **D3.4** `DxTradePositionBackend`: position list, history,
//!   close-by-id, modify-stop-loss / take-profit.
//! - **D3.5** MT4/MT5 facade — wraps the DXtrade trait set
//!   behind a translation layer that talks to an MT4/MT5 terminal
//!   (or a hidden Wine-hosted instance on Linux). The operator
//!   directive is explicit: the user never sees "MT4/5", only
//!   "our bot". Pure Rust per directive — no Python embedded.
//!
//! Each subphase ships as its own focused commit with full tests
//! against a captured-fixture transport, mirroring the cTrader
//! adapter's pattern.

use std::sync::Arc;

use anyhow::Result;

use crate::app_services::broker_config::DxTradeBrokerSettings;

// ---------------------------------------------------------------------------
// Auth (D3.1)
// ---------------------------------------------------------------------------

/// Auth handshake result — session token + expiry returned by
/// the DXtrade auth REST endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxTradeAuthSession {
    pub session_token: String,
    /// Unix seconds since epoch when the token expires.
    pub expires_at_unix: i64,
    /// Account ID the session is bound to.
    pub account_id: String,
}

/// Auth backend trait. Production implementation:
/// [`ProductionDxTradeAuthBackend`] (D3.1 stub today).
pub trait DxTradeAuthBackend: Send + Sync {
    /// Authenticate with the supplied broker config. Returns a
    /// usable session token.
    fn login(&self, settings: &DxTradeBrokerSettings) -> Result<DxTradeAuthSession>;

    /// Refresh a session before its `expires_at_unix`. Returns a
    /// new token + new expiry.
    fn refresh(
        &self,
        settings: &DxTradeBrokerSettings,
        previous: &DxTradeAuthSession,
    ) -> Result<DxTradeAuthSession>;
}

/// Production DXtrade auth backend.
///
/// CURRENT STATE: D3.1 stub. Returns "not yet implemented" loudly
/// so any call path that lands here surfaces the gap rather than
/// silently failing or returning fake data.
pub struct ProductionDxTradeAuthBackend;

impl DxTradeAuthBackend for ProductionDxTradeAuthBackend {
    fn login(&self, _settings: &DxTradeBrokerSettings) -> Result<DxTradeAuthSession> {
        anyhow::bail!(
            "DXtrade auth is not yet implemented (Phase D3.1 — REST handshake + \
             session-token caching). The trait surface exists so the wizard / \
             TradingSession can route here; the production fill lands in a \
             focused follow-up commit. Use the cTrader backend in the meantime."
        )
    }

    fn refresh(
        &self,
        _settings: &DxTradeBrokerSettings,
        _previous: &DxTradeAuthSession,
    ) -> Result<DxTradeAuthSession> {
        anyhow::bail!("DXtrade refresh is not yet implemented (Phase D3.1)")
    }
}

// ---------------------------------------------------------------------------
// Orders (D3.2)
// ---------------------------------------------------------------------------

/// Trade side for a DXtrade order. Mirrors the cTrader-side enum
/// so call sites can write generic broker-agnostic code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DxTradeOrderSide {
    Buy,
    Sell,
}

/// Order kind — market / limit / stop. DXtrade's REST surface
/// accepts these as discriminated variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DxTradeOrderKind {
    Market,
    Limit,
    Stop,
}

/// New-order request to DXtrade.
#[derive(Debug, Clone)]
pub struct DxTradeNewOrder {
    pub symbol: String,
    pub side: DxTradeOrderSide,
    pub kind: DxTradeOrderKind,
    /// Volume in broker-canonical units (typically lots × 100).
    pub volume: i64,
    /// Limit / stop price; `None` for market.
    pub price: Option<f64>,
    /// Optional stop-loss in price units.
    pub stop_loss_price: Option<f64>,
    /// Optional take-profit in price units.
    pub take_profit_price: Option<f64>,
}

/// Outcome of an order submission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DxTradeOrderStatus {
    Filled,
    PartialFill,
    Pending,
    Rejected { reason: String },
}

/// Result of submitting a new order.
#[derive(Debug, Clone)]
pub struct DxTradeOrderOutcome {
    pub broker_order_id: String,
    pub status: DxTradeOrderStatus,
    /// Filled price when `status` is Filled / PartialFill.
    pub fill_price: Option<f64>,
    /// Filled volume when `status` is Filled / PartialFill.
    pub fill_volume: Option<i64>,
}

/// Order execution backend trait. Production: [`ProductionDxTradeOrderBackend`].
pub trait DxTradeOrderBackend: Send + Sync {
    fn submit_order(
        &self,
        session: &DxTradeAuthSession,
        order: &DxTradeNewOrder,
    ) -> Result<DxTradeOrderOutcome>;
    fn cancel_order(&self, session: &DxTradeAuthSession, broker_order_id: &str) -> Result<()>;
}

/// Production DXtrade order backend — D3.2 stub.
pub struct ProductionDxTradeOrderBackend;

impl DxTradeOrderBackend for ProductionDxTradeOrderBackend {
    fn submit_order(
        &self,
        _session: &DxTradeAuthSession,
        _order: &DxTradeNewOrder,
    ) -> Result<DxTradeOrderOutcome> {
        anyhow::bail!(
            "DXtrade order submission is not yet implemented (Phase D3.2 — \
             REST POST /orders with bracket SL/TP). Trait surface exists for \
             routing tests; production fill follows."
        )
    }

    fn cancel_order(
        &self,
        _session: &DxTradeAuthSession,
        _broker_order_id: &str,
    ) -> Result<()> {
        anyhow::bail!("DXtrade cancel_order is not yet implemented (Phase D3.2)")
    }
}

// ---------------------------------------------------------------------------
// Streaming (D3.3)
// ---------------------------------------------------------------------------

/// Live tick / trendbar update from DXtrade's WebSocket feed.
#[derive(Debug, Clone, PartialEq)]
pub struct DxTradeLiveUpdate {
    pub symbol: String,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub timestamp_ms: Option<i64>,
    /// When the broker promotes a tick into a closed bar, the
    /// latest_trendbar slot carries the OHLCV. Mirrors the cTrader
    /// `CTraderLiveChartUpdate.latest_trendbar` field so the
    /// producer's adapter can be broker-agnostic.
    pub latest_trendbar: Option<crate::app_services::ctrader_data::HistoricalBar>,
}

/// Streaming backend trait. Production: [`ProductionDxTradeStreamingBackend`].
pub trait DxTradeStreamingBackend: Send + Sync {
    fn subscribe_live_chart(
        &self,
        session: &DxTradeAuthSession,
        symbol: &str,
        timeframe: &str,
    ) -> Result<DxTradeLiveUpdate>;
}

/// Production DXtrade streaming backend — D3.3 stub.
pub struct ProductionDxTradeStreamingBackend;

impl DxTradeStreamingBackend for ProductionDxTradeStreamingBackend {
    fn subscribe_live_chart(
        &self,
        _session: &DxTradeAuthSession,
        _symbol: &str,
        _timeframe: &str,
    ) -> Result<DxTradeLiveUpdate> {
        anyhow::bail!(
            "DXtrade streaming is not yet implemented (Phase D3.3 — WebSocket \
             subscribe + heartbeat + reconnect). Trait surface exists; production \
             fill follows."
        )
    }
}

// ---------------------------------------------------------------------------
// Bundle that composes the three trait objects
// ---------------------------------------------------------------------------

/// One-stop DXtrade backend handle — wraps the three trait objects
/// the TradingSession needs. Constructed via
/// [`Self::production`] (current stubs) or via test helpers that
/// inject fakes per-trait.
pub struct DxTradeBackend {
    pub auth: Arc<dyn DxTradeAuthBackend>,
    pub orders: Arc<dyn DxTradeOrderBackend>,
    pub streaming: Arc<dyn DxTradeStreamingBackend>,
}

impl DxTradeBackend {
    /// Production-grade backend with the three stub
    /// implementations. Until D3.1/D3.2/D3.3 land, every call
    /// path through these traits fails loud — the operator can
    /// SAFELY route through this in production code because the
    /// failure surfaces cleanly via anyhow + the operator's
    /// chrome banner can render the "not yet implemented" status.
    pub fn production() -> Self {
        Self {
            auth: Arc::new(ProductionDxTradeAuthBackend),
            orders: Arc::new(ProductionDxTradeOrderBackend),
            streaming: Arc::new(ProductionDxTradeStreamingBackend),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_settings() -> DxTradeBrokerSettings {
        DxTradeBrokerSettings::default()
    }

    fn dummy_session() -> DxTradeAuthSession {
        DxTradeAuthSession {
            session_token: "token".to_string(),
            expires_at_unix: 0,
            account_id: "acc-1".to_string(),
        }
    }

    // -- Each stub fails loud rather than silently returning empty data --

    #[test]
    fn production_auth_login_fails_loud_with_phase_tag() {
        let backend = ProductionDxTradeAuthBackend;
        let err = backend.login(&empty_settings()).expect_err("must bail");
        let msg = err.to_string();
        assert!(msg.contains("Phase D3.1"), "missing phase tag: {msg}");
        assert!(msg.contains("DXtrade"));
    }

    #[test]
    fn production_auth_refresh_fails_loud() {
        let backend = ProductionDxTradeAuthBackend;
        let err = backend
            .refresh(&empty_settings(), &dummy_session())
            .expect_err("must bail");
        assert!(err.to_string().contains("Phase D3.1"));
    }

    #[test]
    fn production_orders_submit_fails_loud_with_phase_tag() {
        let backend = ProductionDxTradeOrderBackend;
        let order = DxTradeNewOrder {
            symbol: "EURUSD".to_string(),
            side: DxTradeOrderSide::Buy,
            kind: DxTradeOrderKind::Market,
            volume: 100,
            price: None,
            stop_loss_price: None,
            take_profit_price: None,
        };
        let err = backend
            .submit_order(&dummy_session(), &order)
            .expect_err("must bail");
        assert!(err.to_string().contains("Phase D3.2"));
    }

    #[test]
    fn production_orders_cancel_fails_loud() {
        let backend = ProductionDxTradeOrderBackend;
        let err = backend
            .cancel_order(&dummy_session(), "order-1")
            .expect_err("must bail");
        assert!(err.to_string().contains("Phase D3.2"));
    }

    #[test]
    fn production_streaming_subscribe_fails_loud_with_phase_tag() {
        let backend = ProductionDxTradeStreamingBackend;
        let err = backend
            .subscribe_live_chart(&dummy_session(), "EURUSD", "H1")
            .expect_err("must bail");
        assert!(err.to_string().contains("Phase D3.3"));
    }

    #[test]
    fn dxtrade_backend_production_bundle_constructs_cleanly() {
        // Construction must succeed — every call path through it
        // is what fails. The chrome can build the backend at
        // startup without errors; only when the operator actually
        // routes a DXtrade order does the not-yet-implemented
        // message surface.
        let _ = DxTradeBackend::production();
    }

    #[test]
    fn order_status_variants_round_trip() {
        // Pin the enum shape — the Flutter API layer's order-status
        // response codes depend on these variants existing.
        assert_eq!(
            DxTradeOrderStatus::Rejected {
                reason: "test".to_string()
            },
            DxTradeOrderStatus::Rejected {
                reason: "test".to_string()
            }
        );
        assert_ne!(DxTradeOrderStatus::Filled, DxTradeOrderStatus::Pending);
    }

    #[test]
    fn order_side_and_kind_have_distinct_values() {
        assert_ne!(DxTradeOrderSide::Buy, DxTradeOrderSide::Sell);
        assert_ne!(DxTradeOrderKind::Market, DxTradeOrderKind::Limit);
        assert_ne!(DxTradeOrderKind::Limit, DxTradeOrderKind::Stop);
    }
}
