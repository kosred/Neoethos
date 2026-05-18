//! DXtrade broker integration — REST + WebSocket adapter.
//!
//! Phase D3. D3.1 (auth REST handshake) is now wired against the
//! OFFICIAL DXtrade Developer Portal spec — `POST /dxsca-web/login`
//! returning `{sessionToken, timeout}`, with subsequent calls
//! authenticated via `Authorization: DXAPI <token>`. See
//! <https://demo.dx.trade/developers/#/DXtrade-REST-API> ("Create
//! Session Token" + "Authentication & Authorization" sections).
//!
//! D3.2 (orders) and D3.3 (streaming via the DXtrade Push API)
//! remain fail-loud stubs that name their subphase; the trait
//! surface is stable so call sites (TradingSession, Flutter API
//! layer, wizard) can route through `DxTradeBackend::production()`
//! today and only surface "not yet implemented" when the operator
//! actually invokes those paths.
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
//! - **D3.1 (DONE)** `DxTradeAuthBackend`: account-credentials
//!   login (`username` + `domain` + `password`) via
//!   `POST /dxsca-web/login`. Session-token caching is handled by
//!   the calling layer using the returned `expires_at_unix`;
//!   `refresh()` re-logs in (DXtrade has no separate refresh
//!   endpoint — the Ping API extends but does not mint a new
//!   token).
//! - **D3.2 (DONE)** `DxTradeOrderBackend`: market / limit / stop
//!   order submission, modify, cancel via
//!   `POST/PUT/DELETE /dxsca-web/accounts/{account}/orders`.
//!   Bracket SL/TP is encoded as an IF-THEN Order Group with the
//!   parent at `positionEffect = "OPEN"` and the protection legs
//!   at `positionEffect = "CLOSE"` (`quantity = "0"` on the
//!   children flags them as inherit-from-parent — per the DXtrade
//!   "Adding protections" Example 1 in the developer portal).
//!   `orderCode` is generated client-side from 128 bits of
//!   `OsRng` entropy and is what `submit_order` returns as
//!   `broker_order_id`; that's what `cancel_order` /
//!   `modify_order` accept.
//! - **D3.3 (DONE — single-shot)** `DxTradeStreamingBackend`:
//!   live quote tape over the DXtrade Push API WebSocket. The
//!   implementation opens a WS to `{platform}/dxsca-web/md` with
//!   `?format=JSON`, sends a `MarketDataSubscriptionRequest` for
//!   the symbol with a `Quote` event-type, reads frames until a
//!   matching Quote arrives, parses it into [`DxTradeLiveUpdate`]
//!   and closes the socket. The existing trait signature returns
//!   a single snapshot — the WIRE-FORMAT layer (envelopes,
//!   subscription request, parsing) is reusable and properly
//!   tested via an injectable [`DxTradeWebSocketFactory`]. A
//!   future D3.3.1 enhancement replaces single-shot with a
//!   stream-of-updates handle backed by a `crossbeam-channel`
//!   receiver, matching the cTrader streaming worker pattern.
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
use std::time::Duration as StdDuration;

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::app_services::broker_config::DxTradeBrokerSettings;

// ---------------------------------------------------------------------------
// HTTP transport abstraction (D3.1)
// ---------------------------------------------------------------------------

/// Outcome of an HTTP request — status + body. Tests inject a
/// fake transport that returns canned `DxTradeHttpResponse`s;
/// production uses [`ReqwestDxTradeTransport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxTradeHttpResponse {
    pub status: u16,
    pub body: String,
}

/// Sync HTTP transport trait used by the DXtrade adapter. Sync
/// (not async) because [`DxTradeAuthBackend`] is a sync trait —
/// the calling TradingSession owns its own runtime and dispatches
/// these calls on a worker thread.
///
/// `bearer_token` is the raw session token (no scheme prefix);
/// the transport adds `Authorization: DXAPI <token>` per the
/// official spec.
pub trait DxTradeHttpTransport: Send + Sync {
    fn post_json(
        &self,
        url: &str,
        bearer_token: Option<&str>,
        body: &str,
    ) -> Result<DxTradeHttpResponse>;

    /// PUT a JSON body — used by D3.2 Modify Order. Always
    /// authenticated, so the bearer token is required.
    fn put_json(&self, url: &str, bearer_token: &str, body: &str) -> Result<DxTradeHttpResponse>;

    /// DELETE with no body — used by D3.2 Cancel Order. Always
    /// authenticated.
    fn delete(&self, url: &str, bearer_token: &str) -> Result<DxTradeHttpResponse>;
}

/// Production transport — wraps `reqwest::blocking::Client`.
pub struct ReqwestDxTradeTransport {
    client: Client,
}

impl ReqwestDxTradeTransport {
    /// Build the transport. Panics on TLS / builder failure with
    /// a clear message — that path only fires if the process-wide
    /// rustls CryptoProvider is misconfigured, in which case the
    /// cTrader adapter would equally fail and the bot has no
    /// route to a broker anyway. Keeping construction infallible
    /// lets `DxTradeBackend::production()` stay panic-free at
    /// chrome startup.
    pub fn new() -> Self {
        // Install the process-wide rustls CryptoProvider before
        // reqwest builds its TLS config. Required because
        // rustls 0.23 panics at runtime when both `ring` and
        // `aws-lc-rs` providers are visible in the feature graph
        // (see workspace Cargo.toml comment on the direct rustls
        // edge). `ensure_ctrader_rustls_provider` is idempotent
        // via `std::sync::Once`, so it's safe to call from every
        // adapter that opens a TLS connection.
        crate::app_services::ctrader_tls::ensure_ctrader_rustls_provider();
        let client = Client::builder()
            .timeout(StdDuration::from_secs(30))
            .user_agent(dxtrade_user_agent())
            .build()
            .expect(
                "DXtrade reqwest::blocking::Client::build failed — \
                 rustls CryptoProvider misconfigured?",
            );
        Self { client }
    }
}

impl Default for ReqwestDxTradeTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl DxTradeHttpTransport for ReqwestDxTradeTransport {
    fn post_json(
        &self,
        url: &str,
        bearer_token: Option<&str>,
        body: &str,
    ) -> Result<DxTradeHttpResponse> {
        let mut req = self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .body(body.to_string());
        if let Some(token) = bearer_token {
            req = req.header("Authorization", format!("DXAPI {token}"));
        }
        let resp = req.send().context("DXtrade POST send failed")?;
        let status = resp.status().as_u16();
        let body = resp
            .text()
            .context("DXtrade POST response body read failed")?;
        Ok(DxTradeHttpResponse { status, body })
    }

    fn put_json(&self, url: &str, bearer_token: &str, body: &str) -> Result<DxTradeHttpResponse> {
        let resp = self
            .client
            .put(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("Authorization", format!("DXAPI {bearer_token}"))
            .body(body.to_string())
            .send()
            .context("DXtrade PUT send failed")?;
        let status = resp.status().as_u16();
        let body = resp
            .text()
            .context("DXtrade PUT response body read failed")?;
        Ok(DxTradeHttpResponse { status, body })
    }

    fn delete(&self, url: &str, bearer_token: &str) -> Result<DxTradeHttpResponse> {
        let resp = self
            .client
            .delete(url)
            .header("Accept", "application/json")
            .header("Authorization", format!("DXAPI {bearer_token}"))
            .send()
            .context("DXtrade DELETE send failed")?;
        let status = resp.status().as_u16();
        let body = resp
            .text()
            .context("DXtrade DELETE response body read failed")?;
        Ok(DxTradeHttpResponse { status, body })
    }
}

fn dxtrade_user_agent() -> String {
    format!("forex-ai/{}", env!("CARGO_PKG_VERSION"))
}

/// Fallback session TTL when the server's `timeout` string is
/// unparseable. Conservative — refresh will re-login earlier
/// rather than letting an old token hit a 401 mid-trade.
pub const DXTRADE_DEFAULT_SESSION_TTL_SECONDS: i64 = 60 * 30;

// ---------------------------------------------------------------------------
// Auth (D3.1)
// ---------------------------------------------------------------------------

/// Auth handshake result — session token + expiry returned by
/// the DXtrade `POST /dxsca-web/login` endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxTradeAuthSession {
    /// The `sessionToken` field from the login response. Used as
    /// `Authorization: DXAPI <session_token>` on every subsequent
    /// request.
    pub session_token: String,
    /// Unix seconds since epoch when the token expires. Computed
    /// from the server's `timeout` string at login time (see
    /// [`parse_timeout_seconds`]); falls back to
    /// [`DXTRADE_DEFAULT_SESSION_TTL_SECONDS`] when unparseable.
    pub expires_at_unix: i64,
    /// Account code chosen for execution. Picked from the broker
    /// settings (first enabled, or first overall) — the official
    /// `/login` endpoint does NOT take an account; account lives
    /// in the URL path for trading endpoints.
    pub account_id: String,
    /// Platform base URL the token was issued against, with any
    /// trailing slash stripped. Carried on the session so D3.2
    /// order endpoints can construct URLs without consulting
    /// `DxTradeBrokerSettings` again — the auth layer is the
    /// single source of truth for "which host am I talking to".
    pub platform_url: String,
}

/// Auth backend trait. Production implementation:
/// [`ProductionDxTradeAuthBackend`].
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

/// JSON body for `POST /dxsca-web/login`. Field names mirror the
/// official spec exactly (lowercase `username` / `domain` /
/// `password`).
#[derive(Serialize)]
struct DxTradeLoginRequestBody<'a> {
    username: &'a str,
    domain: &'a str,
    password: &'a str,
}

/// Response shape from `POST /dxsca-web/login`:
/// `{"sessionToken": "...", "timeout": "..."}`.
#[derive(Deserialize)]
struct DxTradeLoginResponseBody {
    #[serde(rename = "sessionToken")]
    session_token: String,
    #[serde(default)]
    timeout: String,
}

/// Production DXtrade auth backend.
pub struct ProductionDxTradeAuthBackend {
    transport: Arc<dyn DxTradeHttpTransport>,
}

impl ProductionDxTradeAuthBackend {
    /// Construct with the real reqwest-based transport. Used by
    /// [`DxTradeBackend::production`].
    pub fn new() -> Self {
        Self {
            transport: Arc::new(ReqwestDxTradeTransport::new()),
        }
    }

    /// Construct with a caller-supplied transport. Used by tests
    /// to inject canned HTTP responses without touching the
    /// network.
    pub fn with_transport(transport: Arc<dyn DxTradeHttpTransport>) -> Self {
        Self { transport }
    }
}

impl Default for ProductionDxTradeAuthBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DxTradeAuthBackend for ProductionDxTradeAuthBackend {
    fn login(&self, settings: &DxTradeBrokerSettings) -> Result<DxTradeAuthSession> {
        let platform = validate_platform_url(&settings.platform_url)?;
        if settings.username.trim().is_empty() {
            anyhow::bail!("DXtrade login: username is empty");
        }
        if settings.domain.trim().is_empty() {
            anyhow::bail!(
                "DXtrade login: domain is empty \
                 (required by POST /dxsca-web/login per the official spec)"
            );
        }
        if settings.password.is_empty() {
            anyhow::bail!("DXtrade login: password is empty");
        }

        let url = format!("{platform}/dxsca-web/login");
        let body = DxTradeLoginRequestBody {
            username: settings.username.as_str(),
            domain: settings.domain.as_str(),
            password: settings.password.as_str(),
        };
        let body_json =
            serde_json::to_string(&body).context("DXtrade: failed to serialize login body")?;

        let resp = self
            .transport
            .post_json(&url, None, &body_json)
            .context("DXtrade /login transport failed")?;

        if !(200..300).contains(&resp.status) {
            anyhow::bail!(
                "DXtrade /login returned HTTP {} from {url}: {}",
                resp.status,
                truncate_for_log(&resp.body)
            );
        }

        let parsed: DxTradeLoginResponseBody =
            serde_json::from_str(&resp.body).with_context(|| {
                format!(
                    "DXtrade /login: failed to parse response body: {}",
                    truncate_for_log(&resp.body)
                )
            })?;

        if parsed.session_token.trim().is_empty() {
            anyhow::bail!("DXtrade /login: server returned empty sessionToken");
        }

        let ttl_seconds =
            parse_timeout_seconds(&parsed.timeout).unwrap_or(DXTRADE_DEFAULT_SESSION_TTL_SECONDS);
        let expires_at_unix = current_unix_seconds().saturating_add(ttl_seconds);

        Ok(DxTradeAuthSession {
            session_token: parsed.session_token,
            expires_at_unix,
            account_id: pick_default_account_id(settings),
            platform_url: platform,
        })
    }

    fn refresh(
        &self,
        settings: &DxTradeBrokerSettings,
        _previous: &DxTradeAuthSession,
    ) -> Result<DxTradeAuthSession> {
        // DXtrade does NOT expose a refresh-token endpoint. The
        // Ping API (`POST /dxsca-web/ping`) extends the existing
        // token's idle timer but does not mint a new one — it
        // returns 200 OK with no body, and the *same* token stays
        // valid. So "refresh" here means: throw away the old
        // token, re-issue credentials, get a brand-new token with
        // a fresh `timeout` window. A follow-up enhancement can
        // try Ping first and fall back to /login on 401 to avoid
        // unnecessary credential round-trips when the token's
        // still alive.
        self.login(settings)
    }
}

/// Validate and normalize a DXtrade `platform_url`. Returns the
/// scheme-bearing host string with any trailing `/` stripped, so
/// `format!("{platform}/dxsca-web/login")` always produces a
/// canonical URL.
fn validate_platform_url(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        anyhow::bail!("DXtrade platform_url is empty");
    }
    if !(trimmed.starts_with("https://") || trimmed.starts_with("http://")) {
        anyhow::bail!("DXtrade platform_url must include scheme (https://...): got {trimmed}");
    }
    Ok(trimmed.trim_end_matches('/').to_string())
}

/// Pick a default account_id from the broker settings. Preference
/// order: (1) first account flagged for execution, (2) first
/// configured account, (3) empty string when no accounts are
/// configured yet (the operator will pick one in the wizard).
fn pick_default_account_id(settings: &DxTradeBrokerSettings) -> String {
    settings
        .accounts
        .iter()
        .find(|a| a.enabled_for_execution)
        .or_else(|| settings.accounts.first())
        .map(|a| a.account_id.clone())
        .unwrap_or_default()
}

/// Parse the `timeout` field returned by `/dxsca-web/login`. The
/// spec types it as `"string"` and brokers in practice use one of
/// three shapes:
/// * ISO 8601 duration: `PT30M`, `PT1H`, `PT15M30S`
/// * Plain integer seconds: `"1800"`
/// * Plain integer milliseconds: `"1800000"`
///
/// Returns `None` for unknown shapes; callers fall back to
/// [`DXTRADE_DEFAULT_SESSION_TTL_SECONDS`]. The heuristic for
/// distinguishing seconds vs millis: anything larger than a
/// week's worth of seconds (`7 * 24 * 3600`) is assumed to be
/// millis, which is safe because no real DXtrade session lasts
/// that long.
pub fn parse_timeout_seconds(raw: &str) -> Option<i64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    // ISO 8601 duration — only the subset DXtrade emits in the
    // wild: `PT[H][M][S]`. Anything more exotic falls through.
    if let Some(rest) = trimmed.strip_prefix("PT") {
        if rest.is_empty() {
            return None;
        }
        let mut total: i64 = 0;
        let mut digits = String::new();
        let mut consumed_any_unit = false;
        for ch in rest.chars() {
            if ch.is_ascii_digit() {
                digits.push(ch);
                continue;
            }
            let n: i64 = digits.parse().ok()?;
            digits.clear();
            let inc = match ch {
                'H' => n.checked_mul(3600)?,
                'M' => n.checked_mul(60)?,
                'S' => n,
                _ => return None,
            };
            total = total.checked_add(inc)?;
            consumed_any_unit = true;
        }
        if !digits.is_empty() {
            return None; // trailing digits without a unit suffix
        }
        return if consumed_any_unit && total > 0 {
            Some(total)
        } else {
            None
        };
    }

    if let Ok(n) = trimmed.parse::<i64>() {
        if n <= 0 {
            return None;
        }
        // Anything bigger than a week (in seconds) is assumed to
        // be milliseconds — a real DXtrade session shorter than a
        // week is the universal case.
        if n > 7 * 24 * 3600 {
            return Some(n / 1000);
        }
        return Some(n);
    }

    None
}

fn current_unix_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn truncate_for_log(s: &str) -> String {
    const MAX: usize = 256;
    if s.len() > MAX {
        format!("{}…(truncated)", &s[..MAX])
    } else {
        s.to_string()
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

impl DxTradeOrderSide {
    /// Wire-format string per the DXtrade Single Order Request
    /// `side` field.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::Buy => "BUY",
            Self::Sell => "SELL",
        }
    }
}

/// Order kind — market / limit / stop. The official DXtrade REST
/// API uses `"type": "MARKET" | "LIMIT" | "STOP"` in the Place
/// Order body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DxTradeOrderKind {
    Market,
    Limit,
    Stop,
}

impl DxTradeOrderKind {
    /// Wire-format string per the DXtrade Single Order Request
    /// `type` field.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::Market => "MARKET",
            Self::Limit => "LIMIT",
            Self::Stop => "STOP",
        }
    }
}

/// New-order request to DXtrade.
#[derive(Debug, Clone)]
pub struct DxTradeNewOrder {
    /// DXtrade `instrument` field — symbol in slash format, e.g.
    /// `"EUR/USD"`. The caller is responsible for normalizing
    /// internal symbol strings to this shape; see
    /// `forex_core::SymbolMetadata`.
    pub symbol: String,
    pub side: DxTradeOrderSide,
    pub kind: DxTradeOrderKind,
    /// Volume in broker-canonical units (DXtrade quantity — NOT
    /// lots). The DXtrade Single Order Request schema describes
    /// this field as "Initial quantity of the order in units (not
    /// in lots)" — for forex this is typically lot_size × 100_000.
    pub volume: i64,
    /// Limit / stop price; `None` for market. Mapped to
    /// `limitPrice` when `kind == Limit` and `stopPrice` when
    /// `kind == Stop`.
    pub price: Option<f64>,
    /// Optional stop-loss in price units. When set, places a
    /// contingent STOP child order in an IF-THEN Order Group.
    pub stop_loss_price: Option<f64>,
    /// Optional take-profit in price units. When set, places a
    /// contingent LIMIT child order in an IF-THEN Order Group.
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
    /// The DXtrade `orderCode` we generated client-side. This is
    /// what [`DxTradeOrderBackend::cancel_order`] /
    /// [`DxTradeOrderBackend::modify_order`] expect back, NOT the
    /// server-side `orderId`. Per the official spec, `orderCode`
    /// is the canonical identifier in the cancel/modify URL
    /// paths.
    pub broker_order_id: String,
    pub status: DxTradeOrderStatus,
    /// Filled price when `status` is Filled / PartialFill.
    pub fill_price: Option<f64>,
    /// Filled volume when `status` is Filled / PartialFill.
    pub fill_volume: Option<i64>,
    /// Server-side numeric order id from the DXtrade Order
    /// Response (`orderId` field). Informational — useful for
    /// joining against history endpoints and WebSocket order
    /// events. None if the response was empty (e.g. 204 No
    /// Content from some broker tenants).
    pub server_order_id: Option<i64>,
}

/// Order execution backend trait. Production:
/// [`ProductionDxTradeOrderBackend`].
pub trait DxTradeOrderBackend: Send + Sync {
    /// Place a new order — Single Order if no SL/TP, IF-THEN
    /// Order Group otherwise. Returns the client-generated
    /// `orderCode` as `broker_order_id`.
    fn submit_order(
        &self,
        session: &DxTradeAuthSession,
        order: &DxTradeNewOrder,
    ) -> Result<DxTradeOrderOutcome>;

    /// Modify an existing order by `orderCode`. Per the spec,
    /// only conditional requests are accepted and some fields
    /// (e.g. `instrument`, `side`) are immutable — attempting to
    /// change them returns `409 Conflict`. The caller passes a
    /// fully-populated [`DxTradeNewOrder`] reflecting the desired
    /// final state.
    fn modify_order(
        &self,
        session: &DxTradeAuthSession,
        order_code: &str,
        order: &DxTradeNewOrder,
    ) -> Result<DxTradeOrderOutcome>;

    /// Cancel a single order by `orderCode`. The order must not
    /// be in a final status (COMPLETED, CANCELED, EXPIRED,
    /// REJECTED) — if it is, the server returns `409 Conflict`
    /// with error code `1005` "Reference order is closed".
    fn cancel_order(&self, session: &DxTradeAuthSession, broker_order_id: &str) -> Result<()>;
}

// -- Wire types ---------------------------------------------------------------

/// Single Order Request body — mirrors the DXtrade REST API
/// schema. All numeric fields are serialized as strings to match
/// the examples in the developer portal (avoids f64 precision
/// drift and quotes-stripping bugs in some broker JSON parsers).
#[derive(Serialize)]
struct SingleOrderRequest<'a> {
    #[serde(rename = "orderCode")]
    order_code: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    instrument: &'a str,
    quantity: String,
    side: &'static str,
    tif: &'static str,
    #[serde(rename = "positionEffect", skip_serializing_if = "Option::is_none")]
    position_effect: Option<&'static str>,
    #[serde(rename = "limitPrice", skip_serializing_if = "Option::is_none")]
    limit_price: Option<String>,
    #[serde(rename = "stopPrice", skip_serializing_if = "Option::is_none")]
    stop_price: Option<String>,
}

/// Order Group Request body — used when SL and/or TP are set.
/// First entry is the parent (OPEN), remaining entries are
/// contingent protections (CLOSE) with `quantity = "0"`
/// (inherit-from-parent) per the developer portal's "Adding
/// protections" example.
#[derive(Serialize)]
struct OrderGroupRequest<'a> {
    orders: Vec<SingleOrderRequest<'a>>,
}

/// Order Response — what the server returns from Place/Modify.
#[derive(Deserialize)]
struct OrderResponse {
    #[serde(rename = "orderId", default)]
    order_id: Option<i64>,
}

/// Default Time-In-Force. The spec doesn't enumerate accepted
/// values exhaustively, but the developer portal examples
/// uniformly use `"GTC"` (Good Till Cancelled) — matches the
/// bot's autonomous-trading model where orders persist until the
/// strategy explicitly cancels them or the bracket closes.
const DXTRADE_DEFAULT_TIF: &str = "GTC";

/// Format a price for the DXtrade wire format. We use enough
/// precision to round-trip 5-decimal forex pip pricing without
/// drift but cap at 8 decimals to avoid trailing noise. Trailing
/// zeros after the decimal point are stripped.
fn format_price(price: f64) -> String {
    let raw = format!("{price:.8}");
    let trimmed = raw.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Generate a client-unique order code from 128 bits of OS
/// entropy. Format: `forex-ai-<32-hex>`. The DXtrade spec
/// requires `orderCode` to be unique per account; 128 bits of
/// entropy is effectively collision-free for any realistic
/// trading volume.
fn generate_order_code() -> String {
    use rand::TryRngCore;
    let mut bytes = [0u8; 16];
    // OsRng failure is exceedingly rare (entropy pool exhaustion
    // on some embedded Linux variants under heavy load). Fall
    // back to a timestamped pseudo-id so the bot doesn't hard-
    // fail an order submission for an unrelated reason — the
    // upstream layer can still reject duplicates via the 409
    // path if a collision ever did occur.
    if rand::rngs::OsRng.try_fill_bytes(&mut bytes).is_err() {
        let now = current_unix_seconds() as u128;
        let hi = (now >> 64) as u64;
        let lo = (now & 0xFFFF_FFFF_FFFF_FFFF) as u64;
        return format!("forex-ai-fallback-{hi:016x}{lo:016x}");
    }
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("forex-ai-{hex}")
}

/// Build a [`SingleOrderRequest`] from internal types. Splits
/// `price` into either `limitPrice` (for LIMIT) or `stopPrice`
/// (for STOP); MARKET orders omit both.
fn build_single_request<'a>(
    order_code: &'a str,
    order: &'a DxTradeNewOrder,
    position_effect: Option<&'static str>,
    quantity_override: Option<&str>,
) -> SingleOrderRequest<'a> {
    let (limit_price, stop_price) = match order.kind {
        DxTradeOrderKind::Limit => (order.price.map(format_price), None),
        DxTradeOrderKind::Stop => (None, order.price.map(format_price)),
        DxTradeOrderKind::Market => (None, None),
    };
    let quantity = match quantity_override {
        Some(q) => q.to_string(),
        None => order.volume.to_string(),
    };
    SingleOrderRequest {
        order_code,
        kind: order.kind.as_wire_str(),
        instrument: order.symbol.as_str(),
        quantity,
        side: order.side.as_wire_str(),
        tif: DXTRADE_DEFAULT_TIF,
        position_effect,
        limit_price,
        stop_price,
    }
}

/// Opposite side for protection legs in an IF-THEN group.
fn opposite_side(side: DxTradeOrderSide) -> DxTradeOrderSide {
    match side {
        DxTradeOrderSide::Buy => DxTradeOrderSide::Sell,
        DxTradeOrderSide::Sell => DxTradeOrderSide::Buy,
    }
}

/// Build a protection child order for an IF-THEN group. Children
/// always use `quantity = "0"` (inherit from parent),
/// `positionEffect = "CLOSE"`, the opposite side from the
/// parent, and either STOP (stop-loss) or LIMIT (take-profit).
fn build_protection_child<'a>(
    order_code: &'a str,
    parent: &'a DxTradeNewOrder,
    kind: DxTradeOrderKind,
    price: f64,
    symbol_holder: &'a str,
) -> SingleOrderRequest<'a> {
    let (limit_price, stop_price) = match kind {
        DxTradeOrderKind::Limit => (Some(format_price(price)), None),
        DxTradeOrderKind::Stop => (None, Some(format_price(price))),
        DxTradeOrderKind::Market => (None, None),
    };
    SingleOrderRequest {
        order_code,
        kind: kind.as_wire_str(),
        instrument: symbol_holder,
        quantity: "0".to_string(),
        side: opposite_side(parent.side).as_wire_str(),
        tif: DXTRADE_DEFAULT_TIF,
        position_effect: Some("CLOSE"),
        limit_price,
        stop_price,
    }
}

/// Production DXtrade order backend — D3.2.
pub struct ProductionDxTradeOrderBackend {
    transport: Arc<dyn DxTradeHttpTransport>,
}

impl ProductionDxTradeOrderBackend {
    pub fn new() -> Self {
        Self {
            transport: Arc::new(ReqwestDxTradeTransport::new()),
        }
    }
    pub fn with_transport(transport: Arc<dyn DxTradeHttpTransport>) -> Self {
        Self { transport }
    }
}

impl Default for ProductionDxTradeOrderBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DxTradeOrderBackend for ProductionDxTradeOrderBackend {
    fn submit_order(
        &self,
        session: &DxTradeAuthSession,
        order: &DxTradeNewOrder,
    ) -> Result<DxTradeOrderOutcome> {
        validate_session_for_trading(session)?;
        validate_order_shape(order)?;

        let url = orders_url(session);
        let parent_code = generate_order_code();
        let has_protections = order.stop_loss_price.is_some() || order.take_profit_price.is_some();

        let body_json = if has_protections {
            let sl_code = order.stop_loss_price.map(|_| generate_order_code());
            let tp_code = order.take_profit_price.map(|_| generate_order_code());

            let parent = build_single_request(&parent_code, order, Some("OPEN"), None);
            let mut orders = Vec::with_capacity(3);
            orders.push(parent);
            if let (Some(price), Some(code)) = (order.stop_loss_price, sl_code.as_ref()) {
                orders.push(build_protection_child(
                    code,
                    order,
                    DxTradeOrderKind::Stop,
                    price,
                    order.symbol.as_str(),
                ));
            }
            if let (Some(price), Some(code)) = (order.take_profit_price, tp_code.as_ref()) {
                orders.push(build_protection_child(
                    code,
                    order,
                    DxTradeOrderKind::Limit,
                    price,
                    order.symbol.as_str(),
                ));
            }
            let group = OrderGroupRequest { orders };
            serde_json::to_string(&group)
                .context("DXtrade: failed to serialize Order Group Request")?
        } else {
            let req = build_single_request(&parent_code, order, None, None);
            serde_json::to_string(&req)
                .context("DXtrade: failed to serialize Single Order Request")?
        };

        let resp = self
            .transport
            .post_json(&url, Some(session.session_token.as_str()), &body_json)
            .context("DXtrade place-order transport failed")?;
        let server_order_id = parse_order_response(&resp, &url, "place")?;

        Ok(DxTradeOrderOutcome {
            broker_order_id: parent_code,
            status: DxTradeOrderStatus::Pending,
            fill_price: None,
            fill_volume: None,
            server_order_id,
        })
    }

    fn modify_order(
        &self,
        session: &DxTradeAuthSession,
        order_code: &str,
        order: &DxTradeNewOrder,
    ) -> Result<DxTradeOrderOutcome> {
        validate_session_for_trading(session)?;
        validate_order_shape(order)?;
        if order_code.trim().is_empty() {
            anyhow::bail!("DXtrade modify_order: order_code is empty");
        }

        let url = orders_url(session);
        let req = build_single_request(order_code, order, None, None);
        let body_json = serde_json::to_string(&req)
            .context("DXtrade: failed to serialize modify-order body")?;

        let resp = self
            .transport
            .put_json(&url, session.session_token.as_str(), &body_json)
            .context("DXtrade modify-order transport failed")?;
        let server_order_id = parse_order_response(&resp, &url, "modify")?;

        Ok(DxTradeOrderOutcome {
            broker_order_id: order_code.to_string(),
            status: DxTradeOrderStatus::Pending,
            fill_price: None,
            fill_volume: None,
            server_order_id,
        })
    }

    fn cancel_order(&self, session: &DxTradeAuthSession, broker_order_id: &str) -> Result<()> {
        validate_session_for_trading(session)?;
        if broker_order_id.trim().is_empty() {
            anyhow::bail!("DXtrade cancel_order: broker_order_id is empty");
        }

        // URL-encode the order code in case it contains characters
        // that need percent-encoding (our generator emits only
        // `[a-z0-9-]` so this is currently a no-op, but a caller
        // who reuses a foreign orderCode shouldn't get a malformed
        // request).
        let escaped = url_path_escape(broker_order_id);
        let url = format!(
            "{platform}/dxsca-web/accounts/{account}/orders/{order}",
            platform = session.platform_url,
            account = url_path_escape(&session.account_id),
            order = escaped,
        );

        let resp = self
            .transport
            .delete(&url, session.session_token.as_str())
            .context("DXtrade cancel-order transport failed")?;
        if !(200..300).contains(&resp.status) {
            anyhow::bail!(
                "DXtrade cancel-order returned HTTP {} from {url}: {}",
                resp.status,
                truncate_for_log(&resp.body)
            );
        }
        Ok(())
    }
}

/// Validate that the session has the fields required for an
/// authenticated trading request. Fails BEFORE any HTTP call.
fn validate_session_for_trading(session: &DxTradeAuthSession) -> Result<()> {
    if session.session_token.trim().is_empty() {
        anyhow::bail!("DXtrade order: session_token is empty (auth not completed?)");
    }
    if session.platform_url.trim().is_empty() {
        anyhow::bail!(
            "DXtrade order: platform_url is empty on session — this should \
             be set by login() but appears blank"
        );
    }
    if session.account_id.trim().is_empty() {
        anyhow::bail!(
            "DXtrade order: account_id is empty on session (no execution \
             account configured?)"
        );
    }
    Ok(())
}

/// Validate that a `DxTradeNewOrder` has internally-consistent
/// fields per the DXtrade Single Order Request rules. Fails
/// BEFORE any HTTP call.
fn validate_order_shape(order: &DxTradeNewOrder) -> Result<()> {
    if order.symbol.trim().is_empty() {
        anyhow::bail!("DXtrade order: symbol/instrument is empty");
    }
    if order.volume <= 0 {
        anyhow::bail!(
            "DXtrade order: volume must be positive (got {})",
            order.volume
        );
    }
    match order.kind {
        DxTradeOrderKind::Market => {
            if order.price.is_some() {
                anyhow::bail!(
                    "DXtrade order: MARKET orders must not carry a price — \
                     the spec says limitPrice/stopPrice are absent for \
                     market orders"
                );
            }
        }
        DxTradeOrderKind::Limit => {
            if order.price.is_none() {
                anyhow::bail!("DXtrade order: LIMIT order requires `price` (limitPrice)");
            }
        }
        DxTradeOrderKind::Stop => {
            if order.price.is_none() {
                anyhow::bail!("DXtrade order: STOP order requires `price` (stopPrice)");
            }
        }
    }
    Ok(())
}

/// Construct the orders endpoint URL for the session.
fn orders_url(session: &DxTradeAuthSession) -> String {
    format!(
        "{platform}/dxsca-web/accounts/{account}/orders",
        platform = session.platform_url,
        account = url_path_escape(&session.account_id),
    )
}

/// Parse a Place/Modify response into the `orderId` field.
/// Accepts empty bodies (some broker tenants return 204) and
/// surfaces non-2xx status with the body excerpt for debugging.
fn parse_order_response(resp: &DxTradeHttpResponse, url: &str, verb: &str) -> Result<Option<i64>> {
    if !(200..300).contains(&resp.status) {
        anyhow::bail!(
            "DXtrade {verb}-order returned HTTP {} from {url}: {}",
            resp.status,
            truncate_for_log(&resp.body)
        );
    }
    if resp.body.trim().is_empty() {
        return Ok(None);
    }
    let parsed: OrderResponse = serde_json::from_str(&resp.body).with_context(|| {
        format!(
            "DXtrade {verb}-order: failed to parse response body: {}",
            truncate_for_log(&resp.body)
        )
    })?;
    Ok(parsed.order_id)
}

/// Minimal RFC 3986 path-segment percent-escaping. Covers the
/// characters that realistically show up in DXtrade account
/// codes (`:` is common — e.g. `default:margin_eur_5_BBook`) and
/// in orderCodes. Avoids pulling a new dep for a 20-line task.
fn url_path_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for b in input.bytes() {
        let safe = matches!(b,
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9'
            | b'-' | b'.' | b'_' | b'~'
        );
        if safe {
            out.push(b as char);
        } else {
            out.push('%');
            out.push_str(&format!("{b:02X}"));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Streaming (D3.3)
// ---------------------------------------------------------------------------

/// Live tick / trendbar update from the DXtrade Push API
/// WebSocket feed.
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
    /// Open the Push API market-data WebSocket, subscribe to
    /// quote updates for `symbol`, block until the first matching
    /// Quote arrives, parse it into a [`DxTradeLiveUpdate`] and
    /// close the socket. `timeframe` is accepted for trait
    /// compatibility with the cTrader-side analogue but is
    /// currently ignored — DXtrade Quote updates are tick-level;
    /// the OHLC `Candle` event-type lives in `latest_trendbar`
    /// once a streaming-handle refactor lands (D3.3.1).
    fn subscribe_live_chart(
        &self,
        session: &DxTradeAuthSession,
        symbol: &str,
        timeframe: &str,
    ) -> Result<DxTradeLiveUpdate>;
}

// -- Push API wire types ------------------------------------------------------

/// Outbound Market Data subscription request. Field order
/// matches the developer-portal Example:
/// ```json
/// {
///   "type": "MarketDataSubscriptionRequest",
///   "requestId": "009",
///   "session": "<token>",
///   "payload": {
///     "account": "default:fx1",
///     "symbols": ["EUR/USD"],
///     "eventTypes": [{ "type": "Quote", "format": "COMPACT" }]
///   }
/// }
/// ```
#[derive(Serialize)]
struct MarketDataSubscriptionRequest<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(rename = "requestId")]
    request_id: &'a str,
    session: &'a str,
    payload: MarketDataSubscriptionPayload<'a>,
}

#[derive(Serialize)]
struct MarketDataSubscriptionPayload<'a> {
    account: &'a str,
    symbols: Vec<&'a str>,
    #[serde(rename = "eventTypes")]
    event_types: Vec<MarketDataEventType>,
}

#[derive(Serialize)]
struct MarketDataEventType {
    #[serde(rename = "type")]
    kind: &'static str,
    format: &'static str,
}

/// Inbound server message envelope — common header for every
/// Push API frame. We probe `type` first to route into the
/// concrete payload parser.
#[derive(Deserialize)]
struct InboundEnvelope {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    payload: serde_json::Value,
}

/// Quote update payload. The exact field names follow the
/// "COMPACT" format described in the developer portal; brokers
/// vary slightly so we tolerate either `bid` / `ask` as numbers
/// or as strings via `serde_json::Value` and convert at parse
/// time.
#[derive(Deserialize)]
struct QuotePayload {
    #[serde(default)]
    symbol: Option<String>,
    #[serde(default)]
    bid: serde_json::Value,
    #[serde(default)]
    ask: serde_json::Value,
    #[serde(default)]
    timestamp: serde_json::Value,
}

// -- WebSocket transport abstraction ------------------------------------------

/// Single open WebSocket connection. Sync (matches the trait
/// surface). Tests inject a fake; production uses
/// `TungsteniteDxTradeWebSocket`.
pub trait DxTradeWebSocket: Send {
    /// Send a text frame. JSON-serialized envelope goes here.
    fn send_text(&mut self, text: &str) -> Result<()>;
    /// Block until the next text frame arrives. Returns the
    /// frame body. Errors propagate including timeouts and
    /// remote-close.
    fn recv_text(&mut self) -> Result<String>;
    /// Best-effort graceful close.
    fn close(&mut self) -> Result<()>;
}

/// Factory that opens a fresh authenticated WebSocket to a given
/// Push API URL. Production uses [`TungsteniteWebSocketFactory`];
/// tests inject a fake that returns a pre-loaded fake socket.
pub trait DxTradeWebSocketFactory: Send + Sync {
    fn connect(&self, url: &str, bearer_token: &str) -> Result<Box<dyn DxTradeWebSocket>>;
}

/// Production WebSocket factory. Uses blocking `tungstenite`
/// because the streaming trait method is sync; this matches
/// the calling worker-thread architecture without dragging a
/// tokio runtime into the trait.
pub struct TungsteniteWebSocketFactory;

impl TungsteniteWebSocketFactory {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TungsteniteWebSocketFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl DxTradeWebSocketFactory for TungsteniteWebSocketFactory {
    fn connect(&self, url: &str, bearer_token: &str) -> Result<Box<dyn DxTradeWebSocket>> {
        // Ensure rustls provider is installed before tungstenite
        // builds its ClientConfig — same dual-provider hazard as
        // the REST client (see ctrader_tls comment).
        crate::app_services::ctrader_tls::ensure_ctrader_rustls_provider();

        use tungstenite::client::IntoClientRequest;
        use tungstenite::http::HeaderValue;
        let mut request = url
            .into_client_request()
            .context("DXtrade WS: failed to build client request from URL")?;
        let auth_value = HeaderValue::from_str(&format!("DXAPI {bearer_token}"))
            .context("DXtrade WS: failed to encode Authorization header")?;
        request.headers_mut().insert("Authorization", auth_value);

        let (socket, _resp) =
            tungstenite::connect(request).context("DXtrade WS: handshake failed")?;
        Ok(Box::new(TungsteniteSocket { inner: socket }))
    }
}

/// Concrete wrapper around the tungstenite blocking socket.
struct TungsteniteSocket {
    inner: tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>,
}

impl DxTradeWebSocket for TungsteniteSocket {
    fn send_text(&mut self, text: &str) -> Result<()> {
        use tungstenite::Message;
        self.inner
            .send(Message::Text(text.to_string().into()))
            .context("DXtrade WS: send_text failed")?;
        Ok(())
    }

    fn recv_text(&mut self) -> Result<String> {
        use tungstenite::Message;
        loop {
            let msg = self.inner.read().context("DXtrade WS: recv_text failed")?;
            match msg {
                Message::Text(t) => return Ok(t.to_string()),
                // Ignore protocol-level pings/pongs/binary frames
                // and continue reading. Tungstenite auto-responds
                // to Ping frames in subsequent writes.
                Message::Binary(_) | Message::Ping(_) | Message::Pong(_) => continue,
                Message::Close(frame) => {
                    anyhow::bail!("DXtrade WS: server closed connection: {frame:?}")
                }
                Message::Frame(_) => continue,
            }
        }
    }

    fn close(&mut self) -> Result<()> {
        // Best-effort: any error here is non-fatal — the socket
        // is about to be dropped anyway.
        let _ = self.inner.close(None);
        Ok(())
    }
}

// -- Helpers ------------------------------------------------------------------

/// Build the market-data WebSocket URL for the session. Default
/// path is `/dxsca-web/md?format=JSON` per the official
/// "Establish a Session" doc (market data resource = `/md`
/// suffix; format selected via query param).
pub fn market_data_ws_url(platform_url: &str) -> Result<String> {
    let trimmed = platform_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        anyhow::bail!("DXtrade streaming: platform_url is empty");
    }
    let ws_base = if let Some(rest) = trimmed.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = trimmed.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        anyhow::bail!("DXtrade streaming: platform_url must start with http(s):// (got {trimmed})");
    };
    Ok(format!("{ws_base}/dxsca-web/md?format=JSON"))
}

/// Generate a fresh `requestId` for a Push API request. The
/// envelope must use unique requestIds per session so that
/// responses don't mix up; we reuse the same 128-bit entropy
/// path as `generate_order_code`.
fn generate_request_id() -> String {
    use rand::TryRngCore;
    let mut bytes = [0u8; 16];
    if rand::rngs::OsRng.try_fill_bytes(&mut bytes).is_err() {
        let now = current_unix_seconds() as u128;
        return format!("req-fallback-{now:032x}");
    }
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("req-{hex}")
}

/// Coerce a `serde_json::Value` to `Option<f64>` — Push API
/// brokers serialize prices either as JSON numbers or as JSON
/// strings; we accept both.
fn json_to_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Coerce a `serde_json::Value` to `Option<i64>` — millisecond
/// timestamp. Accepts numbers, integer strings, and (defensively)
/// rounded floats.
fn json_to_i64_ms(v: &serde_json::Value) -> Option<i64> {
    match v {
        serde_json::Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Default per-symbol subscribe timeout — how long
/// `subscribe_live_chart` blocks waiting for the first Quote
/// before giving up. 30 seconds is generous for the demo
/// environment while still bounded enough to fail fast in
/// production when something's wrong with the subscription
/// pipeline.
pub const DXTRADE_SUBSCRIBE_TIMEOUT_SECONDS: u64 = 30;

/// Production DXtrade streaming backend.
pub struct ProductionDxTradeStreamingBackend {
    factory: Arc<dyn DxTradeWebSocketFactory>,
    max_frames_before_timeout: usize,
}

impl ProductionDxTradeStreamingBackend {
    pub fn new() -> Self {
        Self {
            factory: Arc::new(TungsteniteWebSocketFactory::new()),
            // Default cap on how many non-matching frames we read
            // (server announcements, other symbols, etc.) before
            // giving up. Generous enough to survive a noisy
            // subscription handshake.
            max_frames_before_timeout: 256,
        }
    }

    pub fn with_factory(factory: Arc<dyn DxTradeWebSocketFactory>) -> Self {
        Self {
            factory,
            max_frames_before_timeout: 256,
        }
    }
}

impl Default for ProductionDxTradeStreamingBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DxTradeStreamingBackend for ProductionDxTradeStreamingBackend {
    fn subscribe_live_chart(
        &self,
        session: &DxTradeAuthSession,
        symbol: &str,
        _timeframe: &str,
    ) -> Result<DxTradeLiveUpdate> {
        validate_session_for_trading(session)?;
        if symbol.trim().is_empty() {
            anyhow::bail!("DXtrade streaming: symbol is empty");
        }

        let url = market_data_ws_url(&session.platform_url)?;
        let mut ws = self
            .factory
            .connect(&url, session.session_token.as_str())
            .context("DXtrade streaming: WebSocket connect failed")?;

        let request_id = generate_request_id();
        let sub = MarketDataSubscriptionRequest {
            kind: "MarketDataSubscriptionRequest",
            request_id: &request_id,
            session: session.session_token.as_str(),
            payload: MarketDataSubscriptionPayload {
                account: session.account_id.as_str(),
                symbols: vec![symbol],
                event_types: vec![MarketDataEventType {
                    kind: "Quote",
                    format: "COMPACT",
                }],
            },
        };
        let sub_json = serde_json::to_string(&sub)
            .context("DXtrade streaming: failed to serialize subscription")?;
        ws.send_text(&sub_json)
            .context("DXtrade streaming: failed to send subscription")?;

        // Read frames until we get a Quote for the right symbol
        // (or hit the frame cap, or the socket errors). The
        // server may interleave subscription-ack messages and
        // unrelated frames before delivering the first quote.
        let result = drain_until_quote(ws.as_mut(), symbol, self.max_frames_before_timeout);

        // Best-effort close regardless of outcome.
        let _ = ws.close();

        result
    }
}

/// Read frames until we get a Quote matching `symbol`. Returns
/// the parsed `DxTradeLiveUpdate` on success.
fn drain_until_quote(
    ws: &mut dyn DxTradeWebSocket,
    symbol: &str,
    max_frames: usize,
) -> Result<DxTradeLiveUpdate> {
    for _ in 0..max_frames {
        let text = ws.recv_text().context("DXtrade streaming: read failed")?;
        let envelope: InboundEnvelope = match serde_json::from_str(&text) {
            Ok(e) => e,
            // Drop frames we can't parse — server may emit
            // diagnostic / heartbeat frames in shapes we don't
            // model. The frame budget is what bounds this loop.
            Err(_) => continue,
        };
        if envelope.kind != "Quote" {
            continue;
        }
        // Try to parse the payload as a QuotePayload. If the
        // shape doesn't match (e.g. broker-specific field
        // naming), drop and keep looking.
        let quote: QuotePayload = match serde_json::from_value(envelope.payload.clone()) {
            Ok(q) => q,
            Err(_) => continue,
        };
        // Symbol match — when the server doesn't include a
        // symbol field, accept (we only subscribed to one).
        if let Some(s) = &quote.symbol {
            if s != symbol {
                continue;
            }
        }
        return Ok(DxTradeLiveUpdate {
            symbol: symbol.to_string(),
            bid: json_to_f64(&quote.bid),
            ask: json_to_f64(&quote.ask),
            timestamp_ms: json_to_i64_ms(&quote.timestamp),
            latest_trendbar: None,
        });
    }
    anyhow::bail!(
        "DXtrade streaming: no Quote for {symbol} arrived after \
         reading {max_frames} frames; subscribe handshake may have \
         been rejected or symbol is misnamed"
    )
}

// ---------------------------------------------------------------------------
// Bundle that composes the three trait objects
// ---------------------------------------------------------------------------

/// One-stop DXtrade backend handle — wraps the three trait objects
/// the TradingSession needs. Constructed via
/// [`Self::production`] (D3.1 live + D3.2/D3.3 stubs) or via test
/// helpers that inject fakes per-trait.
pub struct DxTradeBackend {
    pub auth: Arc<dyn DxTradeAuthBackend>,
    pub orders: Arc<dyn DxTradeOrderBackend>,
    pub streaming: Arc<dyn DxTradeStreamingBackend>,
}

impl DxTradeBackend {
    /// Production-grade backend. D3.1 (auth), D3.2 (orders) and
    /// D3.3 (streaming via the DXtrade Push API — single-shot
    /// Quote subscribe) are all live.
    pub fn production() -> Self {
        Self {
            auth: Arc::new(ProductionDxTradeAuthBackend::new()),
            orders: Arc::new(ProductionDxTradeOrderBackend::new()),
            streaming: Arc::new(ProductionDxTradeStreamingBackend::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // -- Test transport -----------------------------------------------------

    struct FakeTransport {
        next: Mutex<Vec<Result<DxTradeHttpResponse>>>,
        recorded: Mutex<Vec<RecordedCall>>,
    }

    #[derive(Clone)]
    struct RecordedCall {
        method: String,
        url: String,
        bearer: Option<String>,
        body: String,
    }

    impl FakeTransport {
        fn with_responses(responses: Vec<Result<DxTradeHttpResponse>>) -> Self {
            Self {
                next: Mutex::new(responses),
                recorded: Mutex::new(vec![]),
            }
        }
        fn calls(&self) -> Vec<RecordedCall> {
            self.recorded.lock().unwrap().clone()
        }
    }

    impl DxTradeHttpTransport for FakeTransport {
        fn post_json(
            &self,
            url: &str,
            bearer_token: Option<&str>,
            body: &str,
        ) -> Result<DxTradeHttpResponse> {
            self.recorded.lock().unwrap().push(RecordedCall {
                method: "POST".to_string(),
                url: url.to_string(),
                bearer: bearer_token.map(|s| s.to_string()),
                body: body.to_string(),
            });
            let mut q = self.next.lock().unwrap();
            if q.is_empty() {
                anyhow::bail!("FakeTransport: no canned response left");
            }
            q.remove(0)
        }

        fn put_json(
            &self,
            url: &str,
            bearer_token: &str,
            body: &str,
        ) -> Result<DxTradeHttpResponse> {
            self.recorded.lock().unwrap().push(RecordedCall {
                method: "PUT".to_string(),
                url: url.to_string(),
                bearer: Some(bearer_token.to_string()),
                body: body.to_string(),
            });
            let mut q = self.next.lock().unwrap();
            if q.is_empty() {
                anyhow::bail!("FakeTransport: no canned response left");
            }
            q.remove(0)
        }

        fn delete(&self, url: &str, bearer_token: &str) -> Result<DxTradeHttpResponse> {
            self.recorded.lock().unwrap().push(RecordedCall {
                method: "DELETE".to_string(),
                url: url.to_string(),
                bearer: Some(bearer_token.to_string()),
                body: String::new(),
            });
            let mut q = self.next.lock().unwrap();
            if q.is_empty() {
                anyhow::bail!("FakeTransport: no canned response left");
            }
            q.remove(0)
        }
    }

    fn ok_response(body: &str) -> Result<DxTradeHttpResponse> {
        Ok(DxTradeHttpResponse {
            status: 200,
            body: body.to_string(),
        })
    }

    fn err_response(status: u16, body: &str) -> Result<DxTradeHttpResponse> {
        Ok(DxTradeHttpResponse {
            status,
            body: body.to_string(),
        })
    }

    fn good_settings() -> DxTradeBrokerSettings {
        DxTradeBrokerSettings {
            platform_url: "https://demo.dx.trade".to_string(),
            username: "alice".to_string(),
            domain: "default".to_string(),
            password: "secret".to_string(),
            accounts: vec![],
        }
    }

    fn dummy_session() -> DxTradeAuthSession {
        DxTradeAuthSession {
            session_token: "token".to_string(),
            expires_at_unix: 0,
            account_id: "acc-1".to_string(),
            platform_url: "https://demo.dx.trade".to_string(),
        }
    }

    fn market_buy_eurusd() -> DxTradeNewOrder {
        DxTradeNewOrder {
            symbol: "EUR/USD".to_string(),
            side: DxTradeOrderSide::Buy,
            kind: DxTradeOrderKind::Market,
            volume: 100_000,
            price: None,
            stop_loss_price: None,
            take_profit_price: None,
        }
    }

    // -- D3.1 auth — happy path --------------------------------------------

    #[test]
    fn login_hits_official_dxsca_web_login_path() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"sessionToken":"tok-1","timeout":"PT30M"}"#,
        )]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport.clone());

        let session = backend.login(&good_settings()).expect("login ok");
        assert_eq!(session.session_token, "tok-1");

        let calls = transport.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].url, "https://demo.dx.trade/dxsca-web/login");
        assert!(
            calls[0].bearer.is_none(),
            "login must not send Authorization header — it's how we get the \
             token in the first place"
        );
    }

    #[test]
    fn login_sends_username_domain_password_per_official_spec() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"sessionToken":"tok","timeout":"PT30M"}"#,
        )]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport.clone());
        backend.login(&good_settings()).expect("login ok");

        let body = &transport.calls()[0].body;
        let parsed: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(parsed["username"], "alice");
        assert_eq!(parsed["domain"], "default");
        assert_eq!(parsed["password"], "secret");
        // No stray fields — e.g. the Go-reference `vendor` /
        // `accountId` keys must NOT leak in (different platform
        // variant; would confuse a real DXtrade server).
        assert!(parsed.get("vendor").is_none());
        assert!(parsed.get("accountId").is_none());
    }

    #[test]
    fn login_strips_trailing_slash_from_platform_url() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"sessionToken":"tok","timeout":"PT30M"}"#,
        )]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport.clone());
        let mut settings = good_settings();
        settings.platform_url = "https://demo.dx.trade/".to_string();
        backend.login(&settings).expect("login ok");
        assert_eq!(
            transport.calls()[0].url,
            "https://demo.dx.trade/dxsca-web/login"
        );
    }

    #[test]
    fn login_iso_duration_becomes_expires_at_unix_offset() {
        let now = current_unix_seconds();
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"sessionToken":"tok","timeout":"PT15M"}"#,
        )]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport);
        let s = backend.login(&good_settings()).unwrap();
        let drift = (s.expires_at_unix - (now + 15 * 60)).abs();
        assert!(drift <= 2, "expected ~15min TTL, drift = {drift}s");
    }

    #[test]
    fn login_picks_enabled_account_first_then_falls_back_to_first_overall() {
        use crate::app_services::broker_config::BrokerAccountTarget;
        let transport = Arc::new(FakeTransport::with_responses(vec![
            ok_response(r#"{"sessionToken":"a","timeout":"PT30M"}"#),
            ok_response(r#"{"sessionToken":"b","timeout":"PT30M"}"#),
            ok_response(r#"{"sessionToken":"c","timeout":"PT30M"}"#),
        ]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport);

        // (1) No accounts → empty account_id.
        let s1 = backend.login(&good_settings()).unwrap();
        assert_eq!(s1.account_id, "");

        // (2) Two accounts, second flagged → picks second.
        let mut two = good_settings();
        two.accounts = vec![
            BrokerAccountTarget {
                account_id: "primary".to_string(),
                label: "P".to_string(),
                enabled_for_execution: false,
            },
            BrokerAccountTarget {
                account_id: "execution".to_string(),
                label: "E".to_string(),
                enabled_for_execution: true,
            },
        ];
        let s2 = backend.login(&two).unwrap();
        assert_eq!(s2.account_id, "execution");

        // (3) None flagged → first overall.
        let mut none_flagged = good_settings();
        none_flagged.accounts = vec![BrokerAccountTarget {
            account_id: "only".to_string(),
            label: "O".to_string(),
            enabled_for_execution: false,
        }];
        let s3 = backend.login(&none_flagged).unwrap();
        assert_eq!(s3.account_id, "only");
    }

    // -- D3.1 auth — error paths --------------------------------------------

    #[test]
    fn login_rejects_missing_domain_with_clear_error_and_no_http_call() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport.clone());
        let mut settings = good_settings();
        settings.domain.clear();

        let err = backend.login(&settings).expect_err("must bail");
        assert!(
            err.to_string().contains("domain"),
            "error must name the missing field: {err}"
        );
        assert!(
            transport.calls().is_empty(),
            "must not hit the network when required fields are missing"
        );
    }

    #[test]
    fn login_rejects_missing_username() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport);
        let mut settings = good_settings();
        settings.username.clear();
        let err = backend.login(&settings).expect_err("must bail");
        assert!(err.to_string().contains("username"));
    }

    #[test]
    fn login_rejects_missing_password() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport);
        let mut settings = good_settings();
        settings.password.clear();
        let err = backend.login(&settings).expect_err("must bail");
        assert!(err.to_string().contains("password"));
    }

    #[test]
    fn login_rejects_platform_url_without_scheme() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport);
        let mut settings = good_settings();
        settings.platform_url = "demo.dx.trade".to_string();
        let err = backend.login(&settings).expect_err("must bail");
        assert!(err.to_string().contains("scheme"));
    }

    #[test]
    fn login_surfaces_http_4xx_status_and_body_excerpt() {
        let transport = Arc::new(FakeTransport::with_responses(vec![err_response(
            401,
            r#"{"error":"Invalid credentials"}"#,
        )]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport);
        let err = backend.login(&good_settings()).expect_err("must bail");
        let msg = err.to_string();
        assert!(msg.contains("401"), "no status code: {msg}");
        assert!(
            msg.contains("Invalid credentials"),
            "no body excerpt: {msg}"
        );
    }

    #[test]
    fn login_rejects_empty_session_token_in_otherwise_2xx_response() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"sessionToken":"","timeout":"PT30M"}"#,
        )]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport);
        let err = backend.login(&good_settings()).expect_err("must bail");
        assert!(err.to_string().contains("empty sessionToken"));
    }

    #[test]
    fn login_falls_back_to_default_ttl_when_timeout_unparseable() {
        let now = current_unix_seconds();
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"sessionToken":"tok","timeout":"???"}"#,
        )]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport);
        let s = backend.login(&good_settings()).unwrap();
        let drift = (s.expires_at_unix - (now + DXTRADE_DEFAULT_SESSION_TTL_SECONDS)).abs();
        assert!(drift <= 2, "expected default TTL, drift = {drift}s");
    }

    #[test]
    fn login_falls_back_to_default_ttl_when_timeout_missing_entirely() {
        // Server sends only `sessionToken` — `timeout` field
        // entirely absent. Our `#[serde(default)]` resolves it to
        // an empty string, which `parse_timeout_seconds` rejects;
        // the auth path must still succeed using the default TTL.
        let now = current_unix_seconds();
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"sessionToken":"tok"}"#,
        )]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport);
        let s = backend.login(&good_settings()).unwrap();
        let drift = (s.expires_at_unix - (now + DXTRADE_DEFAULT_SESSION_TTL_SECONDS)).abs();
        assert!(drift <= 2, "expected default TTL, drift = {drift}s");
    }

    #[test]
    fn login_surfaces_malformed_json_response_body() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            "this is not JSON",
        )]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport);
        let err = backend.login(&good_settings()).expect_err("must bail");
        assert!(err.to_string().contains("failed to parse"), "msg: {err}");
    }

    #[test]
    fn login_propagates_transport_error_with_context() {
        let transport = Arc::new(FakeTransport::with_responses(vec![Err(anyhow::anyhow!(
            "DNS resolution failed"
        ))]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport);
        let err = backend.login(&good_settings()).expect_err("must bail");
        let chain: Vec<String> = err.chain().map(|c| c.to_string()).collect();
        assert!(
            chain
                .iter()
                .any(|c| c.contains("DXtrade /login transport failed")),
            "chain: {chain:?}"
        );
    }

    // -- D3.1 refresh -------------------------------------------------------

    #[test]
    fn refresh_re_calls_login_and_returns_new_token() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"sessionToken":"refreshed","timeout":"PT30M"}"#,
        )]));
        let backend = ProductionDxTradeAuthBackend::with_transport(transport.clone());
        let prev = DxTradeAuthSession {
            session_token: "old".to_string(),
            expires_at_unix: 0,
            account_id: String::new(),
            platform_url: "https://demo.dx.trade".to_string(),
        };
        let s = backend.refresh(&good_settings(), &prev).unwrap();
        assert_eq!(s.session_token, "refreshed");
        assert_eq!(transport.calls().len(), 1);
        assert_eq!(
            transport.calls()[0].url,
            "https://demo.dx.trade/dxsca-web/login"
        );
    }

    // -- Pure helpers -------------------------------------------------------

    #[test]
    fn parse_timeout_seconds_handles_iso8601_duration_subset() {
        assert_eq!(parse_timeout_seconds("PT30M"), Some(1800));
        assert_eq!(parse_timeout_seconds("PT1H"), Some(3600));
        assert_eq!(parse_timeout_seconds("PT15M30S"), Some(15 * 60 + 30));
        assert_eq!(parse_timeout_seconds("PT2H45M"), Some(2 * 3600 + 45 * 60));
        assert_eq!(parse_timeout_seconds("PT90S"), Some(90));
    }

    #[test]
    fn parse_timeout_seconds_handles_plain_integer_seconds() {
        assert_eq!(parse_timeout_seconds("1800"), Some(1800));
        assert_eq!(parse_timeout_seconds("3600"), Some(3600));
        assert_eq!(parse_timeout_seconds("60"), Some(60));
    }

    #[test]
    fn parse_timeout_seconds_interprets_huge_integers_as_milliseconds() {
        // 1.8M ms = 1800s = 30 min — typical Java-y session.
        assert_eq!(parse_timeout_seconds("1800000"), Some(1800));
        assert_eq!(parse_timeout_seconds("3600000"), Some(3600));
    }

    #[test]
    fn parse_timeout_seconds_rejects_garbage_and_edge_cases() {
        assert_eq!(parse_timeout_seconds(""), None);
        assert_eq!(parse_timeout_seconds("   "), None);
        assert_eq!(parse_timeout_seconds("garbage"), None);
        assert_eq!(parse_timeout_seconds("PT"), None);
        assert_eq!(parse_timeout_seconds("PT30"), None); // missing unit
        assert_eq!(parse_timeout_seconds("PTabc"), None);
        assert_eq!(parse_timeout_seconds("0"), None); // zero TTL is nonsense
        assert_eq!(parse_timeout_seconds("-1"), None);
    }

    // -- D3.2 orders — Place Order (Single Order Request) -------------------

    #[test]
    fn submit_market_order_posts_to_canonical_orders_url_with_auth() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"orderId":63655}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());

        let outcome = backend
            .submit_order(&dummy_session(), &market_buy_eurusd())
            .expect("submit ok");

        assert!(outcome.broker_order_id.starts_with("forex-ai-"));
        assert_eq!(outcome.status, DxTradeOrderStatus::Pending);
        assert_eq!(outcome.server_order_id, Some(63655));

        let calls = transport.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].method, "POST");
        assert_eq!(
            calls[0].url,
            "https://demo.dx.trade/dxsca-web/accounts/acc-1/orders"
        );
        assert_eq!(calls[0].bearer.as_deref(), Some("token"));
    }

    #[test]
    fn submit_market_order_emits_single_order_request_body() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"orderId":1}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        backend
            .submit_order(&dummy_session(), &market_buy_eurusd())
            .expect("submit ok");

        let body: serde_json::Value = serde_json::from_str(&transport.calls()[0].body).unwrap();
        // Single Order (not wrapped in {"orders":[]})
        assert!(
            body.get("orders").is_none(),
            "market order without SL/TP must be a Single Order Request, not a group"
        );
        assert_eq!(body["type"], "MARKET");
        assert_eq!(body["instrument"], "EUR/USD");
        assert_eq!(body["side"], "BUY");
        assert_eq!(body["quantity"], "100000");
        assert_eq!(body["tif"], "GTC");
        // MARKET orders carry NEITHER limitPrice NOR stopPrice.
        assert!(body.get("limitPrice").is_none(), "{body:?}");
        assert!(body.get("stopPrice").is_none(), "{body:?}");
        // orderCode is the client-unique id we generate.
        let order_code = body["orderCode"].as_str().unwrap();
        assert!(order_code.starts_with("forex-ai-"));
    }

    #[test]
    fn submit_limit_order_sets_limit_price_only() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"orderId":1}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut order = market_buy_eurusd();
        order.kind = DxTradeOrderKind::Limit;
        order.price = Some(1.08123);
        backend.submit_order(&dummy_session(), &order).unwrap();

        let body: serde_json::Value = serde_json::from_str(&transport.calls()[0].body).unwrap();
        assert_eq!(body["type"], "LIMIT");
        assert_eq!(body["limitPrice"], "1.08123");
        assert!(body.get("stopPrice").is_none());
    }

    #[test]
    fn submit_stop_order_sets_stop_price_only() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"orderId":1}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut order = market_buy_eurusd();
        order.kind = DxTradeOrderKind::Stop;
        order.price = Some(1.05);
        backend.submit_order(&dummy_session(), &order).unwrap();

        let body: serde_json::Value = serde_json::from_str(&transport.calls()[0].body).unwrap();
        assert_eq!(body["type"], "STOP");
        assert_eq!(body["stopPrice"], "1.05");
        assert!(body.get("limitPrice").is_none());
    }

    #[test]
    fn submit_with_sl_only_emits_if_then_group_with_stop_protection() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"orderId":1}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut order = market_buy_eurusd();
        order.stop_loss_price = Some(1.05);
        backend.submit_order(&dummy_session(), &order).unwrap();

        let body: serde_json::Value = serde_json::from_str(&transport.calls()[0].body).unwrap();
        let orders = body["orders"].as_array().expect("must be a group");
        assert_eq!(orders.len(), 2, "parent + SL only");

        // Parent: MARKET BUY, positionEffect OPEN
        assert_eq!(orders[0]["type"], "MARKET");
        assert_eq!(orders[0]["side"], "BUY");
        assert_eq!(orders[0]["positionEffect"], "OPEN");
        assert_eq!(orders[0]["quantity"], "100000");

        // SL child: STOP SELL with quantity "0" (inherit) and CLOSE
        assert_eq!(orders[1]["type"], "STOP");
        assert_eq!(orders[1]["side"], "SELL");
        assert_eq!(orders[1]["positionEffect"], "CLOSE");
        assert_eq!(orders[1]["quantity"], "0");
        assert_eq!(orders[1]["stopPrice"], "1.05");
    }

    #[test]
    fn submit_with_tp_only_emits_if_then_group_with_limit_protection() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"orderId":1}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut order = market_buy_eurusd();
        order.take_profit_price = Some(1.12);
        backend.submit_order(&dummy_session(), &order).unwrap();

        let body: serde_json::Value = serde_json::from_str(&transport.calls()[0].body).unwrap();
        let orders = body["orders"].as_array().expect("must be a group");
        assert_eq!(orders.len(), 2, "parent + TP only");

        assert_eq!(orders[1]["type"], "LIMIT");
        assert_eq!(orders[1]["side"], "SELL");
        assert_eq!(orders[1]["positionEffect"], "CLOSE");
        assert_eq!(orders[1]["quantity"], "0");
        assert_eq!(orders[1]["limitPrice"], "1.12");
    }

    #[test]
    fn submit_with_sl_and_tp_emits_three_leg_group() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"orderId":1}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut order = market_buy_eurusd();
        order.stop_loss_price = Some(1.05);
        order.take_profit_price = Some(1.12);
        backend.submit_order(&dummy_session(), &order).unwrap();

        let body: serde_json::Value = serde_json::from_str(&transport.calls()[0].body).unwrap();
        let orders = body["orders"].as_array().expect("must be a group");
        assert_eq!(orders.len(), 3, "parent + SL + TP");
        assert_eq!(orders[0]["positionEffect"], "OPEN");
        assert_eq!(orders[1]["type"], "STOP");
        assert_eq!(orders[2]["type"], "LIMIT");
        // Each protection leg has its own orderCode — must all be
        // distinct so DXtrade can reference them individually.
        let parent_code = orders[0]["orderCode"].as_str().unwrap();
        let sl_code = orders[1]["orderCode"].as_str().unwrap();
        let tp_code = orders[2]["orderCode"].as_str().unwrap();
        assert_ne!(parent_code, sl_code);
        assert_ne!(parent_code, tp_code);
        assert_ne!(sl_code, tp_code);
    }

    #[test]
    fn submit_sell_with_sl_uses_buy_side_for_protection() {
        // Protection legs always go the OPPOSITE direction from
        // the parent — otherwise they don't close the position.
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"orderId":1}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut order = market_buy_eurusd();
        order.side = DxTradeOrderSide::Sell;
        order.stop_loss_price = Some(1.12);
        backend.submit_order(&dummy_session(), &order).unwrap();

        let body: serde_json::Value = serde_json::from_str(&transport.calls()[0].body).unwrap();
        let orders = body["orders"].as_array().unwrap();
        assert_eq!(orders[0]["side"], "SELL");
        assert_eq!(orders[1]["side"], "BUY", "SL must reverse the parent");
    }

    // -- D3.2 orders — validation gates -------------------------------------

    #[test]
    fn submit_rejects_market_with_price_set() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut order = market_buy_eurusd();
        order.price = Some(1.0);
        let err = backend
            .submit_order(&dummy_session(), &order)
            .expect_err("must bail");
        assert!(err.to_string().contains("MARKET"));
        assert!(transport.calls().is_empty());
    }

    #[test]
    fn submit_rejects_limit_without_price() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut order = market_buy_eurusd();
        order.kind = DxTradeOrderKind::Limit;
        order.price = None;
        let err = backend
            .submit_order(&dummy_session(), &order)
            .expect_err("must bail");
        assert!(err.to_string().contains("LIMIT"));
    }

    #[test]
    fn submit_rejects_zero_or_negative_volume() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut order = market_buy_eurusd();
        order.volume = 0;
        assert!(backend.submit_order(&dummy_session(), &order).is_err());
        order.volume = -1;
        assert!(backend.submit_order(&dummy_session(), &order).is_err());
        assert!(transport.calls().is_empty());
    }

    #[test]
    fn submit_rejects_empty_symbol() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut order = market_buy_eurusd();
        order.symbol.clear();
        let err = backend
            .submit_order(&dummy_session(), &order)
            .expect_err("must bail");
        assert!(err.to_string().contains("symbol") || err.to_string().contains("instrument"));
    }

    #[test]
    fn submit_rejects_unauthenticated_session() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut session = dummy_session();
        session.session_token.clear();
        let err = backend
            .submit_order(&session, &market_buy_eurusd())
            .expect_err("must bail");
        assert!(err.to_string().contains("session_token"));
        assert!(transport.calls().is_empty());
    }

    #[test]
    fn submit_rejects_session_with_no_platform_url() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut session = dummy_session();
        session.platform_url.clear();
        let err = backend
            .submit_order(&session, &market_buy_eurusd())
            .expect_err("must bail");
        assert!(err.to_string().contains("platform_url"));
    }

    #[test]
    fn submit_rejects_session_with_no_account_id() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut session = dummy_session();
        session.account_id.clear();
        let err = backend
            .submit_order(&session, &market_buy_eurusd())
            .expect_err("must bail");
        assert!(err.to_string().contains("account_id"));
    }

    #[test]
    fn submit_url_encodes_account_id_with_special_characters() {
        // DXtrade account codes commonly contain ':' (e.g.
        // "default:margin_eur_5_BBook"). RFC 3986 says ':' is
        // reserved in path segments — our escape function should
        // percent-encode it as %3A so the URL stays valid.
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"orderId":1}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut session = dummy_session();
        session.account_id = "default:margin_eur_5_BBook".to_string();
        backend
            .submit_order(&session, &market_buy_eurusd())
            .unwrap();
        assert!(
            transport.calls()[0]
                .url
                .contains("default%3Amargin_eur_5_BBook"),
            "url = {}",
            transport.calls()[0].url
        );
    }

    #[test]
    fn submit_surfaces_http_4xx_with_status_and_body_excerpt() {
        let transport = Arc::new(FakeTransport::with_responses(vec![err_response(
            422,
            r#"{"error":"InvalidSymbol"}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport);
        let err = backend
            .submit_order(&dummy_session(), &market_buy_eurusd())
            .expect_err("must bail");
        let msg = err.to_string();
        assert!(msg.contains("422"), "{msg}");
        assert!(msg.contains("InvalidSymbol"), "{msg}");
    }

    #[test]
    fn submit_handles_empty_response_body_as_no_server_order_id() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response("")]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport);
        let outcome = backend
            .submit_order(&dummy_session(), &market_buy_eurusd())
            .unwrap();
        assert_eq!(outcome.server_order_id, None);
        assert_eq!(outcome.status, DxTradeOrderStatus::Pending);
    }

    // -- D3.2 orders — Modify Order -----------------------------------------

    #[test]
    fn modify_order_puts_to_orders_url_with_existing_order_code() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response(
            r#"{"orderId":2}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let mut order = market_buy_eurusd();
        order.kind = DxTradeOrderKind::Limit;
        order.price = Some(1.10);

        let outcome = backend
            .modify_order(&dummy_session(), "forex-ai-deadbeef", &order)
            .expect("modify ok");

        assert_eq!(outcome.broker_order_id, "forex-ai-deadbeef");
        assert_eq!(outcome.server_order_id, Some(2));

        let calls = transport.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].method, "PUT");
        assert_eq!(
            calls[0].url,
            "https://demo.dx.trade/dxsca-web/accounts/acc-1/orders"
        );
        let body: serde_json::Value = serde_json::from_str(&calls[0].body).unwrap();
        // Modify reuses the existing orderCode passed in — must
        // NOT mint a new client id.
        assert_eq!(body["orderCode"], "forex-ai-deadbeef");
        assert_eq!(body["limitPrice"], "1.1");
    }

    #[test]
    fn modify_rejects_empty_order_code() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let err = backend
            .modify_order(&dummy_session(), "", &market_buy_eurusd())
            .expect_err("must bail");
        assert!(err.to_string().contains("order_code"));
        assert!(transport.calls().is_empty());
    }

    // -- D3.2 orders — Cancel Order -----------------------------------------

    #[test]
    fn cancel_order_deletes_canonical_url_with_auth() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response("")]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());

        backend
            .cancel_order(&dummy_session(), "forex-ai-deadbeef")
            .expect("cancel ok");

        let calls = transport.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].method, "DELETE");
        assert_eq!(
            calls[0].url,
            "https://demo.dx.trade/dxsca-web/accounts/acc-1/orders/forex-ai-deadbeef"
        );
        assert_eq!(calls[0].bearer.as_deref(), Some("token"));
        assert_eq!(calls[0].body, "");
    }

    #[test]
    fn cancel_surfaces_409_conflict_when_order_is_final() {
        // Per the spec: "the order ... must NOT be in a final
        // status (e.g. COMPLETED, CANCELED, EXPIRED, REJECTED).
        // If it is, the server returns 409 Conflict with error
        // code 1005 ('Reference order is closed')."
        let transport = Arc::new(FakeTransport::with_responses(vec![err_response(
            409,
            r#"{"code":1005,"message":"Reference order is closed"}"#,
        )]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport);
        let err = backend
            .cancel_order(&dummy_session(), "forex-ai-x")
            .expect_err("must bail");
        let msg = err.to_string();
        assert!(msg.contains("409"));
        assert!(msg.contains("Reference order is closed"));
    }

    #[test]
    fn cancel_rejects_empty_order_id() {
        let transport = Arc::new(FakeTransport::with_responses(vec![]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        let err = backend
            .cancel_order(&dummy_session(), "")
            .expect_err("must bail");
        assert!(err.to_string().contains("broker_order_id"));
        assert!(transport.calls().is_empty());
    }

    #[test]
    fn cancel_url_encodes_order_code() {
        let transport = Arc::new(FakeTransport::with_responses(vec![ok_response("")]));
        let backend = ProductionDxTradeOrderBackend::with_transport(transport.clone());
        backend
            .cancel_order(&dummy_session(), "weird/order:code")
            .expect("cancel ok");
        // '/' and ':' must percent-encode to %2F and %3A.
        let url = &transport.calls()[0].url;
        assert!(url.contains("weird%2Forder%3Acode"), "url = {url}");
    }

    // -- D3.2 helpers -------------------------------------------------------

    #[test]
    fn generate_order_code_is_unique_across_calls() {
        let n = 32;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..n {
            assert!(seen.insert(generate_order_code()));
        }
    }

    #[test]
    fn generate_order_code_carries_forex_ai_prefix() {
        let code = generate_order_code();
        assert!(code.starts_with("forex-ai-"), "code = {code}");
    }

    #[test]
    fn format_price_trims_trailing_zeros_and_decimal_point() {
        assert_eq!(format_price(1.05), "1.05");
        assert_eq!(format_price(1.0), "1");
        assert_eq!(format_price(1.50000), "1.5");
        assert_eq!(format_price(1.08123), "1.08123");
        assert_eq!(format_price(0.0), "0");
    }

    #[test]
    fn url_path_escape_handles_reserved_characters() {
        assert_eq!(url_path_escape("simple-id_42"), "simple-id_42");
        assert_eq!(url_path_escape("a:b"), "a%3Ab");
        assert_eq!(url_path_escape("a/b"), "a%2Fb");
        assert_eq!(url_path_escape("a b"), "a%20b");
    }

    #[test]
    fn opposite_side_inverts_buy_sell() {
        assert_eq!(opposite_side(DxTradeOrderSide::Buy), DxTradeOrderSide::Sell);
        assert_eq!(opposite_side(DxTradeOrderSide::Sell), DxTradeOrderSide::Buy);
    }

    // -- D3.3 streaming — Push API WebSocket -------------------------------

    /// Fake WebSocket connection that returns canned frames.
    struct FakeWebSocket {
        sent: Arc<Mutex<Vec<String>>>,
        incoming: Arc<Mutex<Vec<Result<String>>>>,
        closed: Arc<Mutex<bool>>,
    }

    impl DxTradeWebSocket for FakeWebSocket {
        fn send_text(&mut self, text: &str) -> Result<()> {
            self.sent.lock().unwrap().push(text.to_string());
            Ok(())
        }
        fn recv_text(&mut self) -> Result<String> {
            let mut q = self.incoming.lock().unwrap();
            if q.is_empty() {
                anyhow::bail!("FakeWebSocket: no canned frame left");
            }
            q.remove(0)
        }
        fn close(&mut self) -> Result<()> {
            *self.closed.lock().unwrap() = true;
            Ok(())
        }
    }

    /// Fake factory that records what was connected and hands
    /// out a pre-loaded FakeWebSocket on `connect`.
    struct FakeWsFactory {
        sent: Arc<Mutex<Vec<String>>>,
        incoming: Arc<Mutex<Vec<Result<String>>>>,
        closed: Arc<Mutex<bool>>,
        connected_to: Arc<Mutex<Option<(String, String)>>>,
    }

    impl FakeWsFactory {
        fn with_incoming(frames: Vec<Result<String>>) -> Arc<Self> {
            Arc::new(Self {
                sent: Arc::new(Mutex::new(vec![])),
                incoming: Arc::new(Mutex::new(frames)),
                closed: Arc::new(Mutex::new(false)),
                connected_to: Arc::new(Mutex::new(None)),
            })
        }
        fn sent_frames(&self) -> Vec<String> {
            self.sent.lock().unwrap().clone()
        }
        fn was_closed(&self) -> bool {
            *self.closed.lock().unwrap()
        }
        fn connected_url(&self) -> Option<(String, String)> {
            self.connected_to.lock().unwrap().clone()
        }
    }

    impl DxTradeWebSocketFactory for FakeWsFactory {
        fn connect(&self, url: &str, bearer_token: &str) -> Result<Box<dyn DxTradeWebSocket>> {
            *self.connected_to.lock().unwrap() = Some((url.to_string(), bearer_token.to_string()));
            Ok(Box::new(FakeWebSocket {
                sent: self.sent.clone(),
                incoming: self.incoming.clone(),
                closed: self.closed.clone(),
            }))
        }
    }

    fn ok_text(s: &str) -> Result<String> {
        Ok(s.to_string())
    }

    #[test]
    fn subscribe_live_chart_opens_market_data_ws_with_dxapi_auth() {
        let factory = FakeWsFactory::with_incoming(vec![ok_text(
            r#"{"type":"Quote","payload":{"symbol":"EUR/USD","bid":1.08,"ask":1.0801,"timestamp":1716000000000}}"#,
        )]);
        let backend = ProductionDxTradeStreamingBackend::with_factory(factory.clone());

        let update = backend
            .subscribe_live_chart(&dummy_session(), "EUR/USD", "M1")
            .expect("subscribe ok");

        assert_eq!(update.symbol, "EUR/USD");
        assert_eq!(update.bid, Some(1.08));
        assert_eq!(update.ask, Some(1.0801));
        assert_eq!(update.timestamp_ms, Some(1716000000000));

        let (url, bearer) = factory.connected_url().expect("must have connected");
        assert_eq!(url, "wss://demo.dx.trade/dxsca-web/md?format=JSON");
        assert_eq!(bearer, "token");
        assert!(factory.was_closed(), "socket must be closed after read");
    }

    #[test]
    fn subscribe_live_chart_sends_market_data_subscription_with_quote_event_type() {
        let factory = FakeWsFactory::with_incoming(vec![ok_text(
            r#"{"type":"Quote","payload":{"symbol":"EUR/USD","bid":1.0,"ask":1.0001}}"#,
        )]);
        let backend = ProductionDxTradeStreamingBackend::with_factory(factory.clone());
        backend
            .subscribe_live_chart(&dummy_session(), "EUR/USD", "M1")
            .unwrap();

        let sent = factory.sent_frames();
        assert_eq!(sent.len(), 1);
        let body: serde_json::Value = serde_json::from_str(&sent[0]).unwrap();
        assert_eq!(body["type"], "MarketDataSubscriptionRequest");
        assert!(body["requestId"].as_str().unwrap().starts_with("req-"));
        assert_eq!(body["session"], "token");
        assert_eq!(body["payload"]["account"], "acc-1");
        assert_eq!(body["payload"]["symbols"], serde_json::json!(["EUR/USD"]));
        let events = body["payload"]["eventTypes"].as_array().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], "Quote");
        assert_eq!(events[0]["format"], "COMPACT");
    }

    #[test]
    fn subscribe_live_chart_skips_non_quote_frames_until_quote_arrives() {
        // Server may interleave subscription-ack frames and
        // unrelated event frames before the first quote.
        let factory = FakeWsFactory::with_incoming(vec![
            ok_text(r#"{"type":"SubscriptionAck","payload":{}}"#),
            ok_text(r#"{"type":"OrderUpdate","payload":{"orderId":1}}"#),
            ok_text(r#"{"type":"Quote","payload":{"symbol":"EUR/USD","bid":1.1,"ask":1.1001}}"#),
        ]);
        let backend = ProductionDxTradeStreamingBackend::with_factory(factory.clone());
        let update = backend
            .subscribe_live_chart(&dummy_session(), "EUR/USD", "M1")
            .expect("subscribe ok");
        assert_eq!(update.bid, Some(1.1));
    }

    #[test]
    fn subscribe_live_chart_skips_quote_for_other_symbol() {
        let factory = FakeWsFactory::with_incoming(vec![
            ok_text(r#"{"type":"Quote","payload":{"symbol":"GBP/USD","bid":1.25,"ask":1.2501}}"#),
            ok_text(r#"{"type":"Quote","payload":{"symbol":"EUR/USD","bid":1.08,"ask":1.0801}}"#),
        ]);
        let backend = ProductionDxTradeStreamingBackend::with_factory(factory.clone());
        let update = backend
            .subscribe_live_chart(&dummy_session(), "EUR/USD", "M1")
            .expect("subscribe ok");
        assert_eq!(update.bid, Some(1.08), "must skip the GBP/USD quote");
    }

    #[test]
    fn subscribe_live_chart_accepts_string_prices_for_broker_compat() {
        // Some brokers emit prices as JSON strings rather than
        // numbers; we tolerate both.
        let factory = FakeWsFactory::with_incoming(vec![ok_text(
            r#"{"type":"Quote","payload":{"symbol":"EUR/USD","bid":"1.08","ask":"1.0801"}}"#,
        )]);
        let backend = ProductionDxTradeStreamingBackend::with_factory(factory);
        let update = backend
            .subscribe_live_chart(&dummy_session(), "EUR/USD", "M1")
            .expect("subscribe ok");
        assert_eq!(update.bid, Some(1.08));
        assert_eq!(update.ask, Some(1.0801));
    }

    #[test]
    fn subscribe_live_chart_drops_unparseable_frames_without_failing() {
        let factory = FakeWsFactory::with_incoming(vec![
            ok_text("this is not JSON"),
            ok_text("{}"),
            ok_text(r#"{"type":"Quote","payload":{"symbol":"EUR/USD","bid":1.0,"ask":1.0001}}"#),
        ]);
        let backend = ProductionDxTradeStreamingBackend::with_factory(factory);
        let update = backend
            .subscribe_live_chart(&dummy_session(), "EUR/USD", "M1")
            .expect("subscribe ok");
        assert_eq!(update.bid, Some(1.0));
    }

    #[test]
    fn subscribe_live_chart_bails_when_frame_budget_exhausted_without_quote() {
        // Feed only non-Quote frames — must bail rather than
        // block forever.
        let mut frames: Vec<Result<String>> = vec![];
        for _ in 0..300 {
            frames.push(ok_text(r#"{"type":"SubscriptionAck","payload":{}}"#));
        }
        let factory = FakeWsFactory::with_incoming(frames);
        let backend = ProductionDxTradeStreamingBackend::with_factory(factory.clone());
        let err = backend
            .subscribe_live_chart(&dummy_session(), "EUR/USD", "M1")
            .expect_err("must bail");
        assert!(err.to_string().contains("no Quote"));
        assert!(factory.was_closed(), "socket must still close on bail");
    }

    #[test]
    fn subscribe_live_chart_rejects_empty_symbol() {
        let factory = FakeWsFactory::with_incoming(vec![]);
        let backend = ProductionDxTradeStreamingBackend::with_factory(factory.clone());
        let err = backend
            .subscribe_live_chart(&dummy_session(), "", "M1")
            .expect_err("must bail");
        assert!(err.to_string().contains("symbol"));
        assert!(
            factory.connected_url().is_none(),
            "must not open a socket when args are invalid"
        );
    }

    #[test]
    fn subscribe_live_chart_rejects_unauthenticated_session() {
        let factory = FakeWsFactory::with_incoming(vec![]);
        let backend = ProductionDxTradeStreamingBackend::with_factory(factory.clone());
        let mut session = dummy_session();
        session.session_token.clear();
        let err = backend
            .subscribe_live_chart(&session, "EUR/USD", "M1")
            .expect_err("must bail");
        assert!(err.to_string().contains("session_token"));
        assert!(factory.connected_url().is_none());
    }

    #[test]
    fn subscribe_live_chart_propagates_ws_read_error_through_anyhow_chain() {
        let factory =
            FakeWsFactory::with_incoming(vec![Err(anyhow::anyhow!("TLS handshake aborted"))]);
        let backend = ProductionDxTradeStreamingBackend::with_factory(factory);
        let err = backend
            .subscribe_live_chart(&dummy_session(), "EUR/USD", "M1")
            .expect_err("must bail");
        let chain: Vec<String> = err.chain().map(|c| c.to_string()).collect();
        assert!(chain.iter().any(|c| c.contains("read failed")), "{chain:?}");
        assert!(
            chain.iter().any(|c| c.contains("TLS handshake aborted")),
            "{chain:?}"
        );
    }

    // -- D3.3 pure-helper tests --------------------------------------------

    #[test]
    fn market_data_ws_url_rewrites_https_to_wss() {
        assert_eq!(
            market_data_ws_url("https://demo.dx.trade").unwrap(),
            "wss://demo.dx.trade/dxsca-web/md?format=JSON"
        );
    }

    #[test]
    fn market_data_ws_url_rewrites_http_to_ws() {
        assert_eq!(
            market_data_ws_url("http://localhost:8080").unwrap(),
            "ws://localhost:8080/dxsca-web/md?format=JSON"
        );
    }

    #[test]
    fn market_data_ws_url_strips_trailing_slash() {
        assert_eq!(
            market_data_ws_url("https://demo.dx.trade/").unwrap(),
            "wss://demo.dx.trade/dxsca-web/md?format=JSON"
        );
    }

    #[test]
    fn market_data_ws_url_rejects_empty_and_unscheme() {
        assert!(market_data_ws_url("").is_err());
        assert!(market_data_ws_url("demo.dx.trade").is_err());
    }

    #[test]
    fn json_to_f64_accepts_number_and_string_forms() {
        assert_eq!(json_to_f64(&serde_json::json!(1.5)), Some(1.5));
        assert_eq!(json_to_f64(&serde_json::json!("1.5")), Some(1.5));
        assert_eq!(json_to_f64(&serde_json::json!("  1.5  ")), Some(1.5));
        assert_eq!(json_to_f64(&serde_json::json!(null)), None);
        assert_eq!(json_to_f64(&serde_json::json!("nope")), None);
    }

    #[test]
    fn json_to_i64_ms_accepts_int_and_string_forms() {
        assert_eq!(
            json_to_i64_ms(&serde_json::json!(1234567890)),
            Some(1234567890)
        );
        assert_eq!(
            json_to_i64_ms(&serde_json::json!("1234567890")),
            Some(1234567890)
        );
        assert_eq!(json_to_i64_ms(&serde_json::json!(null)), None);
    }

    #[test]
    fn generate_request_id_is_unique_across_calls() {
        let n = 32;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..n {
            assert!(seen.insert(generate_request_id()));
        }
    }

    #[test]
    fn dxtrade_backend_production_bundle_constructs_cleanly() {
        // Construction must succeed — D3.1 is live, D3.2/D3.3
        // fail only when actually invoked. The chrome can build
        // the backend at startup without errors.
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
