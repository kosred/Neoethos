// Phase C3 audit + Flutter pivot context (2026-05-18 operator
// directive): this module is the cTrader Open API history fetcher.
// Its proto result parsers and `fetch_*` entry points are the
// API surface that:
//   - the (now-deferred) egui D2 wizard apply writer would have
//     consumed, AND
//   - the upcoming Flutter API layer (gRPC/REST endpoints
//     exposing position/order history to mobile + desktop
//     clients) WILL consume.
//
// Until the Flutter API stage lands, the helpers stay pub-but-
// uncalled. FILE-LOCAL allow only — NOT a workspace override.
#![allow(dead_code)]

//! High-level cTrader Open API history helpers.
//!
//! Wires the trade-history and historical-data capabilities the user
//! and bot need into a small set of `fetch_*` entry points that take
//! care of the standard `ProtoOAApplicationAuthReq` +
//! `ProtoOAAccountAuthReq` handshake before each query. Every helper
//! goes through the existing [`CTraderOpenApiTransport`] abstraction
//! so the same code path can be unit-tested against a captured fixture
//! (see the `#[ignore = "needs real-data fixture from cTrader"]` tests
//! at the bottom of the module).
//!
//! Operator directive 2026-05-15 (verbatim, in Greek):
//!
//! > "Το cTrader api πρέπει να μπορεί να δίνει όλες τις παροχές που έχει
//! >  ανάγκη ο χρήστης και το bot. Από την άλλη σύνδεση, κατέβασμα ιστορικών
//! >  δεδομένων, μέχρι ιστορικό των trades αν υπάρχει."
//!
//! Translation: "The cTrader API must be able to deliver everything the
//! user and bot need — from connection, downloading historical data,
//! all the way to trade history if available."
//!
//! Documentation cross-checked against:
//! - `docs/audits/research/ctrader_api_reference.md`
//! - `docs/audits/research/spotware_proto_new_messages.md`
//! - Upstream `https://raw.githubusercontent.com/spotware/openapi-proto-messages/master/OpenApiMessages.proto`

use crate::app_services::ctrader_account::{
    CTraderDealSnapshot, CTraderOrderDetailsSnapshot, CTraderPendingOrderSnapshot,
    CTraderSymbolCategorySnapshot, parse_deal_list_by_position_id_response,
    parse_deal_list_response, parse_order_details_response,
    parse_order_list_by_position_id_response, parse_symbol_category_list_response,
};
use crate::app_services::ctrader_data::{
    CTraderResolvedSymbol, CTraderSymbolLookupRequest, HistoricalBar, HistoricalBarsResult, HistoricalTicksResult, parse_tick_data_response, parse_trendbars_response,
    resolve_symbol_with_transport,
};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_DEAL_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_ORDER_DETAILS_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_ORDER_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_SYMBOL_CATEGORY_RESPONSE_PAYLOAD_TYPE, CTRADER_QUOTE_TYPE_ASK,
    CTRADER_QUOTE_TYPE_BID, CTraderDealListRequest, CTraderOpenApiTransport,
    ProductionCTraderOpenApiTransport, build_account_auth_request, build_application_auth_request,
    build_deal_list_by_position_id_request, build_deal_list_request, build_get_tick_data_request,
    build_get_trendbars_request, build_order_details_request,
    build_order_list_by_position_id_request, build_symbol_category_list_request,
    parse_ctrader_error_payload, parse_open_api_envelope, trendbar_period_value,
};
use anyhow::{Context, Result, anyhow};

/// Default cap on a single `ProtoOADealListReq` page when the caller
/// does not pass an explicit `max_rows`. Mirrors
/// `DEFAULT_CTRADER_DEAL_MAX_ROWS` in `ctrader_account.rs` so the two
/// surfaces agree on the per-request budget; bumping the constant in
/// one place therefore does not silently shift the other. The value
/// has to be small enough that one round-trip fits comfortably under
/// the broker's documented response-size ceiling (the Spotware Open
/// API help-centre does not publish a hard cap, but community ports
/// converge on ~1000 deals per page as a safe upper bound).
pub const DEFAULT_DEAL_HISTORY_PAGE_MAX_ROWS: i32 = 1000;

/// Sentinel that signals a fully open lower bound to
/// `ProtoOADealListReq`. The proto declares `fromTimestamp` as
/// optional, so the caller can omit it altogether — but several
/// caller surfaces wanted a single function signature that always
/// takes a window, so passing `None` here is also accepted by every
/// `fetch_*` helper below.
pub const NO_LOWER_TIMESTAMP_BOUND: Option<i64> = None;

/// Request shape for the canonical "give me my trade history" call.
/// The window is bounded by `from_timestamp_ms` / `to_timestamp_ms`
/// inclusive on both sides — passing `None` for `from` removes the
/// lower bound. The broker silently caps the response, so callers
/// must inspect [`CTraderDealHistorySnapshot::has_more`] and
/// re-issue with a tighter window when more rows remain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderDealHistoryRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
    pub account_id: String,
    pub from_timestamp_ms: Option<i64>,
    pub to_timestamp_ms: Option<i64>,
    pub max_rows: Option<i32>,
}

/// Response from [`fetch_deal_history`].
#[derive(Debug, Clone, PartialEq)]
pub struct CTraderDealHistorySnapshot {
    pub account_id: i64,
    pub deals: Vec<CTraderDealSnapshot>,
    /// `true` when the broker truncated the response — the caller must
    /// either widen `max_rows`, narrow the time window, or paginate by
    /// shrinking `to_timestamp_ms` to the oldest returned deal.
    /// Sourced from the broker's own `hasMore` field on
    /// `ProtoOADealListRes`; we do NOT synthesise it.
    pub has_more: bool,
    /// Set when the broker returned fewer rows than `max_rows` despite
    /// reporting `hasMore = true`, or when at least one deal lay
    /// outside the requested window. Empty when nothing surprising
    /// happened. Mirrors the `warnings` field on
    /// [`HistoricalBarsResult`] so callers have a single audit channel.
    pub warnings: Vec<String>,
}

/// Fetch trade history (`ProtoOADealListReq`) for an account.
///
/// Operator directive 2026-05-15 explicitly names this capability:
/// *"ιστορικό των trades αν υπάρχει"* ("trade history if available").
///
/// The function:
///
/// 1. Runs the standard `ProtoOAApplicationAuthReq` +
///    `ProtoOAAccountAuthReq` handshake on the supplied transport.
/// 2. Issues a single `ProtoOADealListReq` covering the requested
///    window. cTrader does NOT publish a documented page-size ceiling
///    on this message; we cap at [`DEFAULT_DEAL_HISTORY_PAGE_MAX_ROWS`]
///    (1000) when the caller does not supply one.
/// 3. Validates that every returned deal's `executionTimestamp` falls
///    inside `[from_timestamp_ms, to_timestamp_ms]` — a deal outside
///    the window indicates a broker bug or a clock-skew issue and
///    must NOT be silently consumed (defends backtests / journal
///    reconciliation from corrupted history). When any deal is out of
///    range we still return the in-range subset but record a
///    `warnings[]` entry.
/// 4. Warns when the broker reports `hasMore = true` so the caller
///    can paginate; we deliberately do NOT silently auto-paginate
///    because the operator's prior directives forbid silent
///    truncation either way.
pub fn fetch_deal_history_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderDealHistoryRequest,
) -> Result<CTraderDealHistorySnapshot> {
    let account_id = request
        .account_id
        .parse::<i64>()
        .context("cTrader account id must be numeric")?;
    let max_rows = request
        .max_rows
        .unwrap_or(DEFAULT_DEAL_HISTORY_PAGE_MAX_ROWS);
    tracing::info!(
        target: "forex_app::ctrader_history",
        account_id,
        from_ms = request.from_timestamp_ms,
        to_ms = request.to_timestamp_ms,
        max_rows,
        "ctrader_history fetch_deal_history requesting ProtoOADealListReq"
    );

    let responses = transport.send_sequence(&[
        build_application_auth_request(
            &request.client_id,
            &request.client_secret,
            "history-app-auth-1",
        ),
        build_account_auth_request(account_id, &request.access_token, "history-account-auth-1"),
        build_deal_list_request(
            &CTraderDealListRequest {
                account_id,
                from_timestamp_ms: request.from_timestamp_ms,
                to_timestamp_ms: request.to_timestamp_ms,
                max_rows: Some(max_rows),
            },
            "history-deals-1",
        ),
    ])?;
    if responses.len() != 3 {
        return Err(anyhow!(
            "expected 3 cTrader deal-history responses, received {}",
            responses.len()
        ));
    }
    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(&responses[2], CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE)?;

    let has_more = response_has_more_flag(&responses[2]).unwrap_or(false);
    let raw_deals = parse_deal_list_response(&responses[2])?;
    let (deals, warnings) = clamp_deals_to_window(
        raw_deals,
        request.from_timestamp_ms,
        request.to_timestamp_ms,
        max_rows,
        has_more,
    );

    Ok(CTraderDealHistorySnapshot {
        account_id,
        deals,
        has_more,
        warnings,
    })
}

/// Convenience wrapper that opens a `ProductionCTraderOpenApiTransport`
/// against the request's environment and delegates to
/// [`fetch_deal_history_with_transport`].
pub fn fetch_deal_history(
    request: &CTraderDealHistoryRequest,
) -> Result<CTraderDealHistorySnapshot> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    fetch_deal_history_with_transport(&transport, request)
}

/// Request shape for [`fetch_historical_bars`]. `timeframe` is the
/// canonical label (M1/M3/M5/M15/M30/H1/H4/H12/D1/W1/MN1); any other
/// value is rejected — including H2, which is not in
/// [`forex_core::CANONICAL_TIMEFRAMES`] and not natively supported by
/// the cTrader `ProtoOATrendbarPeriod` enum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderHistoricalBarsRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
    pub account_id: String,
    pub symbol_name: String,
    pub timeframe: String,
    pub from_timestamp_ms: i64,
    pub to_timestamp_ms: i64,
    pub count: Option<u32>,
}

/// Fetch historical OHLCV bars (`ProtoOAGetTrendbarsReq`) for a symbol.
///
/// Returns the symbol metadata (already converted to f64 prices) plus
/// the warnings vector that signals out-of-range bars or
/// broker-side truncation (`hasMore = true`). All timestamps are
/// validated against the requested window before returning.
pub fn fetch_historical_bars_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderHistoricalBarsRequest,
) -> Result<HistoricalBarsResult> {
    let _ = trendbar_period_value(&request.timeframe).with_context(|| {
        format!(
            "rejected cTrader trendbars request: non-canonical timeframe {}",
            request.timeframe
        )
    })?;
    if !forex_core::is_canonical_timeframe(&request.timeframe) {
        return Err(anyhow!(
            "rejected cTrader trendbars request: timeframe {} is outside the canonical 11-timeframe set",
            request.timeframe
        ));
    }
    if request.from_timestamp_ms > request.to_timestamp_ms {
        return Err(anyhow!(
            "invalid cTrader trendbars window: from_ms {} > to_ms {}",
            request.from_timestamp_ms,
            request.to_timestamp_ms
        ));
    }
    tracing::info!(
        target: "forex_app::ctrader_history",
        symbol = %request.symbol_name,
        timeframe = %request.timeframe,
        from_ms = request.from_timestamp_ms,
        to_ms = request.to_timestamp_ms,
        count = ?request.count,
        "ctrader_history fetch_historical_bars requesting ProtoOAGetTrendbarsReq"
    );
    let resolved = resolve_symbol_with_transport(
        transport,
        &CTraderSymbolLookupRequest {
            client_id: request.client_id.clone(),
            client_secret: request.client_secret.clone(),
            access_token: request.access_token.clone(),
            environment: request.environment,
            account_id: request.account_id.clone(),
            symbol_name: request.symbol_name.clone(),
        },
    )?;
    let period = trendbar_period_value(&request.timeframe)?;
    // v0.5.1.1 — re-authenticate on this fresh WSS connection before
    // the trendbars request. `send_sequence` opens a brand-new socket
    // each call so the previous app/account auth from `resolve_symbol`
    // does not carry over; without re-auth here cTrader returns
    // `ProtoOAErrorRes` (payloadType 2142) for the trendbars request.
    let responses = transport.send_sequence(&[
        build_application_auth_request(&request.client_id, &request.client_secret, "app-auth-3"),
        build_account_auth_request(resolved.account_id, &request.access_token, "account-auth-3"),
        build_get_trendbars_request(
            resolved.account_id,
            resolved.light_symbol.symbol_id,
            period,
            request.from_timestamp_ms,
            request.to_timestamp_ms,
            request.count,
            "history-bars-1",
        ),
    ])?;
    if responses.len() < 3 {
        for response in &responses {
            let envelope = parse_open_api_envelope(response)?;
            if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                return Err(anyhow!(
                    "cTrader trendbars sequence failed (step {}): {}",
                    responses.len(),
                    parse_ctrader_error_payload(&envelope.payload)?
                ));
            }
        }
        return Err(anyhow!(
            "expected 3 cTrader trendbars auth/data responses, received {}",
            responses.len()
        ));
    }
    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(
        &responses[2],
        CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE,
    )?;
    let mut bars = parse_trendbars_response(&responses[2], &resolved.symbol)?;

    let warnings = clamp_bars_to_window(
        &mut bars.bars,
        request.from_timestamp_ms,
        request.to_timestamp_ms,
        request.count,
        bars.has_more,
    );
    bars.warnings.extend(warnings);
    Ok(bars)
}

/// Production wrapper for [`fetch_historical_bars_with_transport`].
pub fn fetch_historical_bars(
    request: &CTraderHistoricalBarsRequest,
) -> Result<HistoricalBarsResult> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    fetch_historical_bars_with_transport(&transport, request)
}

/// Quote side for `ProtoOAGetTickDataReq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTraderQuoteSide {
    Bid,
    Ask,
}

impl CTraderQuoteSide {
    fn as_i32(self) -> i32 {
        match self {
            Self::Bid => CTRADER_QUOTE_TYPE_BID,
            Self::Ask => CTRADER_QUOTE_TYPE_ASK,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Bid => "BID",
            Self::Ask => "ASK",
        }
    }
}

/// Request shape for [`fetch_tick_data`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderTickDataRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
    pub account_id: String,
    pub symbol_name: String,
    pub quote_side: CTraderQuoteSide,
    pub from_timestamp_ms: i64,
    pub to_timestamp_ms: i64,
}

/// Fetch high-resolution tick data (`ProtoOAGetTickDataReq`) for a
/// symbol. Useful for backtest precision (the trendbars-only path
/// snaps to the smallest native period). Validates that every
/// returned tick falls inside the requested window.
pub fn fetch_tick_data_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderTickDataRequest,
) -> Result<HistoricalTicksResult> {
    if request.from_timestamp_ms > request.to_timestamp_ms {
        return Err(anyhow!(
            "invalid cTrader tick window: from_ms {} > to_ms {}",
            request.from_timestamp_ms,
            request.to_timestamp_ms
        ));
    }
    tracing::info!(
        target: "forex_app::ctrader_history",
        symbol = %request.symbol_name,
        side = request.quote_side.label(),
        from_ms = request.from_timestamp_ms,
        to_ms = request.to_timestamp_ms,
        "ctrader_history fetch_tick_data requesting ProtoOAGetTickDataReq"
    );
    let resolved = resolve_symbol_with_transport(
        transport,
        &CTraderSymbolLookupRequest {
            client_id: request.client_id.clone(),
            client_secret: request.client_secret.clone(),
            access_token: request.access_token.clone(),
            environment: request.environment,
            account_id: request.account_id.clone(),
            symbol_name: request.symbol_name.clone(),
        },
    )?;
    let responses = transport.send_sequence(&[build_get_tick_data_request(
        resolved.account_id,
        resolved.light_symbol.symbol_id,
        request.quote_side.as_i32(),
        request.from_timestamp_ms,
        request.to_timestamp_ms,
        "history-ticks-1",
    )])?;
    if responses.len() != 1 {
        return Err(anyhow!(
            "expected 1 cTrader tick-data response, received {}",
            responses.len()
        ));
    }
    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE,
    )?;
    let result = parse_tick_data_response(&responses[0], &resolved.symbol)?;
    validate_tick_window(
        &result,
        request.from_timestamp_ms,
        request.to_timestamp_ms,
        &request.symbol_name,
    );
    Ok(result)
}

/// Production wrapper for [`fetch_tick_data_with_transport`].
pub fn fetch_tick_data(request: &CTraderTickDataRequest) -> Result<HistoricalTicksResult> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    fetch_tick_data_with_transport(&transport, request)
}

/// Resolve a symbol against the broker (used for the by-position
/// helpers below, which take a symbol-agnostic `positionId`).
///
/// Re-exported helper so the trading-side modules don't have to depend
/// on `ctrader_data` directly.
pub fn resolve_symbol(request: &CTraderSymbolLookupRequest) -> Result<CTraderResolvedSymbol> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    resolve_symbol_with_transport(&transport, request)
}

/// Request shape for the per-position lookups (deals + orders).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderPositionLookupRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
    pub account_id: String,
    pub position_id: i64,
    pub from_timestamp_ms: Option<i64>,
    pub to_timestamp_ms: Option<i64>,
}

pub trait CTraderPositionOrderHistoryBackend: Send + Sync {
    fn fetch_orders_by_position_id(
        &self,
        request: &CTraderPositionLookupRequest,
    ) -> Result<Vec<CTraderPendingOrderSnapshot>>;
}

#[derive(Debug, Default)]
pub struct ProductionCTraderPositionOrderHistoryBackend;

#[cfg(test)]
pub struct StubCTraderPositionOrderHistoryBackend {
    outcome: std::sync::Arc<
        std::sync::Mutex<Option<std::result::Result<Vec<CTraderPendingOrderSnapshot>, String>>>,
    >,
    last_request: std::sync::Arc<std::sync::Mutex<Option<CTraderPositionLookupRequest>>>,
}

/// Fetch every deal tied to a single `positionId`
/// (`ProtoOADealListByPositionIdReq`, payload type 2179). New in the
/// 2026-05-14 upstream proto refresh.
pub fn fetch_deals_by_position_id_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderPositionLookupRequest,
) -> Result<Vec<CTraderDealSnapshot>> {
    let account_id = request
        .account_id
        .parse::<i64>()
        .context("cTrader account id must be numeric")?;
    tracing::info!(
        target: "forex_app::ctrader_history",
        account_id,
        position_id = request.position_id,
        from_ms = request.from_timestamp_ms,
        to_ms = request.to_timestamp_ms,
        "ctrader_history fetch_deals_by_position_id requesting ProtoOADealListByPositionIdReq"
    );
    let responses = transport.send_sequence(&[
        build_application_auth_request(
            &request.client_id,
            &request.client_secret,
            "pos-deals-app-auth-1",
        ),
        build_account_auth_request(
            account_id,
            &request.access_token,
            "pos-deals-account-auth-1",
        ),
        build_deal_list_by_position_id_request(
            account_id,
            request.position_id,
            request.from_timestamp_ms,
            request.to_timestamp_ms,
            "pos-deals-1",
        ),
    ])?;
    if responses.len() != 3 {
        return Err(anyhow!(
            "expected 3 cTrader deal-list-by-position responses, received {}",
            responses.len()
        ));
    }
    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(
        &responses[2],
        CTRADER_OA_DEAL_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE,
    )?;
    let raw_deals = parse_deal_list_by_position_id_response(&responses[2])?;
    let max_rows = i32::MAX; // no max_rows on this message — emit no truncation warning
    let (deals, warnings) = clamp_deals_to_window(
        raw_deals,
        request.from_timestamp_ms,
        request.to_timestamp_ms,
        max_rows,
        false,
    );
    for warning in &warnings {
        tracing::warn!(
            target: "forex_app::ctrader_history",
            account_id,
            position_id = request.position_id,
            warning = warning.as_str(),
            "ctrader_history fetch_deals_by_position_id reported a warning"
        );
    }
    Ok(deals)
}

/// Production wrapper for [`fetch_deals_by_position_id_with_transport`].
pub fn fetch_deals_by_position_id(
    request: &CTraderPositionLookupRequest,
) -> Result<Vec<CTraderDealSnapshot>> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    fetch_deals_by_position_id_with_transport(&transport, request)
}

/// Fetch every order tied to a single `positionId`
/// (`ProtoOAOrderListByPositionIdReq`, payload type 2183). New in the
/// 2026-05-14 upstream proto refresh.
pub fn fetch_orders_by_position_id_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderPositionLookupRequest,
) -> Result<Vec<CTraderPendingOrderSnapshot>> {
    let account_id = request
        .account_id
        .parse::<i64>()
        .context("cTrader account id must be numeric")?;
    tracing::info!(
        target: "forex_app::ctrader_history",
        account_id,
        position_id = request.position_id,
        from_ms = request.from_timestamp_ms,
        to_ms = request.to_timestamp_ms,
        "ctrader_history fetch_orders_by_position_id requesting ProtoOAOrderListByPositionIdReq"
    );
    let responses = transport.send_sequence(&[
        build_application_auth_request(
            &request.client_id,
            &request.client_secret,
            "pos-orders-app-auth-1",
        ),
        build_account_auth_request(
            account_id,
            &request.access_token,
            "pos-orders-account-auth-1",
        ),
        build_order_list_by_position_id_request(
            account_id,
            request.position_id,
            request.from_timestamp_ms,
            request.to_timestamp_ms,
            "pos-orders-1",
        ),
    ])?;
    if responses.len() != 3 {
        return Err(anyhow!(
            "expected 3 cTrader order-list-by-position responses, received {}",
            responses.len()
        ));
    }
    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(
        &responses[2],
        CTRADER_OA_ORDER_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE,
    )?;
    let orders = parse_order_list_by_position_id_response(&responses[2])?;
    Ok(filter_orders_to_window(
        orders,
        request.from_timestamp_ms,
        request.to_timestamp_ms,
    ))
}

/// Production wrapper for [`fetch_orders_by_position_id_with_transport`].
pub fn fetch_orders_by_position_id(
    request: &CTraderPositionLookupRequest,
) -> Result<Vec<CTraderPendingOrderSnapshot>> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    fetch_orders_by_position_id_with_transport(&transport, request)
}

impl CTraderPositionOrderHistoryBackend for ProductionCTraderPositionOrderHistoryBackend {
    fn fetch_orders_by_position_id(
        &self,
        request: &CTraderPositionLookupRequest,
    ) -> Result<Vec<CTraderPendingOrderSnapshot>> {
        fetch_orders_by_position_id(request)
    }
}

#[cfg(test)]
impl StubCTraderPositionOrderHistoryBackend {
    pub fn success(orders: Vec<CTraderPendingOrderSnapshot>) -> Self {
        Self {
            outcome: std::sync::Arc::new(std::sync::Mutex::new(Some(Ok(orders)))),
            last_request: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }
}

#[cfg(test)]
impl CTraderPositionOrderHistoryBackend for StubCTraderPositionOrderHistoryBackend {
    fn fetch_orders_by_position_id(
        &self,
        request: &CTraderPositionLookupRequest,
    ) -> Result<Vec<CTraderPendingOrderSnapshot>> {
        *self.last_request.lock().expect("last request lock") = Some(request.clone());
        self.outcome
            .lock()
            .expect("position order history outcome lock")
            .take()
            .unwrap_or_else(|| {
                Err("stub cTrader position order history backend exhausted".to_string())
            })
            .map_err(anyhow::Error::msg)
    }
}

/// Request shape for [`fetch_order_details`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderOrderDetailsRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
    pub account_id: String,
    pub order_id: i64,
}

/// Fetch a single order + all of its child deals
/// (`ProtoOAOrderDetailsReq`, payload type 2181). New in the
/// 2026-05-14 upstream proto refresh.
pub fn fetch_order_details_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderOrderDetailsRequest,
) -> Result<CTraderOrderDetailsSnapshot> {
    let account_id = request
        .account_id
        .parse::<i64>()
        .context("cTrader account id must be numeric")?;
    tracing::info!(
        target: "forex_app::ctrader_history",
        account_id,
        order_id = request.order_id,
        "ctrader_history fetch_order_details requesting ProtoOAOrderDetailsReq"
    );
    let responses = transport.send_sequence(&[
        build_application_auth_request(
            &request.client_id,
            &request.client_secret,
            "order-details-app-auth-1",
        ),
        build_account_auth_request(
            account_id,
            &request.access_token,
            "order-details-account-auth-1",
        ),
        build_order_details_request(account_id, request.order_id, "order-details-1"),
    ])?;
    if responses.len() != 3 {
        return Err(anyhow!(
            "expected 3 cTrader order-details responses, received {}",
            responses.len()
        ));
    }
    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(
        &responses[2],
        CTRADER_OA_ORDER_DETAILS_RESPONSE_PAYLOAD_TYPE,
    )?;
    parse_order_details_response(&responses[2])
}

/// Production wrapper for [`fetch_order_details_with_transport`].
pub fn fetch_order_details(
    request: &CTraderOrderDetailsRequest,
) -> Result<CTraderOrderDetailsSnapshot> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    fetch_order_details_with_transport(&transport, request)
}

/// Request shape for [`fetch_symbol_categories`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderSymbolCategoriesRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
    pub account_id: String,
}

/// Fetch the symbol-category taxonomy (`ProtoOASymbolCategoryListReq`,
/// payload type 2160). New in the 2026-05-14 upstream proto refresh.
pub fn fetch_symbol_categories_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderSymbolCategoriesRequest,
) -> Result<Vec<CTraderSymbolCategorySnapshot>> {
    let account_id = request
        .account_id
        .parse::<i64>()
        .context("cTrader account id must be numeric")?;
    tracing::info!(
        target: "forex_app::ctrader_history",
        account_id,
        "ctrader_history fetch_symbol_categories requesting ProtoOASymbolCategoryListReq"
    );
    let responses = transport.send_sequence(&[
        build_application_auth_request(
            &request.client_id,
            &request.client_secret,
            "categories-app-auth-1",
        ),
        build_account_auth_request(
            account_id,
            &request.access_token,
            "categories-account-auth-1",
        ),
        build_symbol_category_list_request(account_id, "categories-1"),
    ])?;
    if responses.len() != 3 {
        return Err(anyhow!(
            "expected 3 cTrader symbol-category responses, received {}",
            responses.len()
        ));
    }
    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(
        &responses[2],
        CTRADER_OA_SYMBOL_CATEGORY_RESPONSE_PAYLOAD_TYPE,
    )?;
    parse_symbol_category_list_response(&responses[2])
}

/// Production wrapper for [`fetch_symbol_categories_with_transport`].
pub fn fetch_symbol_categories(
    request: &CTraderSymbolCategoriesRequest,
) -> Result<Vec<CTraderSymbolCategorySnapshot>> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    fetch_symbol_categories_with_transport(&transport, request)
}

// -- Helpers ---------------------------------------------------------

fn ensure_success_payload_type(response_json: &str, expected_payload_type: u32) -> Result<()> {
    let envelope = parse_open_api_envelope(response_json)?;
    if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "cTrader history request failed: {}",
            parse_ctrader_error_payload(&envelope.payload)
                .context("failed to format cTrader error payload")?
        ));
    }
    if envelope.payload_type != expected_payload_type {
        return Err(anyhow!(
            "unexpected cTrader history payload type: expected {}, got {}",
            expected_payload_type,
            envelope.payload_type
        ));
    }
    Ok(())
}

fn response_has_more_flag(response_json: &str) -> Option<bool> {
    let envelope = parse_open_api_envelope(response_json).ok()?;
    envelope.payload.get("hasMore").and_then(|v| v.as_bool())
}

/// Filter deals to those inside `[from, to]`. Out-of-range deals are
/// dropped and recorded in `warnings`. The function also adds a
/// truncation warning when `has_more = true` so the caller knows to
/// paginate.
fn clamp_deals_to_window(
    deals: Vec<CTraderDealSnapshot>,
    from_ms: Option<i64>,
    to_ms: Option<i64>,
    max_rows: i32,
    has_more: bool,
) -> (Vec<CTraderDealSnapshot>, Vec<String>) {
    let row_count_before = deals.len();
    let mut warnings = Vec::new();
    let mut in_range = Vec::with_capacity(deals.len());
    for deal in deals {
        let ts = deal.execution_timestamp_ms;
        if let Some(from) = from_ms {
            if ts < from {
                warnings.push(format!(
                    "deal {} dropped: executionTimestamp {} < requested from {}",
                    deal.deal_id, ts, from
                ));
                continue;
            }
        }
        if let Some(to) = to_ms {
            if ts > to {
                warnings.push(format!(
                    "deal {} dropped: executionTimestamp {} > requested to {}",
                    deal.deal_id, ts, to
                ));
                continue;
            }
        }
        in_range.push(deal);
    }
    if has_more {
        warnings.push(format!(
            "broker reported hasMore = true after returning {} deals (max_rows = {}); \
             caller must paginate by lowering the window's upper bound",
            in_range.len(),
            max_rows
        ));
    } else if row_count_before == 0 && max_rows > 0 {
        // No truncation; nothing to record.
    } else if (row_count_before as i32) >= max_rows {
        // The broker filled the page exactly; we cannot tell if more
        // existed without `hasMore = true`, so we surface a soft note
        // rather than a warn so the caller can decide.
        warnings.push(format!(
            "broker returned exactly max_rows={} deals without hasMore flag — \
             paginate just in case",
            max_rows
        ));
    }
    for warning in &warnings {
        tracing::warn!(
            target: "forex_app::ctrader_history",
            warning = warning.as_str(),
            "ctrader_history fetch_deal_history surfaced a warning"
        );
    }
    (in_range, warnings)
}

fn clamp_bars_to_window(
    bars: &mut Vec<HistoricalBar>,
    from_ms: i64,
    to_ms: i64,
    requested_count: Option<u32>,
    has_more: bool,
) -> Vec<String> {
    let row_count_before = bars.len();
    let mut warnings = Vec::new();
    bars.retain(|bar| {
        let ts = bar.timestamp_ms;
        if ts < from_ms {
            warnings.push(format!(
                "bar dropped: timestamp {} < requested from {}",
                ts, from_ms
            ));
            return false;
        }
        if ts > to_ms {
            warnings.push(format!(
                "bar dropped: timestamp {} > requested to {}",
                ts, to_ms
            ));
            return false;
        }
        true
    });
    if has_more {
        warnings.push(format!(
            "broker reported hasMore = true after returning {} bars; caller must paginate",
            bars.len()
        ));
    }
    if let Some(count) = requested_count {
        if (row_count_before as u32) < count {
            warnings.push(format!(
                "broker returned {} bars but caller asked for {} — possible weekend/holiday gap",
                row_count_before, count
            ));
        }
    }
    for warning in &warnings {
        tracing::warn!(
            target: "forex_app::ctrader_history",
            warning = warning.as_str(),
            "ctrader_history fetch_historical_bars surfaced a warning"
        );
    }
    warnings
}

fn validate_tick_window(result: &HistoricalTicksResult, from_ms: i64, to_ms: i64, symbol: &str) {
    for tick in &result.ticks {
        if tick.timestamp_ms < from_ms || tick.timestamp_ms > to_ms {
            tracing::warn!(
                target: "forex_app::ctrader_history",
                symbol = %symbol,
                tick_ts = tick.timestamp_ms,
                from_ms,
                to_ms,
                "ctrader_history fetch_tick_data observed a tick outside the requested window"
            );
        }
    }
    if result.has_more {
        tracing::warn!(
            target: "forex_app::ctrader_history",
            symbol = %symbol,
            from_ms,
            to_ms,
            "ctrader_history fetch_tick_data: broker reported hasMore = true; caller must paginate"
        );
    }
}

fn filter_orders_to_window(
    orders: Vec<CTraderPendingOrderSnapshot>,
    from_ms: Option<i64>,
    to_ms: Option<i64>,
) -> Vec<CTraderPendingOrderSnapshot> {
    if from_ms.is_none() && to_ms.is_none() {
        return orders;
    }
    orders
        .into_iter()
        .filter(|order| {
            let Some(ts) = order.open_timestamp_ms else {
                // Missing timestamp — keep it; the broker did not give
                // us anything to filter on. Logged at debug so an
                // operator can still see the case.
                tracing::debug!(
                    target: "forex_app::ctrader_history",
                    order_id = order.order_id,
                    "ctrader_history filter_orders_to_window: order lacks open_timestamp_ms; keeping"
                );
                return true;
            };
            if let Some(from) = from_ms {
                if ts < from {
                    tracing::warn!(
                        target: "forex_app::ctrader_history",
                        order_id = order.order_id,
                        order_ts = ts,
                        from_ms = from,
                        "ctrader_history filter_orders_to_window: order dropped (older than requested window)"
                    );
                    return false;
                }
            }
            if let Some(to) = to_ms {
                if ts > to {
                    tracing::warn!(
                        target: "forex_app::ctrader_history",
                        order_id = order.order_id,
                        order_ts = ts,
                        to_ms = to,
                        "ctrader_history filter_orders_to_window: order dropped (newer than requested window)"
                    );
                    return false;
                }
            }
            true
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! All integration-style tests in this module are guarded by
    //! `#[ignore = "needs real-data fixture from cTrader"]` per the
    //! repo-wide rule against synthetic broker data. To run them
    //! offline, capture a real `ProtoOA*Res` JSON envelope from the
    //! demo broker and stash it under `tests/fixtures/ctrader/`, then
    //! point the test at it. We deliberately do NOT vendor sample
    //! envelopes here so a future agent does not lift placeholder
    //! numbers into a unit test and inadvertently encode them as
    //! "expected".
    //!
    //! TODO(real-data): add a captured `ProtoOADealListRes` fixture so
    //! the window-clamping logic in [`clamp_deals_to_window`] can be
    //! exercised against bytes the broker actually emitted (rather
    //! than a hand-rolled JSON literal that drifts from the proto).

    use super::*;

    #[test]
    #[ignore = "needs real-data fixture from cTrader"]
    fn fetch_deal_history_clamps_out_of_range_real_fixture() {
        // Placeholder for the real-fixture test. Intentionally empty;
        // the operator's policy is "no synthetic data, even in tests"
        // (verbatim, prior sessions). The body lands once the fixture
        // file is committed under `tests/fixtures/ctrader/`.
    }

    #[test]
    #[ignore = "needs real-data fixture from cTrader"]
    fn fetch_historical_bars_warns_on_weekend_gap_real_fixture() {
        // Placeholder — same policy as above.
    }

    /// Verify the server-side per-position drill-down
    /// (`ProtoOAOrderListByPositionIdReq`, payload 2183) — the helper
    /// must return the broker's response verbatim with NO client-side
    /// re-filter. The capture procedure + expected file name live in
    /// `crates/forex-app/tests/fixtures/ctrader/order_list_by_position_id/README.md`.
    /// Wired into:
    /// - `trading/orders.rs::execute_ctrader_request`
    ///   (post-execution drill-down on success).
    /// - `trading/orders.rs::cancel_selected_order` (pre-cancel
    ///   `fetch_order_details` lookup).
    /// - `trading/orders.rs::close_selected_position` (pre-close
    ///   `fetch_orders_by_position_id` lookup).
    /// See `docs/audits/research/ctrader_api_full_reference.md`
    /// Appendix C item #5 for the rationale.
    #[test]
    #[ignore = "needs cTrader fixture"]
    fn fetch_orders_by_position_id_clamps_to_position_real_fixture() {
        // Placeholder — same policy as the other ignored fixture-tests
        // above (operator directive 2026-05-15: no synthetic broker
        // data, even in tests). Body lands once the captured
        // `position_full_chain.json` is committed under
        // `tests/fixtures/ctrader/order_list_by_position_id/`.
        //
        // The expected assertions (locked in by the README):
        // 1. `fetch_orders_by_position_id_with_transport` parses the
        //    captured `ProtoOAOrderListByPositionIdRes` envelope.
        // 2. Every returned order's `tradeData.positionId` matches
        //    the requested `position_id` — proves the broker did the
        //    filtering (no client-side re-filter).
        // 3. The `from_timestamp_ms` / `to_timestamp_ms` clamp via
        //    `filter_orders_to_window` keeps the broker's response
        //    intact when the window is `None`.
    }

    #[test]
    fn rejects_h2_historical_bars_request() {
        // No transport call; this should fail before any network I/O
        // because H2 is not in `forex_core::CANONICAL_TIMEFRAMES` and
        // not in cTrader's `ProtoOATrendbarPeriod` enum.
        let request = CTraderHistoricalBarsRequest {
            client_id: "cid".into(),
            client_secret: "csec".into(),
            access_token: "tok".into(),
            environment: CTraderEnvironment::Demo,
            account_id: "1".into(),
            symbol_name: "EURUSD".into(),
            timeframe: "H2".into(),
            from_timestamp_ms: 0,
            to_timestamp_ms: 1,
            count: None,
        };
        let err = fetch_historical_bars_with_transport(&FailingTransport, &request)
            .expect_err("H2 must be rejected before network I/O");
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("H2")
                || msg.to_ascii_lowercase().contains("canonical")
                || msg.to_ascii_lowercase().contains("trendbar"),
            "unexpected error message: {msg}"
        );
    }

    struct FailingTransport;
    impl CTraderOpenApiTransport for FailingTransport {
        fn send_sequence(
            &self,
            _messages: &[crate::app_services::ctrader_messages::CTraderOpenApiJsonMessage],
        ) -> Result<Vec<String>> {
            Err(anyhow!(
                "FailingTransport: transport.send_sequence must not be reached for an invalid timeframe"
            ))
        }
    }
}
