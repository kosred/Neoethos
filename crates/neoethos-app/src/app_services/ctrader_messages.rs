// Phase C3 audit + Flutter pivot context (2026-05-18 operator
// directive): this file is the Spotware Open API proto wire-format
// surface. Every `ProtoOA*` variant exists because Spotware's spec
// requires it — a missing variant breaks the protocol parser. The
// "unused" variants (ProtoOAGetCtidProfileByTokenReq, the
// bonus-deposit-withdraw lifecycle, dealOffsetList, etc.) are
// SPEC-COMPLETE wire format kept for upcoming consumers:
//   - cTrader history / account-profile RPCs that the Flutter API
//     layer will expose
//   - position lifecycle events the manual-trading mobile UI needs
//
// FILE-LOCAL allow only — NOT a workspace lint override.
#![allow(dead_code)]

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tungstenite::{Message, connect};

/// Environment-variable name that selects the cTrader Open API transport
/// at runtime. Recognised values (case-insensitive, trimmed):
///
/// * `json_wss` (default) → port 5036, WebSocket+TLS+JSON envelopes.
/// * `protobuf` → port 5035, raw TCP+TLS with native Protobuf framing
///   (length-prefix + serialised `ProtoMessage`). Migrates the reconcile
///   + historical-bars endpoints per the v0.4.5 batch documented in
///   `docs/audits/research/ctrader_api_full_reference.md` §10 item #3.
///
/// Unset → default (`json_wss`). Unknown value → JSON-WSS with a warn-
/// level trace event (the binary stays usable even if the operator
/// typoes the value).
pub const CTRADER_TRANSPORT_ENV_VAR: &str = "NEOETHOS_BOT_CTRADER_TRANSPORT";

/// `ProtoHeartbeatEvent` — sent every ~30 s by the cTrader Open API
/// server (and by the client to keep the socket alive). Carries no
/// `clientMsgId` and an empty `payload`. v0.4.13 — the WSS transport
/// must skip these envelopes silently in the inner read loop, which
/// is the existing `is_matching_open_api_response` behavior; this
/// constant exists so that consumers reasoning about the wire format
/// can name the type instead of comparing against the magic number.
pub const CTRADER_OA_HEARTBEAT_PAYLOAD_TYPE: u32 = 51;

pub const CTRADER_OA_APPLICATION_AUTH_REQUEST_PAYLOAD_TYPE: u32 = 2100;
pub const CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE: u32 = 2101;
pub const CTRADER_OA_ACCOUNT_AUTH_REQUEST_PAYLOAD_TYPE: u32 = 2102;
pub const CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE: u32 = 2103;
pub const CTRADER_OA_NEW_ORDER_REQUEST_PAYLOAD_TYPE: u32 = 2106;
pub const CTRADER_OA_CANCEL_ORDER_REQUEST_PAYLOAD_TYPE: u32 = 2108;
pub const CTRADER_OA_AMEND_ORDER_REQUEST_PAYLOAD_TYPE: u32 = 2109;
pub const CTRADER_OA_CLOSE_POSITION_REQUEST_PAYLOAD_TYPE: u32 = 2111;
pub const CTRADER_OA_TRADER_REQUEST_PAYLOAD_TYPE: u32 = 2121;
pub const CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE: u32 = 2122;
pub const CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE: u32 = 2124;
pub const CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE: u32 = 2125;
pub const CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE: u32 = 2126;
pub const CTRADER_OA_SUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE: u32 = 2127;
pub const CTRADER_OA_SUBSCRIBE_SPOTS_RESPONSE_PAYLOAD_TYPE: u32 = 2128;
pub const CTRADER_OA_UNSUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE: u32 = 2129;
pub const CTRADER_OA_UNSUBSCRIBE_SPOTS_RESPONSE_PAYLOAD_TYPE: u32 = 2130;
pub const CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE: u32 = 2131;
pub const CTRADER_OA_ORDER_ERROR_EVENT_PAYLOAD_TYPE: u32 = 2132;
pub const CTRADER_OA_DEAL_LIST_REQUEST_PAYLOAD_TYPE: u32 = 2133;
pub const CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE: u32 = 2134;
pub const CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE: u32 = 2135;
pub const CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE: u32 = 2136;
pub const CTRADER_OA_SYMBOLS_LIST_REQUEST_PAYLOAD_TYPE: u32 = 2114;
pub const CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE: u32 = 2115;
pub const CTRADER_OA_SYMBOL_BY_ID_REQUEST_PAYLOAD_TYPE: u32 = 2116;
pub const CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE: u32 = 2117;
// **Phase D.1 follow-up (2026-05-28)**: asset class + symbol category
// payloads. The broker uses these to group symbols into "Forex",
// "Metals", "Indices", "Commodities", "Stocks", "Cryptocurrencies",
// etc. We filter the catalog by these classes so the forex-bot's
// cost model doesn't waste cycles on the 700+ equity symbols that
// will never be traded.
pub const CTRADER_OA_ASSET_CLASS_LIST_REQUEST_PAYLOAD_TYPE: u32 = 2153;
pub const CTRADER_OA_ASSET_CLASS_LIST_RESPONSE_PAYLOAD_TYPE: u32 = 2154;
pub const CTRADER_OA_SYMBOL_CATEGORY_REQUEST_PAYLOAD_TYPE: u32 = 2160;
pub const CTRADER_OA_SYMBOL_CATEGORY_RESPONSE_PAYLOAD_TYPE: u32 = 2161;
// **Phase D.2a (2026-05-28)**: `ProtoOAAssetListReq/Res` — the
// broker's asset registry. Joins `ProtoOALightSymbol.{base,quote}AssetId`
// to a 3-letter currency code (or metal/commodity unit) so the
// downstream `SymbolMetadata` can populate `base`/`quote` strings
// without name-pattern guesses against symbolName.
pub const CTRADER_OA_ASSET_LIST_REQUEST_PAYLOAD_TYPE: u32 = 2112;
pub const CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE: u32 = 2113;
pub const CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE: u32 = 2137;
pub const CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE: u32 = 2138;
pub const CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE: u32 = 2142;
pub const CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE: u32 = 2145;
pub const CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE: u32 = 2146;
pub const CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_REQUEST_PAYLOAD_TYPE: u32 = 2149;
pub const CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_RESPONSE_PAYLOAD_TYPE: u32 = 2150;
pub const CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_RESPONSE_PAYLOAD_TYPE: u32 = 2165;
pub const CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_RESPONSE_PAYLOAD_TYPE: u32 = 2166;
/// Server-pushed event raised when the broker drops the account session.
/// New in the 2026-05-14 upstream proto refresh (Batch 6). Until this
/// landed we only learned about a stale session indirectly from a failed
/// `ProtoOAErrorRes` on the next request — by then the streaming-side had
/// usually already drifted by several heartbeat intervals. Numeric value
/// is fixed in `OpenApiModelMessages.proto::ProtoOAPayloadType`.
pub const CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE: u32 = 2164;
/// Request for current per-position unrealized PnL computed on the broker.
/// New in the 2026-05-14 upstream proto refresh (Batch 6). Used as an
/// audit cross-check against the local PnL calculation; see
/// `crates/neoethos-app/src/app_services/pnl.rs`.
pub const CTRADER_OA_GET_POSITION_UNREALIZED_PNL_REQUEST_PAYLOAD_TYPE: u32 = 2187;
pub const CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE: u32 = 2188;
// **Phase D.1b (2026-05-28)** — note: `CTRADER_OA_SYMBOL_CATEGORY_*`
// (2160/2161) are declared above with the other 2154/2160 family,
// alongside `CTRADER_OA_ASSET_CLASS_LIST_*` (2153/2154). The earlier
// pre-Phase-D constant set lived here; consolidated upward so the
// group-G chunk is grep-able as one block.

/// `ProtoOADealListByPositionIdReq` / `…Res` — group D, narrow-window
/// trade-history lookup tied to a single `positionId`.
pub const CTRADER_OA_DEAL_LIST_BY_POSITION_ID_REQUEST_PAYLOAD_TYPE: u32 = 2179;
pub const CTRADER_OA_DEAL_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE: u32 = 2180;
/// `ProtoOAOrderDetailsReq` / `…Res` — group D, fetch one order plus
/// all of its child deals by `orderId`.
pub const CTRADER_OA_ORDER_DETAILS_REQUEST_PAYLOAD_TYPE: u32 = 2181;
pub const CTRADER_OA_ORDER_DETAILS_RESPONSE_PAYLOAD_TYPE: u32 = 2182;
/// `ProtoOAOrderListByPositionIdReq` / `…Res` — group D, list of orders
/// that hit a single `positionId` over a time window.
pub const CTRADER_OA_ORDER_LIST_BY_POSITION_ID_REQUEST_PAYLOAD_TYPE: u32 = 2183;
pub const CTRADER_OA_ORDER_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE: u32 = 2184;

// ─────────────────────────────────────────────────────────────────────────
// **2026-06-10 — full ProtoOAPayloadType catalog (operator directive
// "read the whole cTrader API and make it part of the bot").** Every
// remaining `ProtoOAPayloadType` from Spotware's `OpenApiModelMessages.proto`
// that the bot did not yet name. Numeric values verified against the upstream
// proto enum. Naming every variant means `classify_open_api_payload` can
// recognise — and the read loops can deliberately handle — every frame the
// broker can send, instead of treating unknown types as opaque.
// ─────────────────────────────────────────────────────────────────────────

/// `ProtoOAVersionReq/Res` — query the broker's Open API proto version.
pub const CTRADER_OA_VERSION_REQUEST_PAYLOAD_TYPE: u32 = 2104;
pub const CTRADER_OA_VERSION_RESPONSE_PAYLOAD_TYPE: u32 = 2105;
/// `ProtoOATrailingSLChangedEvent` — pushed when the broker moves a
/// position's trailing stop-loss to a new trigger price.
pub const CTRADER_OA_TRAILING_SL_CHANGED_EVENT_PAYLOAD_TYPE: u32 = 2107;
/// `ProtoOAAmendPositionSLTPReq` — modify the stop-loss / take-profit (and
/// trailing / guaranteed flags) of an ALREADY-OPEN position. The response is
/// a `ProtoOAExecutionEvent` (2126), like the other trade actions. This is the
/// capability that lets the bot move a stop to breakeven or trail a winner.
pub const CTRADER_OA_AMEND_POSITION_SLTP_REQUEST_PAYLOAD_TYPE: u32 = 2110;
/// `ProtoOASymbolsForConversionReq/Res` — the symbol chain used to convert a
/// P&L expressed in one asset into the account's deposit currency.
pub const CTRADER_OA_SYMBOLS_FOR_CONVERSION_REQUEST_PAYLOAD_TYPE: u32 = 2118;
pub const CTRADER_OA_SYMBOLS_FOR_CONVERSION_RESPONSE_PAYLOAD_TYPE: u32 = 2119;
/// `ProtoOASymbolChangedEvent` — pushed when a symbol's specification (digits,
/// trading hours, swap, etc.) changes broker-side; consumers should refetch.
pub const CTRADER_OA_SYMBOL_CHANGED_EVENT_PAYLOAD_TYPE: u32 = 2120;
/// `ProtoOATraderUpdatedEvent` — pushed when the trader/account record changes
/// (balance, leverage, bonus, …) WITHOUT us polling `ProtoOATraderReq`. Lets
/// the live account view stay fresh between reconcile ticks.
pub const CTRADER_OA_TRADER_UPDATE_EVENT_PAYLOAD_TYPE: u32 = 2123;
/// `ProtoOAExpectedMarginReq/Res` — ask the broker how much margin a given
/// volume on a symbol would lock up, BEFORE placing the order (pre-trade risk).
pub const CTRADER_OA_EXPECTED_MARGIN_REQUEST_PAYLOAD_TYPE: u32 = 2139;
pub const CTRADER_OA_EXPECTED_MARGIN_RESPONSE_PAYLOAD_TYPE: u32 = 2140;
/// `ProtoOAMarginChangedEvent` — pushed when the used margin on the account
/// changes (a fill, a partial close, a price move on a marked position).
pub const CTRADER_OA_MARGIN_CHANGED_EVENT_PAYLOAD_TYPE: u32 = 2141;
/// `ProtoOACashFlowHistoryListReq/Res` — deposits, withdrawals, swap and
/// commission charges over a time window. Backs the money-view / equity-curve.
pub const CTRADER_OA_CASH_FLOW_HISTORY_LIST_REQUEST_PAYLOAD_TYPE: u32 = 2143;
pub const CTRADER_OA_CASH_FLOW_HISTORY_LIST_RESPONSE_PAYLOAD_TYPE: u32 = 2144;
/// `ProtoOAAccountsTokenInvalidatedEvent` — pushed when the access token tied
/// to the streamed accounts is revoked/expired broker-side. The single most
/// important auth event: it tells us to force an OAuth re-auth NOW.
pub const CTRADER_OA_ACCOUNTS_TOKEN_INVALIDATED_EVENT_PAYLOAD_TYPE: u32 = 2147;
/// `ProtoOAClientDisconnectEvent` — pushed when the broker disconnects the
/// whole application session (not just one account).
pub const CTRADER_OA_CLIENT_DISCONNECT_EVENT_PAYLOAD_TYPE: u32 = 2148;
/// `ProtoOAGetCtidProfileByTokenReq/Res` — the cTID profile (user id, nickname)
/// behind an access token.
pub const CTRADER_OA_GET_CTID_PROFILE_BY_TOKEN_REQUEST_PAYLOAD_TYPE: u32 = 2151;
pub const CTRADER_OA_GET_CTID_PROFILE_BY_TOKEN_RESPONSE_PAYLOAD_TYPE: u32 = 2152;
/// `ProtoOADepthEvent` + subscribe/unsubscribe — Level-2 depth-of-market.
pub const CTRADER_OA_DEPTH_EVENT_PAYLOAD_TYPE: u32 = 2155;
pub const CTRADER_OA_SUBSCRIBE_DEPTH_QUOTES_REQUEST_PAYLOAD_TYPE: u32 = 2156;
pub const CTRADER_OA_SUBSCRIBE_DEPTH_QUOTES_RESPONSE_PAYLOAD_TYPE: u32 = 2157;
pub const CTRADER_OA_UNSUBSCRIBE_DEPTH_QUOTES_REQUEST_PAYLOAD_TYPE: u32 = 2158;
pub const CTRADER_OA_UNSUBSCRIBE_DEPTH_QUOTES_RESPONSE_PAYLOAD_TYPE: u32 = 2159;
/// `ProtoOAAccountLogoutReq/Res` — cleanly release an account session
/// (counterpart to account-auth) so the broker frees server-side state.
pub const CTRADER_OA_ACCOUNT_LOGOUT_REQUEST_PAYLOAD_TYPE: u32 = 2162;
pub const CTRADER_OA_ACCOUNT_LOGOUT_RESPONSE_PAYLOAD_TYPE: u32 = 2163;
/// `ProtoOAMarginCallListReq/Res`, `…UpdateReq/Res`, `…UpdateEvent`,
/// `…TriggerEvent` — the broker's margin-call thresholds and the pushed
/// alert when one is hit (a critical risk signal for a live bot).
pub const CTRADER_OA_MARGIN_CALL_LIST_REQUEST_PAYLOAD_TYPE: u32 = 2167;
pub const CTRADER_OA_MARGIN_CALL_LIST_RESPONSE_PAYLOAD_TYPE: u32 = 2168;
pub const CTRADER_OA_MARGIN_CALL_UPDATE_REQUEST_PAYLOAD_TYPE: u32 = 2169;
pub const CTRADER_OA_MARGIN_CALL_UPDATE_RESPONSE_PAYLOAD_TYPE: u32 = 2170;
pub const CTRADER_OA_MARGIN_CALL_UPDATE_EVENT_PAYLOAD_TYPE: u32 = 2171;
pub const CTRADER_OA_MARGIN_CALL_TRIGGER_EVENT_PAYLOAD_TYPE: u32 = 2172;
/// `ProtoOARefreshTokenReq/Res` — refresh the OAuth access token over the
/// Open API socket itself (alternative to the HTTPS token endpoint).
pub const CTRADER_OA_REFRESH_TOKEN_REQUEST_PAYLOAD_TYPE: u32 = 2173;
pub const CTRADER_OA_REFRESH_TOKEN_RESPONSE_PAYLOAD_TYPE: u32 = 2174;
/// `ProtoOAOrderListReq/Res` — historical orders over a time window (the
/// account-wide order history that backs the trade journal / analytics).
pub const CTRADER_OA_ORDER_LIST_REQUEST_PAYLOAD_TYPE: u32 = 2175;
pub const CTRADER_OA_ORDER_LIST_RESPONSE_PAYLOAD_TYPE: u32 = 2176;
/// `ProtoOAGetDynamicLeverageByIDReq/Res` — the broker's dynamic-leverage
/// tiers (volume-banded leverage) referenced by a symbol's `leverageId`.
pub const CTRADER_OA_GET_DYNAMIC_LEVERAGE_REQUEST_PAYLOAD_TYPE: u32 = 2177;
pub const CTRADER_OA_GET_DYNAMIC_LEVERAGE_RESPONSE_PAYLOAD_TYPE: u32 = 2178;
/// `ProtoOADealOffsetListReq/Res` — for a closing deal, the opening deals it
/// offset (FIFO matching) and the realised gross/net per offset.
pub const CTRADER_OA_DEAL_OFFSET_LIST_REQUEST_PAYLOAD_TYPE: u32 = 2185;
pub const CTRADER_OA_DEAL_OFFSET_LIST_RESPONSE_PAYLOAD_TYPE: u32 = 2186;

pub const CTRADER_QUOTE_TYPE_BID: i32 = 1;
pub const CTRADER_QUOTE_TYPE_ASK: i32 = 2;
pub const CTRADER_TRADE_SIDE_BUY: i32 = 1;
pub const CTRADER_TRADE_SIDE_SELL: i32 = 2;
pub const CTRADER_ORDER_TYPE_MARKET: i32 = 1;
pub const CTRADER_ORDER_TYPE_LIMIT: i32 = 2;
pub const CTRADER_ORDER_TYPE_STOP: i32 = 3;
pub const CTRADER_ORDER_TYPE_STOP_LOSS_TAKE_PROFIT: i32 = 4;
pub const CTRADER_ORDER_TYPE_MARKET_RANGE: i32 = 5;
pub const CTRADER_ORDER_TYPE_STOP_LIMIT: i32 = 6;
pub const CTRADER_TIME_IN_FORCE_GOOD_TILL_DATE: i32 = 1;
pub const CTRADER_TIME_IN_FORCE_GOOD_TILL_CANCEL: i32 = 2;
pub const CTRADER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL: i32 = 3;
pub const CTRADER_TIME_IN_FORCE_FILL_OR_KILL: i32 = 4;
pub const CTRADER_TIME_IN_FORCE_MARKET_ON_OPEN: i32 = 5;
pub const CTRADER_ORDER_TRIGGER_METHOD_TRADE: i32 = 1;
pub const CTRADER_ORDER_TRIGGER_METHOD_OPPOSITE: i32 = 2;
pub const CTRADER_ORDER_TRIGGER_METHOD_DOUBLE_TRADE: i32 = 3;
pub const CTRADER_ORDER_TRIGGER_METHOD_DOUBLE_OPPOSITE: i32 = 4;

// v0.4.13 — `clientMsgId` and `payload` are now defaultable so the
// production WSS transport can parse out-of-band envelopes that the
// cTrader Open API server sends without correlation IDs (heartbeat
// payloadType 51, unsolicited push events like ProtoExecutionEvent
// with `clientMsgId` absent because there was no client request to
// correlate to). Before this change, a heartbeat arriving between
// the application-auth response and the account-list response would
// fail `parse_open_api_envelope` with the generic "failed to parse
// cTrader JSON envelope" error, and the wizard's account-discovery
// leg would abort. Phase X1 walkthrough on 2026-05-19 caught this:
// even though the OAuth token bundle was received, the next message
// off the wire was a heartbeat-shaped frame the parser couldn't
// accept, and the wizard reported "OAuth error: failed to parse
// cTrader JSON envelope". With #[serde(default)] both fields fall
// back to "" / Value::Null on absence, the heartbeat-skip loop in
// the transport (`is_matching_open_api_response`) gets a chance to
// fire, and the genuine account-list response is read on the next
// iteration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CTraderOpenApiJsonMessage {
    #[serde(rename = "clientMsgId", default)]
    pub client_msg_id: String,
    #[serde(rename = "payloadType")]
    pub payload_type: u32,
    #[serde(default)]
    pub payload: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTraderTradeSide {
    Buy,
    Sell,
}

pub const SUPPORTED_CTRADER_TRADE_SIDES: [CTraderTradeSide; 2] =
    [CTraderTradeSide::Buy, CTraderTradeSide::Sell];

impl CTraderTradeSide {
    fn as_i32(self) -> i32 {
        match self {
            Self::Buy => CTRADER_TRADE_SIDE_BUY,
            Self::Sell => CTRADER_TRADE_SIDE_SELL,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Buy => "BUY",
            Self::Sell => "SELL",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTraderOrderType {
    Market,
    Limit,
    Stop,
    StopLossTakeProfit,
    MarketRange,
    StopLimit,
}

pub const SUPPORTED_CTRADER_ORDER_TYPES: [CTraderOrderType; 6] = [
    CTraderOrderType::Market,
    CTraderOrderType::Limit,
    CTraderOrderType::Stop,
    CTraderOrderType::StopLossTakeProfit,
    CTraderOrderType::MarketRange,
    CTraderOrderType::StopLimit,
];

impl CTraderOrderType {
    fn as_i32(self) -> i32 {
        match self {
            Self::Market => CTRADER_ORDER_TYPE_MARKET,
            Self::Limit => CTRADER_ORDER_TYPE_LIMIT,
            Self::Stop => CTRADER_ORDER_TYPE_STOP,
            Self::StopLossTakeProfit => CTRADER_ORDER_TYPE_STOP_LOSS_TAKE_PROFIT,
            Self::MarketRange => CTRADER_ORDER_TYPE_MARKET_RANGE,
            Self::StopLimit => CTRADER_ORDER_TYPE_STOP_LIMIT,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Market => "MARKET",
            Self::Limit => "LIMIT",
            Self::Stop => "STOP",
            Self::StopLossTakeProfit => "STOP_LOSS_TAKE_PROFIT",
            Self::MarketRange => "MARKET_RANGE",
            Self::StopLimit => "STOP_LIMIT",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTraderTimeInForce {
    GoodTillDate,
    GoodTillCancel,
    ImmediateOrCancel,
    FillOrKill,
    MarketOnOpen,
}

pub const SUPPORTED_CTRADER_TIME_IN_FORCE: [CTraderTimeInForce; 5] = [
    CTraderTimeInForce::GoodTillDate,
    CTraderTimeInForce::GoodTillCancel,
    CTraderTimeInForce::ImmediateOrCancel,
    CTraderTimeInForce::FillOrKill,
    CTraderTimeInForce::MarketOnOpen,
];

impl CTraderTimeInForce {
    fn as_i32(self) -> i32 {
        match self {
            Self::GoodTillDate => CTRADER_TIME_IN_FORCE_GOOD_TILL_DATE,
            Self::GoodTillCancel => CTRADER_TIME_IN_FORCE_GOOD_TILL_CANCEL,
            Self::ImmediateOrCancel => CTRADER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
            Self::FillOrKill => CTRADER_TIME_IN_FORCE_FILL_OR_KILL,
            Self::MarketOnOpen => CTRADER_TIME_IN_FORCE_MARKET_ON_OPEN,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::GoodTillDate => "GOOD_TILL_DATE",
            Self::GoodTillCancel => "GOOD_TILL_CANCEL",
            Self::ImmediateOrCancel => "IMMEDIATE_OR_CANCEL",
            Self::FillOrKill => "FILL_OR_KILL",
            Self::MarketOnOpen => "MARKET_ON_OPEN",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTraderOrderTriggerMethod {
    Trade,
    Opposite,
    DoubleTrade,
    DoubleOpposite,
}

pub const SUPPORTED_CTRADER_ORDER_TRIGGER_METHODS: [CTraderOrderTriggerMethod; 4] = [
    CTraderOrderTriggerMethod::Trade,
    CTraderOrderTriggerMethod::Opposite,
    CTraderOrderTriggerMethod::DoubleTrade,
    CTraderOrderTriggerMethod::DoubleOpposite,
];

impl CTraderOrderTriggerMethod {
    fn as_i32(self) -> i32 {
        match self {
            Self::Trade => CTRADER_ORDER_TRIGGER_METHOD_TRADE,
            Self::Opposite => CTRADER_ORDER_TRIGGER_METHOD_OPPOSITE,
            Self::DoubleTrade => CTRADER_ORDER_TRIGGER_METHOD_DOUBLE_TRADE,
            Self::DoubleOpposite => CTRADER_ORDER_TRIGGER_METHOD_DOUBLE_OPPOSITE,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Trade => "TRADE",
            Self::Opposite => "OPPOSITE",
            Self::DoubleTrade => "DOUBLE_TRADE",
            Self::DoubleOpposite => "DOUBLE_OPPOSITE",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderDealListRequest {
    pub account_id: i64,
    pub from_timestamp_ms: Option<i64>,
    pub to_timestamp_ms: Option<i64>,
    pub max_rows: Option<i32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderNewOrderRequest {
    pub account_id: i64,
    pub symbol_id: i64,
    pub order_type: CTraderOrderType,
    pub trade_side: CTraderTradeSide,
    pub volume: i64,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub time_in_force: Option<CTraderTimeInForce>,
    pub expiration_timestamp_ms: Option<i64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub comment: Option<String>,
    pub base_slippage_price: Option<f64>,
    pub slippage_in_points: Option<i32>,
    pub label: Option<String>,
    pub position_id: Option<i64>,
    pub client_order_id: Option<String>,
    pub relative_stop_loss: Option<i64>,
    pub relative_take_profit: Option<i64>,
    pub guaranteed_stop_loss: Option<bool>,
    pub trailing_stop_loss: Option<bool>,
    pub stop_trigger_method: Option<CTraderOrderTriggerMethod>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderAmendOrderRequest {
    pub account_id: i64,
    pub order_id: i64,
    pub volume: Option<i64>,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub expiration_timestamp_ms: Option<i64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub slippage_in_points: Option<i32>,
    pub relative_stop_loss: Option<i64>,
    pub relative_take_profit: Option<i64>,
    pub guaranteed_stop_loss: Option<bool>,
    pub trailing_stop_loss: Option<bool>,
    pub stop_trigger_method: Option<CTraderOrderTriggerMethod>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderCancelOrderRequest {
    pub account_id: i64,
    pub order_id: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderClosePositionRequest {
    pub account_id: i64,
    pub position_id: i64,
    pub volume: i64,
}

/// `ProtoOAAmendPositionSLTPReq` (2110) — change the stop-loss / take-profit
/// of an ALREADY-OPEN position. Both prices are ABSOLUTE (unlike the
/// new-order relative pips). Any `None` field is left untouched by the broker.
/// The response is a `ProtoOAExecutionEvent` (2126). **2026-06-10.**
#[derive(Debug, Clone, PartialEq)]
pub struct CTraderAmendPositionSltpRequest {
    pub account_id: i64,
    pub position_id: i64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub guaranteed_stop_loss: Option<bool>,
    pub trailing_stop_loss: Option<bool>,
    pub stop_loss_trigger_method: Option<CTraderOrderTriggerMethod>,
}

pub trait CTraderOpenApiTransport {
    fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> Result<Vec<String>>;
}

pub struct ProductionCTraderOpenApiTransport {
    endpoint_host: String,
}

impl ProductionCTraderOpenApiTransport {
    pub fn new(endpoint_host: impl Into<String>) -> Self {
        Self {
            endpoint_host: endpoint_host.into(),
        }
    }
}

pub fn build_application_auth_json(
    client_id: &str,
    client_secret: &str,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_APPLICATION_AUTH_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "clientId": client_id,
            "clientSecret": client_secret,
        }),
    }
}

pub fn build_application_auth_request(
    client_id: &str,
    client_secret: &str,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    build_application_auth_json(client_id, client_secret, client_msg_id)
}

pub fn build_account_auth_request(
    ctid_trader_account_id: i64,
    access_token: &str,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_ACCOUNT_AUTH_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "accessToken": access_token,
        }),
    }
}

pub fn build_account_list_by_access_token_request(
    access_token: &str,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "accessToken": access_token,
        }),
    }
}

pub fn build_deal_list_request(
    request: &CTraderDealListRequest,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    let mut payload = serde_json::json!({
        "ctidTraderAccountId": request.account_id,
    });
    if let Some(from_timestamp_ms) = request.from_timestamp_ms {
        payload["fromTimestamp"] = serde_json::json!(from_timestamp_ms);
    }
    if let Some(to_timestamp_ms) = request.to_timestamp_ms {
        payload["toTimestamp"] = serde_json::json!(to_timestamp_ms);
    }
    if let Some(max_rows) = request.max_rows {
        payload["maxRows"] = serde_json::json!(max_rows);
    }
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_DEAL_LIST_REQUEST_PAYLOAD_TYPE,
        payload,
    }
}

pub fn build_new_order_request(
    request: &CTraderNewOrderRequest,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    let mut payload = serde_json::json!({
        "ctidTraderAccountId": request.account_id,
        "symbolId": request.symbol_id,
        "orderType": request.order_type.as_i32(),
        "tradeSide": request.trade_side.as_i32(),
        "volume": request.volume,
    });
    if let Some(limit_price) = request.limit_price {
        payload["limitPrice"] = serde_json::json!(limit_price);
    }
    if let Some(stop_price) = request.stop_price {
        payload["stopPrice"] = serde_json::json!(stop_price);
    }
    if let Some(time_in_force) = request.time_in_force {
        payload["timeInForce"] = serde_json::json!(time_in_force.as_i32());
    }
    if let Some(expiration_timestamp_ms) = request.expiration_timestamp_ms {
        payload["expirationTimestamp"] = serde_json::json!(expiration_timestamp_ms);
    }
    if let Some(stop_loss) = request.stop_loss {
        payload["stopLoss"] = serde_json::json!(stop_loss);
    }
    if let Some(take_profit) = request.take_profit {
        payload["takeProfit"] = serde_json::json!(take_profit);
    }
    if let Some(comment) = &request.comment {
        payload["comment"] = serde_json::json!(comment);
    }
    if let Some(base_slippage_price) = request.base_slippage_price {
        payload["baseSlippagePrice"] = serde_json::json!(base_slippage_price);
    }
    if let Some(slippage_in_points) = request.slippage_in_points {
        payload["slippageInPoints"] = serde_json::json!(slippage_in_points);
    }
    if let Some(label) = &request.label {
        payload["label"] = serde_json::json!(label);
    }
    if let Some(position_id) = request.position_id {
        payload["positionId"] = serde_json::json!(position_id);
    }
    if let Some(client_order_id) = &request.client_order_id {
        payload["clientOrderId"] = serde_json::json!(client_order_id);
    }
    if let Some(relative_stop_loss) = request.relative_stop_loss {
        payload["relativeStopLoss"] = serde_json::json!(relative_stop_loss);
    }
    if let Some(relative_take_profit) = request.relative_take_profit {
        payload["relativeTakeProfit"] = serde_json::json!(relative_take_profit);
    }
    if let Some(guaranteed_stop_loss) = request.guaranteed_stop_loss {
        payload["guaranteedStopLoss"] = serde_json::json!(guaranteed_stop_loss);
    }
    if let Some(trailing_stop_loss) = request.trailing_stop_loss {
        payload["trailingStopLoss"] = serde_json::json!(trailing_stop_loss);
    }
    if let Some(stop_trigger_method) = request.stop_trigger_method {
        payload["stopTriggerMethod"] = serde_json::json!(stop_trigger_method.as_i32());
    }
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_NEW_ORDER_REQUEST_PAYLOAD_TYPE,
        payload,
    }
}

pub fn build_amend_order_request(
    request: &CTraderAmendOrderRequest,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    let mut payload = serde_json::json!({
        "ctidTraderAccountId": request.account_id,
        "orderId": request.order_id,
    });
    if let Some(volume) = request.volume {
        payload["volume"] = serde_json::json!(volume);
    }
    if let Some(limit_price) = request.limit_price {
        payload["limitPrice"] = serde_json::json!(limit_price);
    }
    if let Some(stop_price) = request.stop_price {
        payload["stopPrice"] = serde_json::json!(stop_price);
    }
    if let Some(expiration_timestamp_ms) = request.expiration_timestamp_ms {
        payload["expirationTimestamp"] = serde_json::json!(expiration_timestamp_ms);
    }
    if let Some(stop_loss) = request.stop_loss {
        payload["stopLoss"] = serde_json::json!(stop_loss);
    }
    if let Some(take_profit) = request.take_profit {
        payload["takeProfit"] = serde_json::json!(take_profit);
    }
    if let Some(slippage_in_points) = request.slippage_in_points {
        payload["slippageInPoints"] = serde_json::json!(slippage_in_points);
    }
    if let Some(relative_stop_loss) = request.relative_stop_loss {
        payload["relativeStopLoss"] = serde_json::json!(relative_stop_loss);
    }
    if let Some(relative_take_profit) = request.relative_take_profit {
        payload["relativeTakeProfit"] = serde_json::json!(relative_take_profit);
    }
    if let Some(guaranteed_stop_loss) = request.guaranteed_stop_loss {
        payload["guaranteedStopLoss"] = serde_json::json!(guaranteed_stop_loss);
    }
    if let Some(trailing_stop_loss) = request.trailing_stop_loss {
        payload["trailingStopLoss"] = serde_json::json!(trailing_stop_loss);
    }
    if let Some(stop_trigger_method) = request.stop_trigger_method {
        payload["stopTriggerMethod"] = serde_json::json!(stop_trigger_method.as_i32());
    }
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_AMEND_ORDER_REQUEST_PAYLOAD_TYPE,
        payload,
    }
}

pub fn build_cancel_order_request(
    request: &CTraderCancelOrderRequest,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_CANCEL_ORDER_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": request.account_id,
            "orderId": request.order_id,
        }),
    }
}

pub fn build_close_position_request(
    request: &CTraderClosePositionRequest,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_CLOSE_POSITION_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": request.account_id,
            "positionId": request.position_id,
            "volume": request.volume,
        }),
    }
}

pub fn build_trader_request(
    ctid_trader_account_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_TRADER_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
        }),
    }
}

pub fn build_reconcile_request(
    ctid_trader_account_id: i64,
    return_protection_orders: bool,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "returnProtectionOrders": return_protection_orders,
        }),
    }
}

/// Build the JSON envelope for `ProtoOAGetPositionUnrealizedPnLReq`
/// (payload type 2187). The proto carries only the two required fields
/// `payloadType` (filled by the envelope's `payload_type`) and
/// `ctidTraderAccountId`; this matches the 2026-05-14 upstream refresh.
///
/// Use [`crate::app_services::pnl::fetch_broker_unrealized_pnl`] for the
/// full audit flow that compares broker values against the locally
/// computed PnL on every reconcile tick.
pub fn build_get_position_unrealized_pnl_request(
    ctid_trader_account_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_GET_POSITION_UNREALIZED_PNL_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
        }),
    }
}

/// `ProtoOAAmendPositionSLTPReq` (2110) — move an open position's stops.
/// Flagship of the 2026-06-10 API-completeness pass: this is what lets the
/// bot trail a winner or pull a stop to breakeven. Only the provided fields
/// are sent; omitted ones are left as-is on the broker.
pub fn build_amend_position_sltp_request(
    request: &CTraderAmendPositionSltpRequest,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    let mut payload = serde_json::json!({
        "ctidTraderAccountId": request.account_id,
        "positionId": request.position_id,
    });
    if let Some(stop_loss) = request.stop_loss {
        payload["stopLoss"] = serde_json::json!(stop_loss);
    }
    if let Some(take_profit) = request.take_profit {
        payload["takeProfit"] = serde_json::json!(take_profit);
    }
    if let Some(guaranteed_stop_loss) = request.guaranteed_stop_loss {
        payload["guaranteedStopLoss"] = serde_json::json!(guaranteed_stop_loss);
    }
    if let Some(trailing_stop_loss) = request.trailing_stop_loss {
        payload["trailingStopLoss"] = serde_json::json!(trailing_stop_loss);
    }
    if let Some(stop_loss_trigger_method) = request.stop_loss_trigger_method {
        payload["stopLossTriggerMethod"] = serde_json::json!(stop_loss_trigger_method.as_i32());
    }
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_AMEND_POSITION_SLTP_REQUEST_PAYLOAD_TYPE,
        payload,
    }
}

/// `ProtoOAVersionReq` (2104) — ask the broker which Open API proto version
/// it speaks. Carries no account; useful as a connectivity/handshake probe.
pub fn build_version_request(client_msg_id: impl Into<String>) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_VERSION_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({}),
    }
}

/// `ProtoOAExpectedMarginReq` (2139) — pre-trade margin: how much margin would
/// each of `volumes` (in 0.01-unit wire volume) lock up on `symbol_id`.
pub fn build_expected_margin_request(
    ctid_trader_account_id: i64,
    symbol_id: i64,
    volumes: &[i64],
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_EXPECTED_MARGIN_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_id,
            "volume": volumes,
        }),
    }
}

/// `ProtoOAOrderListReq` (2175) — account-wide historical orders over a time
/// window. Backs the trade journal / analytics (the order history that the
/// per-position lookups can't give in one call).
pub fn build_order_list_request(
    ctid_trader_account_id: i64,
    from_timestamp_ms: i64,
    to_timestamp_ms: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_ORDER_LIST_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "fromTimestamp": from_timestamp_ms,
            "toTimestamp": to_timestamp_ms,
        }),
    }
}

/// `ProtoOACashFlowHistoryListReq` (2143) — deposits, withdrawals, swap and
/// commission over a time window. Backs the €-money view / equity curve.
pub fn build_cash_flow_history_list_request(
    ctid_trader_account_id: i64,
    from_timestamp_ms: i64,
    to_timestamp_ms: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_CASH_FLOW_HISTORY_LIST_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "fromTimestamp": from_timestamp_ms,
            "toTimestamp": to_timestamp_ms,
        }),
    }
}

/// `ProtoOARefreshTokenReq` (2173) — refresh the OAuth access token over the
/// Open API socket itself (the alternative to the HTTPS token endpoint).
pub fn build_refresh_token_request(
    refresh_token: impl Into<String>,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_REFRESH_TOKEN_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "refreshToken": refresh_token.into(),
        }),
    }
}

/// `ProtoOAGetCtidProfileByTokenReq` (2151) — the cTID profile behind an
/// access token (user id, nickname).
pub fn build_get_ctid_profile_by_token_request(
    access_token: impl Into<String>,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_GET_CTID_PROFILE_BY_TOKEN_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "accessToken": access_token.into(),
        }),
    }
}

/// `ProtoOAAccountLogoutReq` (2162) — cleanly release an account session so the
/// broker frees server-side state (counterpart to account-auth).
pub fn build_account_logout_request(
    ctid_trader_account_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_ACCOUNT_LOGOUT_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
        }),
    }
}

/// `ProtoOAGetDynamicLeverageByIDReq` (2177) — the broker's dynamic-leverage
/// tier table referenced by a symbol's `leverageId`.
pub fn build_get_dynamic_leverage_request(
    ctid_trader_account_id: i64,
    leverage_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_GET_DYNAMIC_LEVERAGE_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "leverageId": leverage_id,
        }),
    }
}

/// `ProtoOASymbolsForConversionReq` (2118) — the symbol chain to convert a P&L
/// expressed in `first_asset_id` into `last_asset_id` (the deposit currency).
pub fn build_symbols_for_conversion_request(
    ctid_trader_account_id: i64,
    first_asset_id: i64,
    last_asset_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_SYMBOLS_FOR_CONVERSION_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "firstAssetId": first_asset_id,
            "lastAssetId": last_asset_id,
        }),
    }
}

/// `ProtoOADealOffsetListReq` (2185) — for a closing deal, the opening deals it
/// offset (FIFO) and the realised gross/net per offset.
pub fn build_deal_offset_list_request(
    ctid_trader_account_id: i64,
    deal_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_DEAL_OFFSET_LIST_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "dealId": deal_id,
        }),
    }
}

/// `ProtoOAMarginCallListReq` (2167) — the account's configured margin-call
/// thresholds (the levels whose breach fires a `…TriggerEvent`).
pub fn build_margin_call_list_request(
    ctid_trader_account_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_MARGIN_CALL_LIST_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
        }),
    }
}

/// `ProtoOASubscribeDepthQuotesReq` (2156) — subscribe to Level-2 depth-of-
/// market for the given symbols (pushes `ProtoOADepthEvent`s).
pub fn build_subscribe_depth_quotes_request(
    ctid_trader_account_id: i64,
    symbol_ids: &[i64],
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_SUBSCRIBE_DEPTH_QUOTES_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_ids,
        }),
    }
}

/// `ProtoOAUnsubscribeDepthQuotesReq` (2158) — stop Level-2 depth for symbols.
pub fn build_unsubscribe_depth_quotes_request(
    ctid_trader_account_id: i64,
    symbol_ids: &[i64],
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_UNSUBSCRIBE_DEPTH_QUOTES_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_ids,
        }),
    }
}

pub fn build_subscribe_spots_request(
    ctid_trader_account_id: i64,
    symbol_ids: &[i64],
    subscribe_to_spot_timestamp: bool,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_SUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_ids,
            "subscribeToSpotTimestamp": subscribe_to_spot_timestamp,
        }),
    }
}

pub fn build_unsubscribe_spots_request(
    ctid_trader_account_id: i64,
    symbol_ids: &[i64],
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_UNSUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_ids,
        }),
    }
}

pub fn build_subscribe_live_trendbar_request(
    ctid_trader_account_id: i64,
    symbol_id: i64,
    period: i32,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_id,
            "period": period,
        }),
    }
}

pub fn build_unsubscribe_live_trendbar_request(
    ctid_trader_account_id: i64,
    symbol_id: i64,
    period: i32,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_id,
            "period": period,
        }),
    }
}

pub fn build_symbols_list_request(
    ctid_trader_account_id: i64,
    include_archived_symbols: bool,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_SYMBOLS_LIST_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "includeArchivedSymbols": include_archived_symbols,
        }),
    }
}

pub fn build_symbol_by_id_request(
    ctid_trader_account_id: i64,
    symbol_ids: &[i64],
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_SYMBOL_BY_ID_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_ids,
        }),
    }
}

/// **Phase D.2a (2026-05-28)** — request the broker's asset list
/// (NOT asset CLASS list — this is the per-currency table).
/// Returns `[{assetId, name="EUR"|"USD"|"XAU"|..., displayName,
/// digits}, ...]`. Used to map a LightSymbol's
/// `baseAssetId`/`quoteAssetId` integers to the 3-letter currency
/// strings that `SymbolMetadata` uses.
pub fn build_asset_list_request(
    ctid_trader_account_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_ASSET_LIST_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
        }),
    }
}

/// **Phase D.1 (2026-05-28)** — request the broker's asset class
/// list. Returns top-level groupings like "Forex", "Metals",
/// "Indices", "Commodities", "Stocks", "Cryptocurrencies", "ETFs".
/// Used by `--bootstrap-broker-catalog` to filter the catalog to
/// only the asset classes relevant to forex-ai.
pub fn build_asset_class_list_request(
    ctid_trader_account_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_ASSET_CLASS_LIST_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
        }),
    }
}

/// **Phase D.1 (2026-05-28)** — request the broker's symbol category
/// list. Each category links a `symbolCategoryId` (carried on
/// `ProtoOALightSymbol`) to a parent `assetClassId` and a human
/// name like "FX Majors" / "Spot Metals" / "US Indices". The chain
/// asset_class → symbol_category → light_symbol lets us classify
/// every catalog entry without name-pattern hacks.
pub fn build_symbol_category_list_request(
    ctid_trader_account_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_SYMBOL_CATEGORY_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
        }),
    }
}

pub fn build_get_trendbars_request(
    ctid_trader_account_id: i64,
    symbol_id: i64,
    period: i32,
    from_timestamp_ms: i64,
    to_timestamp_ms: i64,
    count: Option<u32>,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    let mut payload = serde_json::json!({
        "ctidTraderAccountId": ctid_trader_account_id,
        "symbolId": symbol_id,
        "period": period,
        "fromTimestamp": from_timestamp_ms,
        "toTimestamp": to_timestamp_ms,
    });
    if let Some(count) = count {
        payload["count"] = serde_json::json!(count);
    }
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE,
        payload,
    }
}

pub fn trendbar_period_value(label: &str) -> Result<i32> {
    // Map our canonical timeframe labels to cTrader's wire-protocol
    // `ProtoOATrendbarPeriod` codes. cTrader itself supports a slightly
    // different set (M2/M4/M10) that we deliberately omit; we only emit
    // the canonical 11 to keep every subsystem (UI, training, discovery)
    // aligned. M2/M4/M10 and any non-canonical label are rejected.
    //
    // Native cTrader periods (per `openapi-proto-messages/OpenApiModelMessages.proto`):
    //   M1=1, M2=2, M3=3, M4=4, M5=5, M10=6, M15=7, M30=8,
    //   H1=9, H4=10, H12=11, D1=12, W1=13, MN1=14.
    //
    // Note: M3 IS native (enum value 3) — we send it directly. H2 is
    // intentionally absent from neoethos_core::CANONICAL_TIMEFRAMES (see
    // the verbatim operator instruction documented there) and cTrader
    // does not natively expose H2 either, so no H2 routing is needed.
    let upper = label.trim().to_ascii_uppercase();
    if !neoethos_core::is_canonical_timeframe(&upper) {
        return Err(anyhow!(
            "unsupported cTrader trendbar period label {} (not in canonical timeframes)",
            label
        ));
    }
    match upper.as_str() {
        "M1" => Ok(1),
        "M3" => Ok(3),
        "M5" => Ok(5),
        "M15" => Ok(7),
        "M30" => Ok(8),
        "H1" => Ok(9),
        "H4" => Ok(10),
        "H12" => Ok(11),
        "D1" => Ok(12),
        "W1" => Ok(13),
        "MN1" => Ok(14),
        other => Err(anyhow!(
            "unsupported cTrader trendbar period label {} (canonical but unmapped)",
            other
        )),
    }
}

pub fn build_get_tick_data_request(
    ctid_trader_account_id: i64,
    symbol_id: i64,
    quote_type: i32,
    from_timestamp_ms: i64,
    to_timestamp_ms: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_id,
            "type": quote_type,
            "fromTimestamp": from_timestamp_ms,
            "toTimestamp": to_timestamp_ms,
        }),
    }
}

/// Build the JSON envelope for `ProtoOADealListByPositionIdReq`
/// (payload type 2179). Used by
/// [`crate::app_services::ctrader_history::fetch_deals_by_position_id`].
/// The proto carries `ctidTraderAccountId` (required) plus
/// `positionId` (required) and optional `fromTimestamp` / `toTimestamp`
/// time bounds that the broker uses to slice the lookup.
pub fn build_deal_list_by_position_id_request(
    ctid_trader_account_id: i64,
    position_id: i64,
    from_timestamp_ms: Option<i64>,
    to_timestamp_ms: Option<i64>,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    let mut payload = serde_json::json!({
        "ctidTraderAccountId": ctid_trader_account_id,
        "positionId": position_id,
    });
    if let Some(from_timestamp_ms) = from_timestamp_ms {
        payload["fromTimestamp"] = serde_json::json!(from_timestamp_ms);
    }
    if let Some(to_timestamp_ms) = to_timestamp_ms {
        payload["toTimestamp"] = serde_json::json!(to_timestamp_ms);
    }
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_DEAL_LIST_BY_POSITION_ID_REQUEST_PAYLOAD_TYPE,
        payload,
    }
}

/// Build the JSON envelope for `ProtoOAOrderListByPositionIdReq`
/// (payload type 2183). Same shape as
/// [`build_deal_list_by_position_id_request`] but returns the orders
/// rather than the deals tied to a `positionId`.
pub fn build_order_list_by_position_id_request(
    ctid_trader_account_id: i64,
    position_id: i64,
    from_timestamp_ms: Option<i64>,
    to_timestamp_ms: Option<i64>,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    let mut payload = serde_json::json!({
        "ctidTraderAccountId": ctid_trader_account_id,
        "positionId": position_id,
    });
    if let Some(from_timestamp_ms) = from_timestamp_ms {
        payload["fromTimestamp"] = serde_json::json!(from_timestamp_ms);
    }
    if let Some(to_timestamp_ms) = to_timestamp_ms {
        payload["toTimestamp"] = serde_json::json!(to_timestamp_ms);
    }
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_ORDER_LIST_BY_POSITION_ID_REQUEST_PAYLOAD_TYPE,
        payload,
    }
}

/// Build the JSON envelope for `ProtoOAOrderDetailsReq` (payload type
/// 2181). Carries only the two required fields: `ctidTraderAccountId`
/// and `orderId`.
pub fn build_order_details_request(
    ctid_trader_account_id: i64,
    order_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_ORDER_DETAILS_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "orderId": order_id,
        }),
    }
}

// **Phase D.1b (2026-05-28)** — note: `build_symbol_category_list_request`
// is now declared earlier in this file alongside its sibling
// `build_asset_class_list_request` so the two related RPCs are
// physically co-located. The pre-Phase-D definition that lived here
// has been merged upward.

pub fn parse_open_api_envelope(response_json: &str) -> Result<CTraderOpenApiJsonMessage> {
    serde_json::from_str(response_json).with_context(|| {
        // v0.4.13 — include the head of the offending body and total
        // length so the wizard's "OAuth error: …" surface has enough
        // signal to tell heartbeat-shaped frames apart from genuine
        // schema drifts. We cap at 200 chars to avoid leaking long
        // access tokens; cTrader access tokens are ~512 chars, so a
        // 200-char head still shows the envelope shape without ever
        // including the full token string.
        let total = response_json.len();
        let head: String = response_json.chars().take(200).collect();
        format!("failed to parse cTrader JSON envelope (len={total}, head={head:?})")
    })
}

pub fn expected_response_payload_type(request_payload_type: u32) -> Result<u32> {
    match request_payload_type {
        CTRADER_OA_APPLICATION_AUTH_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_ACCOUNT_AUTH_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_NEW_ORDER_REQUEST_PAYLOAD_TYPE
        | CTRADER_OA_CANCEL_ORDER_REQUEST_PAYLOAD_TYPE
        | CTRADER_OA_AMEND_ORDER_REQUEST_PAYLOAD_TYPE
        | CTRADER_OA_AMEND_POSITION_SLTP_REQUEST_PAYLOAD_TYPE
        | CTRADER_OA_CLOSE_POSITION_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE)
        }
        CTRADER_OA_TRADER_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_SUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_SUBSCRIBE_SPOTS_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_UNSUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_UNSUBSCRIBE_SPOTS_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_SYMBOLS_LIST_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_SYMBOL_BY_ID_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_DEAL_LIST_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_GET_POSITION_UNREALIZED_PNL_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_SYMBOL_CATEGORY_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_SYMBOL_CATEGORY_RESPONSE_PAYLOAD_TYPE)
        }
        // Phase D.1b (2026-05-28) — `ProtoOAAssetClassListReq/Res`,
        // used by the bootstrap to fetch the broker's top-level
        // asset class table so we can keep only Forex/Metals/
        // Indices/Commodities and drop the equity catalog.
        CTRADER_OA_ASSET_CLASS_LIST_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_ASSET_CLASS_LIST_RESPONSE_PAYLOAD_TYPE)
        }
        // Phase D.2a (2026-05-28) — `ProtoOAAssetListReq/Res`, the
        // per-currency asset registry. Joins LightSymbol.{base,quote}AssetId
        // to 3-letter codes (EUR, USD, XAU, ...).
        CTRADER_OA_ASSET_LIST_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_DEAL_LIST_BY_POSITION_ID_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_DEAL_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_ORDER_LIST_BY_POSITION_ID_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_ORDER_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_ORDER_DETAILS_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_ORDER_DETAILS_RESPONSE_PAYLOAD_TYPE)
        }
        // 2026-06-10 API-completeness pass — the remaining request/response
        // pairs from the full ProtoOAPayloadType catalog.
        CTRADER_OA_VERSION_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_VERSION_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_EXPECTED_MARGIN_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_EXPECTED_MARGIN_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_ORDER_LIST_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_ORDER_LIST_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_CASH_FLOW_HISTORY_LIST_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_CASH_FLOW_HISTORY_LIST_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_REFRESH_TOKEN_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_REFRESH_TOKEN_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_GET_CTID_PROFILE_BY_TOKEN_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_GET_CTID_PROFILE_BY_TOKEN_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_ACCOUNT_LOGOUT_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_ACCOUNT_LOGOUT_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_GET_DYNAMIC_LEVERAGE_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_GET_DYNAMIC_LEVERAGE_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_SYMBOLS_FOR_CONVERSION_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_SYMBOLS_FOR_CONVERSION_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_DEAL_OFFSET_LIST_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_DEAL_OFFSET_LIST_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_MARGIN_CALL_LIST_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_MARGIN_CALL_LIST_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_MARGIN_CALL_UPDATE_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_MARGIN_CALL_UPDATE_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_SUBSCRIBE_DEPTH_QUOTES_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_SUBSCRIBE_DEPTH_QUOTES_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_UNSUBSCRIBE_DEPTH_QUOTES_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_UNSUBSCRIBE_DEPTH_QUOTES_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE => Err(anyhow!(
            "cTrader spot events are push-only payloads and are not valid request messages"
        )),
        payload_type => Err(anyhow!(
            "unsupported cTrader request payload type: {}",
            payload_type
        )),
    }
}

pub fn is_matching_open_api_response(
    envelope: &CTraderOpenApiJsonMessage,
    request: &CTraderOpenApiJsonMessage,
    expected_payload_type: u32,
) -> bool {
    if envelope.payload_type == CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE {
        return false;
    }
    if matches!(
        request.payload_type,
        CTRADER_OA_NEW_ORDER_REQUEST_PAYLOAD_TYPE
            | CTRADER_OA_CANCEL_ORDER_REQUEST_PAYLOAD_TYPE
            | CTRADER_OA_AMEND_ORDER_REQUEST_PAYLOAD_TYPE
            | CTRADER_OA_AMEND_POSITION_SLTP_REQUEST_PAYLOAD_TYPE
            | CTRADER_OA_CLOSE_POSITION_REQUEST_PAYLOAD_TYPE
    ) {
        return envelope.client_msg_id == request.client_msg_id
            && matches!(
                envelope.payload_type,
                CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE | CTRADER_OA_ORDER_ERROR_EVENT_PAYLOAD_TYPE
            );
    }
    envelope.payload_type == expected_payload_type
        && envelope.client_msg_id == request.client_msg_id
}

pub fn parse_ctrader_error_payload(payload: &Value) -> Result<String> {
    let (_code, message) = parse_ctrader_error_payload_parts(payload)?;
    Ok(message)
}

pub fn parse_ctrader_error_payload_parts(payload: &Value) -> Result<(String, String)> {
    #[derive(Debug, Deserialize)]
    struct CTraderErrorPayload {
        #[serde(rename = "errorCode")]
        error_code: String,
        description: Option<String>,
    }

    let error: CTraderErrorPayload =
        serde_json::from_value(payload.clone()).context("failed to parse cTrader error payload")?;
    let formatted = match &error.description {
        Some(description) if !description.trim().is_empty() => {
            format!("{}: {}", error.error_code, description)
        }
        _ => error.error_code.clone(),
    };
    Ok((error.error_code, formatted))
}

/// Snapshot of a `ProtoOAAccountDisconnectEvent` (payload type 2164).
///
/// The proto carries only `ctidTraderAccountId` plus the implicit
/// `payloadType` discriminator; future field additions are tolerated by
/// the `#[serde(deny_unknown_fields)]` opt-out (we do not set it, so
/// unknown fields are silently ignored, matching prost's behaviour for
/// optional default-initialised fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CTraderAccountDisconnectEvent {
    pub ctid_trader_account_id: i64,
}

#[derive(Debug, Deserialize)]
struct AccountDisconnectEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: AccountDisconnectPayload,
}

#[derive(Debug, Deserialize)]
struct AccountDisconnectPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
}

/// Parses a `ProtoOAAccountDisconnectEvent` JSON envelope.
///
/// Errors if the envelope is not of type
/// [`CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE`] so callers can
/// reuse this on dispatch paths that also see spot events / errors.
pub fn parse_account_disconnect_event(
    response_json: &str,
) -> Result<CTraderAccountDisconnectEvent> {
    let envelope: AccountDisconnectEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader account disconnect event")?;
    if envelope.payload_type != CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader account disconnect payload type: {}",
            envelope.payload_type
        ));
    }
    Ok(CTraderAccountDisconnectEvent {
        ctid_trader_account_id: envelope.payload.ctid_trader_account_id,
    })
}

/// Per-position unrealized PnL row returned by
/// `ProtoOAGetPositionUnrealizedPnLRes`. Values are denoted in the
/// account deposit currency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CTraderPositionUnrealizedPnL {
    pub position_id: i64,
    pub gross_unrealized_pnl: f64,
    pub net_unrealized_pnl: f64,
}

/// Snapshot of a `ProtoOAGetPositionUnrealizedPnLRes` (payload type
/// 2188). `money_digits` is applied to convert the raw i64 fields into
/// account-currency f64 values.
#[derive(Debug, Clone, PartialEq)]
pub struct CTraderUnrealizedPnLSnapshot {
    pub account_id: i64,
    pub money_digits: u32,
    pub positions: Vec<CTraderPositionUnrealizedPnL>,
}

#[derive(Debug, Deserialize)]
struct UnrealizedPnLEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: UnrealizedPnLPayload,
}

#[derive(Debug, Deserialize)]
struct UnrealizedPnLPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
    #[serde(default, rename = "positionUnrealizedPnL")]
    position_unrealized_pnl: Vec<UnrealizedPnLRow>,
}

#[derive(Debug, Deserialize)]
struct UnrealizedPnLRow {
    #[serde(rename = "positionId")]
    position_id: i64,
    #[serde(rename = "grossUnrealizedPnL")]
    gross_unrealized_pnl: i64,
    #[serde(rename = "netUnrealizedPnL")]
    net_unrealized_pnl: i64,
}

/// Parses a `ProtoOAGetPositionUnrealizedPnLRes` JSON envelope.
///
/// `money_digits` is required on the wire; we treat its absence as a
/// hard error because the gross/net fields are otherwise un-scalable.
/// This matches `parse_trader_response`'s strict policy for the same
/// `moneyDigits` field on the trader payload.
pub fn parse_get_position_unrealized_pnl_response(
    response_json: &str,
) -> Result<CTraderUnrealizedPnLSnapshot> {
    let envelope: UnrealizedPnLEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader unrealized pnl response")?;
    if envelope.payload_type != CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader unrealized pnl payload type: {}",
            envelope.payload_type
        ));
    }
    let money_digits = envelope.payload.money_digits.ok_or_else(|| {
        anyhow!(
            "cTrader unrealized pnl response missing required moneyDigits field; \
             cannot scale gross/net PnL"
        )
    })?;
    // Centralised cTrader spec helper (see ctrader_money.rs and
    // docs/audits/research/ctrader_api_full_reference.md §5.14). For an
    // unrealized-pnl response we treat an out-of-spec `moneyDigits` as a
    // hard error (rather than the silent fallback used in display-only
    // paths) because the gross/net values feed the risk-gate drift audit.
    let positions = envelope
        .payload
        .position_unrealized_pnl
        .into_iter()
        .map(|row| -> Result<CTraderPositionUnrealizedPnL> {
            Ok(CTraderPositionUnrealizedPnL {
                position_id: row.position_id,
                gross_unrealized_pnl: crate::app_services::ctrader_money::scale_ctrader_money_int(
                    row.gross_unrealized_pnl,
                    money_digits as i32,
                )?,
                net_unrealized_pnl: crate::app_services::ctrader_money::scale_ctrader_money_int(
                    row.net_unrealized_pnl,
                    money_digits as i32,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(CTraderUnrealizedPnLSnapshot {
        account_id: envelope.payload.ctid_trader_account_id,
        money_digits,
        positions,
    })
}

/// True when a cTrader Open API error code indicates the OAuth access token
/// is no longer valid and a refresh + retry should be attempted before giving
/// up. Codes are matched case-insensitively against the patterns published by
/// Spotware's Open API (see Open API Bridge error codes documentation).
pub fn is_ctrader_auth_token_error(error_code: &str) -> bool {
    let upper = error_code.trim().to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "OA_AUTH_TOKEN_EXPIRED"
            | "CH_ACCESS_TOKEN_INVALID"
            | "CH_ACCESS_TOKEN_EXPIRED"
            | "INVALID_TOKEN"
            | "INVALID_ACCESS_TOKEN"
            | "ACCESS_TOKEN_EXPIRED"
            | "TOKEN_EXPIRED"
            | "EXPIRED_TOKEN"
    ) || upper.contains("TOKEN_EXPIRED")
        || upper.contains("ACCESS_TOKEN_INVALID")
        || upper.contains("INVALID_ACCESS_TOKEN")
}

/// Sentinel prefix that `ProductionCTraderExecutionBackend` uses to flag
/// auth-token failures so the trading-session caller can force-refresh and
/// retry once. Kept as a constant so caller and producer agree on the marker.
pub const CTRADER_TOKEN_EXPIRED_SENTINEL: &str = "CTRADER_TOKEN_EXPIRED";

impl CTraderOpenApiTransport for ProductionCTraderOpenApiTransport {
    fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> Result<Vec<String>> {
        crate::app_services::ctrader_tls::ensure_ctrader_rustls_provider();
        let url = format!("wss://{}:5036", self.endpoint_host);
        let (mut socket, _) = connect(url.as_str())
            .with_context(|| format!("failed to connect to cTrader endpoint {url}"))?;
        let mut responses = Vec::with_capacity(messages.len());

        for message in messages {
            let expected_payload_type = expected_response_payload_type(message.payload_type)?;
            let serialized = serde_json::to_string(message)
                .context("failed to serialize cTrader open api message")?;
            // DIAG: stamp request shape so we can see exactly what was sent
            // before we hit the response-count mismatch in the bridge.
            // Remove this block once the protocol-flow bug is closed.
            tracing::info!(
                target: "neoethos_app::ctrader_transport",
                request_msg_id = %message.client_msg_id,
                request_payload_type = message.payload_type,
                expected_response_payload_type = expected_payload_type,
                "send_sequence → ctrader"
            );
            socket
                .send(Message::Text(serialized.into()))
                .context("failed to send cTrader open api message")?;

            loop {
                match socket
                    .read()
                    .context("failed to read cTrader open api response")?
                {
                    Message::Text(text) => {
                        if text.trim().is_empty() {
                            return Err(anyhow!("empty cTrader open api response"));
                        }
                        let envelope = parse_open_api_envelope(text.as_ref())?;
                        // DIAG (remove with the trace above): log every incoming
                        // envelope so we can see whether non-matching responses
                        // are spot ticks, errors, or correctly-targeted replies
                        // with mismatched client_msg_id.
                        tracing::info!(
                            target: "neoethos_app::ctrader_transport",
                            recv_payload_type = envelope.payload_type,
                            recv_msg_id = %envelope.client_msg_id,
                            awaiting_payload_type = expected_payload_type,
                            awaiting_msg_id = %message.client_msg_id,
                            body_preview = %text.chars().take(220).collect::<String>(),
                            "send_sequence ← ctrader"
                        );
                        if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                            responses.push(text.to_string());
                            let _ = socket.close(None);
                            return Ok(responses);
                        }
                        if is_matching_open_api_response(&envelope, message, expected_payload_type)
                        {
                            responses.push(text.to_string());
                            break;
                        }
                    }
                    Message::Binary(bytes) => {
                        let text = String::from_utf8(bytes.to_vec())
                            .context("failed to decode cTrader binary response")?;
                        if text.trim().is_empty() {
                            return Err(anyhow!("empty cTrader open api response"));
                        }
                        let envelope = parse_open_api_envelope(&text)?;
                        if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                            responses.push(text);
                            let _ = socket.close(None);
                            return Ok(responses);
                        }
                        if is_matching_open_api_response(&envelope, message, expected_payload_type)
                        {
                            responses.push(text);
                            break;
                        }
                    }
                    Message::Ping(payload) => {
                        socket
                            .send(Message::Pong(payload))
                            .context("failed to reply to cTrader ping")?;
                    }
                    Message::Pong(_) => {}
                    Message::Close(_) => {
                        return Err(anyhow!("cTrader open api socket closed unexpectedly"));
                    }
                    Message::Frame(_) => {}
                }
            }
        }

        let _ = socket.close(None);
        Ok(responses)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transport selector — JSON-WSS (port 5036, default) vs Protobuf (port 5035).
//
// Per cTrader exhaustive docs sweep §10 item #3
// (`docs/audits/research/ctrader_api_full_reference.md`), the native
// Protobuf-over-TCP transport on port 5035 saves ~3× bandwidth compared
// to JSON-WSS and removes JSON field-name brittleness. The migration is
// staged: v0.4.5 ships the codec + reconcile + historical-bars and
// keeps order placement on JSON-WSS for a follow-up
// operator-acknowledged batch (the directive treats orders as
// money-critical).
//
// The opt-in is via the `NEOETHOS_BOT_CTRADER_TRANSPORT` environment
// variable (see `CTRADER_TRANSPORT_ENV_VAR`).
// ─────────────────────────────────────────────────────────────────────────────

/// Wire format selected for the cTrader Open API transport at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTraderTransportKind {
    /// Port 5036 — WebSocket + TLS + JSON envelopes. Default.
    JsonWss,
    /// Port 5035 — raw TCP + TLS + native Protobuf framing
    /// (4-byte big-endian length prefix + serialised `ProtoMessage`).
    /// Migrates the reconcile and historical-bars endpoints in v0.4.5;
    /// other endpoints fall back to JSON-WSS (the v0.4.5 batch scope).
    Protobuf,
}

impl CTraderTransportKind {
    /// Stable label suitable for `tracing` event fields.
    pub fn label(self) -> &'static str {
        match self {
            Self::JsonWss => "json_wss",
            Self::Protobuf => "protobuf",
        }
    }
}

/// Pick the cTrader transport based on `NEOETHOS_BOT_CTRADER_TRANSPORT`.
/// Returns `JsonWss` for unset / empty / unrecognised values (with a
/// `tracing::warn!` for unrecognised values so the operator can spot a
/// typo). Recognised values: `json_wss`, `protobuf` (case-insensitive,
/// trimmed).
pub fn select_ctrader_transport_from_env() -> CTraderTransportKind {
    match std::env::var(CTRADER_TRANSPORT_ENV_VAR) {
        Ok(raw) => {
            let normalised = raw.trim().to_ascii_lowercase();
            match normalised.as_str() {
                "" => CTraderTransportKind::JsonWss,
                "json_wss" | "json-wss" | "json" | "wss" => CTraderTransportKind::JsonWss,
                "protobuf" | "proto" | "pb" => {
                    tracing::info!(
                        target: "neoethos_app::ctrader",
                        transport = "protobuf",
                        "Using native Protobuf-over-TCP transport (3× bandwidth vs JSON-WSS)"
                    );
                    CTraderTransportKind::Protobuf
                }
                other => {
                    tracing::warn!(
                        target: "neoethos_app::ctrader",
                        value = other,
                        env_var = CTRADER_TRANSPORT_ENV_VAR,
                        "unrecognised cTrader transport value; defaulting to JSON-WSS"
                    );
                    CTraderTransportKind::JsonWss
                }
            }
        }
        Err(_) => CTraderTransportKind::JsonWss,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection entry points — public helpers exposing the per-wire-format
// dial semantics. The high-level transport type
// (`ProductionCTraderOpenApiTransport`) calls into these on each
// `send_sequence` invocation.
// ─────────────────────────────────────────────────────────────────────────────

/// Build the canonical JSON-WSS endpoint URL (port 5036, TLS WebSocket).
/// Used by [`ProductionCTraderOpenApiTransport::send_sequence`].
pub fn ctrader_json_wss_url(endpoint_host: &str) -> String {
    format!("wss://{}:5036", endpoint_host)
}

#[cfg(test)]
#[path = "ctrader_messages_tests.rs"]
mod tests;
