use crate::app_services::broker_deal_economics::BrokerPnlConversionFeeV1;
use crate::app_services::ctrader_data::{CTraderAssetInfo, parse_asset_list_response};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_CASH_FLOW_HISTORY_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_DEAL_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_EXPECTED_MARGIN_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_CTID_PROFILE_BY_TOKEN_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_ORDER_DETAILS_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_ORDER_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_ORDER_LIST_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_SYMBOL_CATEGORY_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_VERSION_RESPONSE_PAYLOAD_TYPE, CTraderDealListRequest, CTraderOpenApiTransport,
    CTraderPositionUnrealizedPnL, CTraderUnrealizedPnLSnapshot, ProductionCTraderOpenApiTransport,
    build_account_auth_request, build_application_auth_request, build_asset_list_request,
    build_deal_list_request, build_get_position_unrealized_pnl_request, build_reconcile_request,
    build_trader_request, parse_ctrader_error_payload, parse_get_position_unrealized_pnl_response,
    parse_open_api_envelope,
};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_CTRADER_DEAL_LOOKBACK_HOURS: i64 = 24;
const DEFAULT_CTRADER_DEAL_MAX_ROWS: i32 = 100;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderAccountRuntimeRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
    pub account_id: String,
    pub return_protection_orders: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderTraderSnapshot {
    pub account_id: i64,
    pub balance: f64,
    pub leverage: Option<f64>,
    pub trader_login: Option<i64>,
    pub account_type: Option<String>,
    pub broker_name: Option<String>,
    pub money_digits: u32,
    /// Numeric asset id of the deposit currency (e.g. 6 = EUR, 8 = USD,
    /// 4 = GBP). Comes from `ProtoOATrader.depositAssetId`. Used by the
    /// bridge to render the right currency symbol on the dashboard.
    /// `None` only when the broker omitted the field (rare).
    pub deposit_asset_id: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderPositionSnapshot {
    pub position_id: i64,
    pub symbol_id: i64,
    pub trade_side: String,
    pub volume: f64,
    pub open_timestamp_ms: Option<i64>,
    pub price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub swap: Option<f64>,
    pub commission: Option<f64>,
    pub mirroring_commission: Option<f64>,
    pub used_margin: Option<f64>,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub client_order_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderPendingOrderSnapshot {
    pub order_id: i64,
    pub symbol_id: i64,
    pub trade_side: String,
    pub order_type: String,
    pub volume: f64,
    pub open_timestamp_ms: Option<i64>,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub client_order_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderDealSnapshot {
    pub account_id: i64,
    pub deal_id: i64,
    pub order_id: i64,
    pub position_id: i64,
    pub symbol_id: i64,
    pub trade_side: String,
    pub deal_status: String,
    pub volume: f64,
    pub filled_volume: f64,
    /// Exact `ProtoOADeal.filledVolume` wire integer. cTrader volume fields are
    /// centi-units; standard lots require the same symbol's broker `lotSize`.
    pub filled_volume_raw_centi_units: i64,
    pub execution_timestamp_ms: i64,
    pub execution_price: Option<f64>,
    pub entry_price: Option<f64>,
    pub gross_profit: Option<f64>,
    pub fee: Option<f64>,
    pub swap: Option<f64>,
    pub pnl_conversion_fee: Option<f64>,
    /// Scale carried by `ProtoOAClosePositionDetail` for all closing money.
    pub money_digits: Option<u32>,
    pub gross_profit_raw_scaled: Option<i64>,
    pub commission_raw_scaled_signed: Option<i64>,
    pub swap_raw_scaled_signed: Option<i64>,
    pub pnl_conversion_fee_state: Option<BrokerPnlConversionFeeV1>,
    /// Checked sum of signed broker components. This is derived locally and is
    /// deliberately not named or treated as an independently reported net.
    pub component_sum_account_currency: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderReconcileSnapshot {
    pub account_id: i64,
    pub positions: Vec<CTraderPositionSnapshot>,
    pub pending_orders: Vec<CTraderPendingOrderSnapshot>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderAccountRuntimeSnapshot {
    pub environment: CTraderEnvironment,
    pub trader: CTraderTraderSnapshot,
    pub reconcile: CTraderReconcileSnapshot,
    pub recent_deals: Vec<CTraderDealSnapshot>,
    /// Sum of the broker-supplied `ProtoOAGetPositionUnrealizedPnLRes`
    /// `netUnrealizedPnL` rows in account currency. Runtime loading fails when
    /// this response or its scale is absent.
    pub unrealized_pnl: f64,
    /// The same validated broker rows keyed by position id. Keeping these in
    /// the account snapshot prevents a second, time-skewed PnL request in UI
    /// and risk consumers.
    pub unrealized_pnl_by_position: BTreeMap<i64, CTraderPositionUnrealizedPnL>,
    /// Broker-canonical deposit asset name resolved from the same account's
    /// `ProtoOAAssetListRes`. No numeric-id lookup table or fiat fallback is
    /// permitted because a wrong label changes the meaning of every monetary
    /// value in this snapshot.
    pub deposit_asset_name: String,
}

// The account-runtime backend (trait + production impl) loads trader balance /
// reconcile / recent-deals for the live account panel. `dead_code` because the
// production server doesn't yet poll account runtime through this seam — the
// live engine wires it in Phase 2-5. Exercised today only by the stub in tests.
#[allow(dead_code)]
pub trait CTraderAccountRuntimeBackend: Send + Sync {
    fn load_account_runtime(
        &self,
        request: &CTraderAccountRuntimeRequest,
    ) -> Result<CTraderAccountRuntimeSnapshot>;
}

#[allow(dead_code)]
#[derive(Clone, Default)]
pub struct ProductionCTraderAccountRuntimeBackend;

#[cfg(test)]
#[derive(Clone)]
pub struct StubCTraderAccountRuntimeBackend {
    outcome: Arc<Mutex<Option<Result<CTraderAccountRuntimeSnapshot, String>>>>,
    last_request: Arc<Mutex<Option<CTraderAccountRuntimeRequest>>>,
}

#[derive(Debug, Deserialize)]
struct TraderEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: TraderPayload,
}

/// Wire shape for `ProtoOATraderRes.payload` per the cTrader Open API.
///
/// The outer payload carries a nested `trader: { ... }` object that
/// holds balance / leverage / login. Pre-`82b075` we tried to read
/// those fields at the top level (matching no documented shape), which
/// silently failed `parse_trader_response` with `failed to parse cTrader
/// trader response` and propagated as `EQUITY $0.00` on the dashboard
/// — the bug that finally surfaced when we got an end-to-end successful
/// reply path from a sandbox account.
#[derive(Debug, Deserialize)]
struct TraderPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    trader: TraderInfo,
}

#[derive(Debug, Deserialize)]
struct TraderInfo {
    balance: i64,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
    #[serde(rename = "leverageInCents")]
    leverage_in_cents: Option<u32>,
    #[serde(rename = "traderLogin")]
    trader_login: Option<i64>,
    #[serde(rename = "accountType")]
    account_type: Option<i32>,
    #[serde(rename = "brokerName")]
    broker_name: Option<String>,
    /// Numeric asset id referenced into `ProtoOAAssetListReq`'s
    /// catalog. Well-known values: 4=GBP, 5=CHF, 6=EUR, 8=USD,
    /// 14=JPY. Used by the bridge to render the dashboard
    /// currency symbol without an extra round-trip to the asset
    /// list endpoint (#144).
    #[serde(rename = "depositAssetId")]
    deposit_asset_id: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ReconcileEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: ReconcilePayload,
}

#[derive(Debug, Deserialize)]
struct ReconcilePayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    #[serde(default, rename = "position")]
    positions: Vec<PositionPayload>,
    #[serde(default, rename = "order")]
    orders: Vec<OrderPayload>,
}

#[derive(Debug, Deserialize)]
struct DealListEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: DealListPayload,
}

#[derive(Debug, Deserialize)]
struct DealListPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    #[serde(default, rename = "deal")]
    deals: Vec<DealPayload>,
}

#[derive(Debug, Deserialize)]
struct OrderListByPositionEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: OrderListByPositionPayload,
}

#[derive(Debug, Deserialize)]
struct OrderListByPositionPayload {
    #[serde(default, rename = "order")]
    orders: Vec<OrderPayload>,
}

#[derive(Debug, Deserialize)]
struct OrderDetailsEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: OrderDetailsPayload,
}

#[derive(Debug, Deserialize)]
struct OrderDetailsPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    order: OrderPayload,
    #[serde(default, rename = "deal")]
    deals: Vec<DealPayload>,
}

#[derive(Debug, Deserialize)]
struct SymbolCategoryEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: SymbolCategoryPayload,
}

#[derive(Debug, Deserialize)]
struct SymbolCategoryPayload {
    #[serde(default, rename = "symbolCategory")]
    symbol_category: Vec<SymbolCategoryRow>,
}

#[derive(Debug, Deserialize)]
struct SymbolCategoryRow {
    id: i64,
    #[serde(rename = "assetClassId")]
    asset_class_id: i64,
    name: String,
    #[serde(rename = "sortingNumber")]
    sorting_number: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct PositionPayload {
    #[serde(rename = "positionId")]
    position_id: i64,
    #[serde(rename = "tradeData")]
    trade_data: TradeDataPayload,
    price: Option<f64>,
    #[serde(rename = "stopLoss")]
    stop_loss: Option<f64>,
    #[serde(rename = "takeProfit")]
    take_profit: Option<f64>,
    swap: Option<i64>,
    commission: Option<i64>,
    #[serde(rename = "mirroringCommission")]
    mirroring_commission: Option<i64>,
    #[serde(rename = "usedMargin")]
    used_margin: Option<u64>,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OrderPayload {
    #[serde(rename = "orderId")]
    order_id: i64,
    #[serde(rename = "tradeData")]
    trade_data: TradeDataPayload,
    #[serde(rename = "orderType")]
    order_type: i32,
    #[serde(rename = "limitPrice")]
    limit_price: Option<f64>,
    #[serde(rename = "stopPrice")]
    stop_price: Option<f64>,
    #[serde(rename = "stopLoss")]
    stop_loss: Option<f64>,
    #[serde(rename = "takeProfit")]
    take_profit: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct TradeDataPayload {
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    volume: i64,
    #[serde(rename = "tradeSide")]
    trade_side: i32,
    #[serde(rename = "openTimestamp")]
    open_timestamp: Option<i64>,
    label: Option<String>,
    comment: Option<String>,
    /// Echo of `clientOrderId` that the bot sent on the original order.
    /// SECURITY (audit-fix F3): the duplicate-order pre-retry check uses
    /// this field to detect that a previous attempt was already accepted
    /// by the broker before the network timeout, so retries don't double
    /// the position.
    #[serde(default, rename = "clientOrderId")]
    client_order_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DealPayload {
    #[serde(rename = "dealId")]
    deal_id: i64,
    #[serde(rename = "orderId")]
    order_id: i64,
    #[serde(rename = "positionId")]
    position_id: i64,
    volume: i64,
    #[serde(rename = "filledVolume")]
    filled_volume: i64,
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    #[serde(rename = "executionTimestamp")]
    execution_timestamp: i64,
    #[serde(rename = "executionPrice")]
    execution_price: Option<f64>,
    #[serde(rename = "tradeSide")]
    trade_side: i32,
    #[serde(rename = "dealStatus")]
    deal_status: i32,
    commission: Option<i64>,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
    #[serde(rename = "closePositionDetail")]
    close_position_detail: Option<ClosePositionDetailPayload>,
}

#[derive(Debug, Deserialize)]
struct ClosePositionDetailPayload {
    #[serde(rename = "entryPrice")]
    entry_price: Option<f64>,
    #[serde(rename = "grossProfit")]
    gross_profit: i64,
    swap: i64,
    commission: i64,
    #[serde(rename = "pnlConversionFee")]
    pnl_conversion_fee: Option<i64>,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
}

pub fn parse_trader_response(response_json: &str) -> Result<CTraderTraderSnapshot> {
    let envelope: TraderEnvelope =
        serde_json::from_str(response_json).context("failed to parse cTrader trader response")?;
    if envelope.payload_type != CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader trader payload type: {}",
            envelope.payload_type
        ));
    }

    let trader = envelope.payload.trader;
    let money_digits = required_money_digits(trader.money_digits, "trader.money_digits")?;
    Ok(CTraderTraderSnapshot {
        account_id: envelope.payload.ctid_trader_account_id,
        balance: scaled_money(trader.balance, money_digits)?,
        leverage: trader.leverage_in_cents.map(|value| value as f64 / 100.0),
        trader_login: trader.trader_login,
        account_type: trader.account_type.map(account_type_label),
        broker_name: trader.broker_name,
        money_digits,
        deposit_asset_id: trader.deposit_asset_id,
    })
}

pub fn parse_reconcile_response(response_json: &str) -> Result<CTraderReconcileSnapshot> {
    let envelope: ReconcileEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader reconcile response")?;
    if envelope.payload_type != CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader reconcile payload type: {}",
            envelope.payload_type
        ));
    }

    Ok(CTraderReconcileSnapshot {
        account_id: envelope.payload.ctid_trader_account_id,
        positions: envelope
            .payload
            .positions
            .into_iter()
            .map(|position| -> Result<CTraderPositionSnapshot> {
                let (swap, commission, mirroring_commission, used_margin) =
                    if position.swap.is_some()
                        || position.commission.is_some()
                        || position.mirroring_commission.is_some()
                        || position.used_margin.is_some()
                    {
                        let money_digits =
                            required_money_digits(position.money_digits, "position.money_digits")?;
                        (
                            position
                                .swap
                                .map(|raw| scaled_money(raw, money_digits))
                                .transpose()?,
                            position
                                .commission
                                .map(|raw| scaled_money(raw, money_digits))
                                .transpose()?,
                            position
                                .mirroring_commission
                                .map(|raw| scaled_money(raw, money_digits))
                                .transpose()?,
                            position
                                .used_margin
                                .map(|raw| scaled_unsigned_money(raw, money_digits))
                                .transpose()?,
                        )
                    } else {
                        (None, None, None, None)
                    };
                Ok(CTraderPositionSnapshot {
                    swap,
                    commission,
                    mirroring_commission,
                    used_margin,
                    position_id: position.position_id,
                    symbol_id: position.trade_data.symbol_id,
                    trade_side: trade_side_label(position.trade_data.trade_side),
                    volume: volume_to_units(position.trade_data.volume),
                    open_timestamp_ms: position.trade_data.open_timestamp,
                    price: position.price,
                    stop_loss: position.stop_loss,
                    take_profit: position.take_profit,
                    label: position.trade_data.label,
                    comment: position.trade_data.comment,
                    client_order_id: position.trade_data.client_order_id,
                })
            })
            .collect::<Result<Vec<_>>>()?,
        pending_orders: envelope
            .payload
            .orders
            .into_iter()
            .map(|order| CTraderPendingOrderSnapshot {
                order_id: order.order_id,
                symbol_id: order.trade_data.symbol_id,
                trade_side: trade_side_label(order.trade_data.trade_side),
                order_type: order_type_label(order.order_type),
                volume: volume_to_units(order.trade_data.volume),
                open_timestamp_ms: order.trade_data.open_timestamp,
                limit_price: order.limit_price,
                stop_price: order.stop_price,
                stop_loss: order.stop_loss,
                take_profit: order.take_profit,
                label: order.trade_data.label,
                comment: order.trade_data.comment,
                client_order_id: order.trade_data.client_order_id,
            })
            .collect(),
    })
}

pub fn parse_deal_list_response(response_json: &str) -> Result<Vec<CTraderDealSnapshot>> {
    let envelope: DealListEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader deal list response")?;
    if envelope.payload_type != CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader deal list payload type: {}",
            envelope.payload_type
        ));
    }
    let account_id = envelope.payload.ctid_trader_account_id;
    Ok(envelope
        .payload
        .deals
        .into_iter()
        .map(|deal| deal_payload_to_snapshot(account_id, deal))
        .collect::<Result<Vec<_>>>()?)
}

/// Parse a `ProtoOADealListByPositionIdRes` (payload type 2180).
///
/// Same per-deal shape as [`parse_deal_list_response`] but the broker
/// gates the response on a single `positionId`. New in the 2026-05-14
/// upstream proto refresh; see
/// `docs/audits/research/spotware_proto_new_messages.md` group D.
pub fn parse_deal_list_by_position_id_response(
    response_json: &str,
) -> Result<Vec<CTraderDealSnapshot>> {
    let envelope: DealListEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader deal-list-by-position response")?;
    if envelope.payload_type != CTRADER_OA_DEAL_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader deal-list-by-position payload type: {}",
            envelope.payload_type
        ));
    }
    let account_id = envelope.payload.ctid_trader_account_id;
    Ok(envelope
        .payload
        .deals
        .into_iter()
        .map(|deal| deal_payload_to_snapshot(account_id, deal))
        .collect::<Result<Vec<_>>>()?)
}

/// Parse a `ProtoOAOrderListByPositionIdRes` (payload type 2184). The
/// per-order shape is the same `ProtoOAOrder` carried by
/// `ProtoOAReconcileRes`, so this reuses [`OrderPayload`] internally
/// and emits the same [`CTraderPendingOrderSnapshot`] type.
pub fn parse_order_list_by_position_id_response(
    response_json: &str,
) -> Result<Vec<CTraderPendingOrderSnapshot>> {
    let envelope: OrderListByPositionEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader order-list-by-position response")?;
    if envelope.payload_type != CTRADER_OA_ORDER_LIST_BY_POSITION_ID_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader order-list-by-position payload type: {}",
            envelope.payload_type
        ));
    }
    Ok(envelope
        .payload
        .orders
        .into_iter()
        .map(order_payload_to_snapshot)
        .collect())
}

/// Snapshot of a `ProtoOAOrderDetailsRes` (payload type 2182). Carries
/// the single requested order plus all of its child deals. New in the
/// 2026-05-14 upstream proto refresh; see
/// `docs/audits/research/spotware_proto_new_messages.md` group D.
#[derive(Debug, Clone, PartialEq)]
pub struct CTraderOrderDetailsSnapshot {
    pub account_id: i64,
    pub order: CTraderPendingOrderSnapshot,
    pub deals: Vec<CTraderDealSnapshot>,
}

/// Parse a `ProtoOAOrderDetailsRes` JSON envelope.
pub fn parse_order_details_response(response_json: &str) -> Result<CTraderOrderDetailsSnapshot> {
    let envelope: OrderDetailsEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader order-details response")?;
    if envelope.payload_type != CTRADER_OA_ORDER_DETAILS_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader order-details payload type: {}",
            envelope.payload_type
        ));
    }
    let account_id = envelope.payload.ctid_trader_account_id;
    Ok(CTraderOrderDetailsSnapshot {
        account_id,
        order: order_payload_to_snapshot(envelope.payload.order),
        deals: envelope
            .payload
            .deals
            .into_iter()
            .map(|deal| deal_payload_to_snapshot(account_id, deal))
            .collect::<Result<Vec<_>>>()?,
    })
}

/// Symbol-category row returned by `ProtoOASymbolCategoryListRes`.
/// See `OpenApiModelMessages.proto::ProtoOASymbolCategory`.
#[derive(Debug, Clone, PartialEq)]
pub struct CTraderSymbolCategorySnapshot {
    pub id: i64,
    pub asset_class_id: i64,
    pub name: String,
    pub sorting_number: Option<f64>,
}

/// Parse a `ProtoOASymbolCategoryListRes` JSON envelope (payload type
/// 2161). New in the 2026-05-14 upstream proto refresh; see
/// `docs/audits/research/spotware_proto_new_messages.md` group G.
pub fn parse_symbol_category_list_response(
    response_json: &str,
) -> Result<Vec<CTraderSymbolCategorySnapshot>> {
    let envelope: SymbolCategoryEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader symbol-category list response")?;
    if envelope.payload_type != CTRADER_OA_SYMBOL_CATEGORY_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader symbol-category payload type: {}",
            envelope.payload_type
        ));
    }
    Ok(envelope
        .payload
        .symbol_category
        .into_iter()
        .map(|category| CTraderSymbolCategorySnapshot {
            id: category.id,
            asset_class_id: category.asset_class_id,
            name: category.name,
            sorting_number: category.sorting_number,
        })
        .collect())
}

fn deal_payload_to_snapshot(account_id: i64, deal: DealPayload) -> Result<CTraderDealSnapshot> {
    let (
        gross_profit,
        close_fee,
        swap,
        pnl_conversion_fee,
        close_money_digits,
        gross_profit_raw_scaled,
        close_commission_raw_scaled_signed,
        swap_raw_scaled_signed,
        pnl_conversion_fee_state,
        component_sum_account_currency,
    ) = match deal.close_position_detail.as_ref() {
        Some(detail) => {
            let digits = required_money_digits(detail.money_digits, "deal.close.money_digits")?;
            let conversion_state = match detail.pnl_conversion_fee {
                Some(raw_scaled_signed) => BrokerPnlConversionFeeV1::Charged { raw_scaled_signed },
                None => BrokerPnlConversionFeeV1::NotApplied,
            };
            let component_sum_raw_scaled = detail
                .gross_profit
                .checked_add(detail.commission)
                .and_then(|sum| sum.checked_add(detail.swap))
                .and_then(|sum| sum.checked_add(conversion_state.raw_scaled_signed()))
                .ok_or_else(|| anyhow!("cTrader closing-deal money components overflow i64"))?;
            (
                Some(scaled_money(detail.gross_profit, digits)?),
                Some(scaled_money(detail.commission, digits)?),
                Some(scaled_money(detail.swap, digits)?),
                detail
                    .pnl_conversion_fee
                    .map(|fee| scaled_money(fee, digits))
                    .transpose()?,
                Some(digits),
                Some(detail.gross_profit),
                Some(detail.commission),
                Some(detail.swap),
                Some(conversion_state),
                Some(scaled_money(component_sum_raw_scaled, digits)?),
            )
        }
        None => (None, None, None, None, None, None, None, None, None, None),
    };
    let fee = match close_fee {
        Some(value) => Some(value),
        None => match deal.commission {
            Some(commission) => {
                let digits = required_money_digits(deal.money_digits, "deal.money_digits")?;
                Some(scaled_money(commission, digits)?)
            }
            None => None,
        },
    };
    let (money_digits, commission_raw_scaled_signed) = match close_money_digits {
        Some(digits) => (Some(digits), close_commission_raw_scaled_signed),
        None => match deal.commission {
            Some(commission) => (
                Some(required_money_digits(
                    deal.money_digits,
                    "deal.money_digits",
                )?),
                Some(commission),
            ),
            None => (None, None),
        },
    };

    Ok(CTraderDealSnapshot {
        account_id,
        deal_id: deal.deal_id,
        order_id: deal.order_id,
        position_id: deal.position_id,
        symbol_id: deal.symbol_id,
        trade_side: trade_side_label(deal.trade_side),
        deal_status: deal_status_label(deal.deal_status),
        volume: volume_to_units(deal.volume),
        filled_volume: volume_to_units(deal.filled_volume),
        filled_volume_raw_centi_units: deal.filled_volume,
        execution_timestamp_ms: deal.execution_timestamp,
        execution_price: deal.execution_price,
        entry_price: deal
            .close_position_detail
            .as_ref()
            .and_then(|detail| detail.entry_price),
        gross_profit,
        fee,
        swap,
        pnl_conversion_fee,
        money_digits,
        gross_profit_raw_scaled,
        commission_raw_scaled_signed,
        swap_raw_scaled_signed,
        pnl_conversion_fee_state,
        component_sum_account_currency,
    })
}

fn order_payload_to_snapshot(order: OrderPayload) -> CTraderPendingOrderSnapshot {
    CTraderPendingOrderSnapshot {
        order_id: order.order_id,
        symbol_id: order.trade_data.symbol_id,
        trade_side: trade_side_label(order.trade_data.trade_side),
        order_type: order_type_label(order.order_type),
        volume: volume_to_units(order.trade_data.volume),
        open_timestamp_ms: order.trade_data.open_timestamp,
        limit_price: order.limit_price,
        stop_price: order.stop_price,
        stop_loss: order.stop_loss,
        take_profit: order.take_profit,
        label: order.trade_data.label,
        comment: order.trade_data.comment,
        client_order_id: order.trade_data.client_order_id,
    }
}

pub fn load_account_runtime_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderAccountRuntimeRequest,
) -> Result<CTraderAccountRuntimeSnapshot> {
    let account_id = request
        .account_id
        .parse::<i64>()
        .context("cTrader account id must be numeric")?;
    // Resilient: retry transient cold-connection / CANT_ROUTE failures and
    // surface the real cTrader error instead of a misleading "received N"
    // count. All 7 responses are required for a broker-truthful account
    // snapshot, including authoritative unrealized PnL and deposit-asset name.
    let responses = crate::app_services::ctrader_messages::send_sequence_resilient(
        transport,
        &[
            build_application_auth_request(
                &request.client_id,
                &request.client_secret,
                "app-auth-1",
            ),
            build_account_auth_request(account_id, &request.access_token, "account-auth-1"),
            build_trader_request(account_id, "trader-1"),
            build_reconcile_request(account_id, request.return_protection_orders, "reconcile-1"),
            build_deal_list_request(
                &CTraderDealListRequest {
                    account_id,
                    from_timestamp_ms: Some(
                        current_unix_millis()?
                            - DEFAULT_CTRADER_DEAL_LOOKBACK_HOURS * 60 * 60 * 1000,
                    ),
                    to_timestamp_ms: Some(current_unix_millis()?),
                    max_rows: Some(DEFAULT_CTRADER_DEAL_MAX_ROWS),
                },
                "deals-1",
            ),
            build_get_position_unrealized_pnl_request(account_id, "unrealized-pnl-1"),
            build_asset_list_request(account_id, "asset-list-1"),
        ],
        7,
        "cTrader account runtime",
    )?;
    if responses.len() != 7 {
        return Err(anyhow!(
            "expected 7 cTrader account runtime responses, received {}",
            responses.len()
        ));
    }

    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(&responses[2], CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(&responses[3], CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(&responses[4], CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(
        &responses[5],
        CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[6], CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE)?;

    let trader = parse_trader_response(&responses[2])?;
    let reconcile = parse_reconcile_response(&responses[3])?;
    let assets = parse_asset_list_response(&responses[6])?;
    let deposit_asset_name = resolve_deposit_asset_name(&trader, &assets)?;
    let broker_pnl = parse_get_position_unrealized_pnl_response(&responses[5])?;
    let unrealized_pnl_by_position = reconcile_broker_unrealized_pnl(&reconcile, &broker_pnl)?;
    let unrealized_pnl = unrealized_pnl_by_position
        .values()
        .map(|position| position.net_unrealized_pnl)
        .sum();
    let recent_deals = parse_deal_list_response(&responses[4])?;
    if trader.account_id != account_id
        || reconcile.account_id != account_id
        || recent_deals
            .iter()
            .any(|deal| deal.account_id != account_id)
    {
        return Err(anyhow!(
            "cTrader account-runtime response identity differs from requested account {account_id}"
        ));
    }

    Ok(CTraderAccountRuntimeSnapshot {
        environment: request.environment,
        trader,
        reconcile,
        recent_deals,
        unrealized_pnl,
        unrealized_pnl_by_position,
        deposit_asset_name,
    })
}

fn resolve_deposit_asset_name(
    trader: &CTraderTraderSnapshot,
    assets: &[CTraderAssetInfo],
) -> Result<String> {
    let deposit_asset_id = trader
        .deposit_asset_id
        .ok_or_else(|| anyhow!("cTrader trader response omitted required depositAssetId"))?;
    let mut matches = assets
        .iter()
        .filter(|asset| asset.asset_id == deposit_asset_id);
    let asset = matches.next().ok_or_else(|| {
        anyhow!("cTrader asset registry has no row for depositAssetId={deposit_asset_id}")
    })?;
    if matches.next().is_some() {
        return Err(anyhow!(
            "cTrader asset registry contains duplicate rows for depositAssetId={deposit_asset_id}"
        ));
    }
    if asset.name.trim().is_empty() {
        return Err(anyhow!(
            "cTrader asset registry row for depositAssetId={deposit_asset_id} has an empty name"
        ));
    }
    Ok(asset.name.clone())
}

fn reconcile_broker_unrealized_pnl(
    reconcile: &CTraderReconcileSnapshot,
    pnl: &CTraderUnrealizedPnLSnapshot,
) -> Result<BTreeMap<i64, CTraderPositionUnrealizedPnL>> {
    if pnl.account_id != reconcile.account_id {
        return Err(anyhow!(
            "cTrader unrealized-PnL account {} does not match reconcile account {}",
            pnl.account_id,
            reconcile.account_id
        ));
    }

    let mut open_ids = BTreeSet::new();
    for position in &reconcile.positions {
        if !open_ids.insert(position.position_id) {
            return Err(anyhow!(
                "cTrader reconcile returned duplicate position {}",
                position.position_id
            ));
        }
    }

    let mut pnl_ids = BTreeSet::new();
    let mut by_position = BTreeMap::new();
    for position in &pnl.positions {
        if !pnl_ids.insert(position.position_id) {
            return Err(anyhow!(
                "cTrader unrealized-PnL response returned duplicate position {}",
                position.position_id
            ));
        }
        if !open_ids.contains(&position.position_id) {
            return Err(anyhow!(
                "cTrader unrealized-PnL response returned unknown position {}",
                position.position_id
            ));
        }
        if !position.gross_unrealized_pnl.is_finite() || !position.net_unrealized_pnl.is_finite() {
            return Err(anyhow!(
                "cTrader unrealized-PnL response returned non-finite values for position {}",
                position.position_id
            ));
        }
        by_position.insert(position.position_id, *position);
    }
    if pnl_ids != open_ids {
        let missing: Vec<i64> = open_ids.difference(&pnl_ids).copied().collect();
        return Err(anyhow!(
            "cTrader unrealized-PnL response omitted open positions: {missing:?}"
        ));
    }
    let total: f64 = by_position
        .values()
        .map(|position| position.net_unrealized_pnl)
        .sum();
    if !total.is_finite() {
        return Err(anyhow!("cTrader unrealized-PnL total is not finite"));
    }
    Ok(by_position)
}

pub fn load_account_runtime(
    request: &CTraderAccountRuntimeRequest,
) -> Result<CTraderAccountRuntimeSnapshot> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    load_account_runtime_with_transport(&transport, request)
}

impl CTraderAccountRuntimeBackend for ProductionCTraderAccountRuntimeBackend {
    fn load_account_runtime(
        &self,
        request: &CTraderAccountRuntimeRequest,
    ) -> Result<CTraderAccountRuntimeSnapshot> {
        load_account_runtime(request)
    }
}

#[cfg(test)]
impl StubCTraderAccountRuntimeBackend {
    pub fn success(snapshot: CTraderAccountRuntimeSnapshot) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Ok(snapshot)))),
            last_request: Arc::new(Mutex::new(None)),
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Err(message.into())))),
            last_request: Arc::new(Mutex::new(None)),
        }
    }

    pub fn last_request(&self) -> Option<CTraderAccountRuntimeRequest> {
        self.last_request.lock().expect("last request lock").clone()
    }
}

#[cfg(test)]
impl CTraderAccountRuntimeBackend for StubCTraderAccountRuntimeBackend {
    fn load_account_runtime(
        &self,
        request: &CTraderAccountRuntimeRequest,
    ) -> Result<CTraderAccountRuntimeSnapshot> {
        *self.last_request.lock().expect("last request lock") = Some(request.clone());
        let outcome = self
            .outcome
            .lock()
            .expect("runtime outcome lock")
            .take()
            .unwrap_or_else(|| Err("stub cTrader account runtime backend exhausted".to_string()));
        outcome.map_err(|message| anyhow!(message))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2026-06-10 — cTrader Open API response consumers (operator: "make the whole
// API part of the bot"). Field names + scaling rules verified against the
// upstream Spotware proto (spotware/openapi-proto-messages:
// OpenApiMessages.proto / OpenApiModelMessages.proto - the local vendored copy
// under crates/neoethos-app/proto/ was deleted 2026-08-09, batch D2). The
// request builders + req→res map already exist in ctrader_messages.rs; this
// block adds the response parsers. Public structs derive Serialize + camelCase
// so the server handlers can return them directly.
// ═══════════════════════════════════════════════════════════════════════════

// ─── ProtoOAOrderListRes (2176) — account-wide order history ───────────────

#[derive(Debug, Deserialize)]
struct OrderListEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: OrderListPayload,
}
#[derive(Debug, Deserialize)]
struct OrderListPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    #[serde(default)]
    order: Vec<HistoricalOrderPayload>,
    #[serde(rename = "hasMore", default)]
    has_more: bool,
}
#[derive(Debug, Deserialize)]
struct HistoricalOrderPayload {
    #[serde(rename = "orderId")]
    order_id: i64,
    #[serde(rename = "tradeData")]
    trade_data: HistoricalTradeDataPayload,
    #[serde(rename = "orderType")]
    order_type: i32,
    #[serde(rename = "orderStatus")]
    order_status: i32,
    #[serde(rename = "expirationTimestamp")]
    expiration_timestamp: Option<i64>,
    #[serde(rename = "executionPrice")]
    execution_price: Option<f64>,
    #[serde(rename = "executedVolume")]
    executed_volume: Option<i64>,
    #[serde(rename = "utcLastUpdateTimestamp")]
    utc_last_update_timestamp: Option<i64>,
    #[serde(rename = "limitPrice")]
    limit_price: Option<f64>,
    #[serde(rename = "stopPrice")]
    stop_price: Option<f64>,
    #[serde(rename = "stopLoss")]
    stop_loss: Option<f64>,
    #[serde(rename = "takeProfit")]
    take_profit: Option<f64>,
    #[serde(rename = "clientOrderId")]
    client_order_id: Option<String>,
    #[serde(rename = "timeInForce")]
    time_in_force: Option<i32>,
    #[serde(rename = "positionId")]
    position_id: Option<i64>,
    #[serde(rename = "closingOrder")]
    closing_order: Option<bool>,
    #[serde(rename = "isStopOut")]
    is_stop_out: Option<bool>,
    #[serde(rename = "trailingStopLoss")]
    trailing_stop_loss: Option<bool>,
}
#[derive(Debug, Deserialize)]
struct HistoricalTradeDataPayload {
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    volume: i64,
    #[serde(rename = "tradeSide")]
    trade_side: i32,
    #[serde(rename = "openTimestamp")]
    open_timestamp: Option<i64>,
    label: Option<String>,
    comment: Option<String>,
    #[serde(rename = "closeTimestamp")]
    close_timestamp: Option<u64>,
}

/// One historical order. `volume`/`executedVolume` are cents → lots via
/// `volume_to_units` (NOT moneyDigits-scaled — `ProtoOAOrder` carries no
/// moneyDigits). Prices pass through raw; all timestamps are ms.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CTraderHistoricalOrderSnapshot {
    pub order_id: i64,
    pub position_id: Option<i64>,
    pub symbol_id: i64,
    pub side: String,
    pub order_type: String,
    pub order_status: String,
    pub volume_lots: f64,
    pub executed_volume_lots: Option<f64>,
    pub execution_price: Option<f64>,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub time_in_force: Option<String>,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub client_order_id: Option<String>,
    pub closing_order: bool,
    pub is_stop_out: bool,
    pub trailing_stop_loss: bool,
    pub open_timestamp_ms: Option<i64>,
    pub close_timestamp_ms: Option<i64>,
    pub expiration_timestamp_ms: Option<i64>,
    pub utc_last_update_timestamp_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CTraderOrderHistoryBundle {
    pub account_id: i64,
    pub orders: Vec<CTraderHistoricalOrderSnapshot>,
    pub has_more: bool,
}

fn historical_order_payload_to_snapshot(
    p: HistoricalOrderPayload,
) -> CTraderHistoricalOrderSnapshot {
    CTraderHistoricalOrderSnapshot {
        order_id: p.order_id,
        position_id: p.position_id,
        symbol_id: p.trade_data.symbol_id,
        side: trade_side_label(p.trade_data.trade_side),
        order_type: order_type_label(p.order_type),
        order_status: order_status_label(p.order_status),
        volume_lots: volume_to_units(p.trade_data.volume),
        executed_volume_lots: p.executed_volume.map(volume_to_units),
        execution_price: p.execution_price,
        limit_price: p.limit_price,
        stop_price: p.stop_price,
        stop_loss: p.stop_loss,
        take_profit: p.take_profit,
        time_in_force: p.time_in_force.map(time_in_force_label),
        label: p.trade_data.label,
        comment: p.trade_data.comment,
        client_order_id: p.client_order_id,
        closing_order: p.closing_order.unwrap_or(false),
        is_stop_out: p.is_stop_out.unwrap_or(false),
        trailing_stop_loss: p.trailing_stop_loss.unwrap_or(false),
        open_timestamp_ms: p.trade_data.open_timestamp,
        close_timestamp_ms: p.trade_data.close_timestamp.map(|t| t as i64),
        expiration_timestamp_ms: p.expiration_timestamp,
        utc_last_update_timestamp_ms: p.utc_last_update_timestamp,
    }
}

pub fn parse_order_list_response(response_json: &str) -> Result<CTraderOrderHistoryBundle> {
    let envelope: OrderListEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader order list response")?;
    if envelope.payload_type != CTRADER_OA_ORDER_LIST_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader order list payload type: {}",
            envelope.payload_type
        ));
    }
    Ok(CTraderOrderHistoryBundle {
        account_id: envelope.payload.ctid_trader_account_id,
        has_more: envelope.payload.has_more,
        orders: envelope
            .payload
            .order
            .into_iter()
            .map(historical_order_payload_to_snapshot)
            .collect(),
    })
}

// ─── ProtoOACashFlowHistoryListRes (2144) — deposits / withdrawals / fees ──

#[derive(Debug, Deserialize)]
struct CashFlowListEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: CashFlowListPayload,
}
#[derive(Debug, Deserialize)]
struct CashFlowListPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    #[serde(rename = "depositWithdraw", default)]
    deposit_withdraw: Vec<DepositWithdrawPayload>,
}
#[derive(Debug, Deserialize)]
struct DepositWithdrawPayload {
    #[serde(rename = "operationType")]
    operation_type: i32,
    #[serde(rename = "balanceHistoryId")]
    balance_history_id: i64,
    balance: i64,
    delta: i64,
    #[serde(rename = "changeBalanceTimestamp")]
    change_balance_timestamp: i64,
    #[serde(rename = "externalNote")]
    external_note: Option<String>,
    #[serde(rename = "balanceVersion")]
    balance_version: Option<i64>,
    equity: Option<i64>,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
}

/// One cash-flow record. `balance`/`delta`/`equity` ARE moneyDigits-scaled
/// (per-item field) — use `scaled_money`, never a hardcoded /100. The signed
/// `delta` is the authoritative deposit(+)/withdraw(-) direction; the raw
/// `operation_type_code` is preserved for downstream classification.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CTraderCashFlowSnapshot {
    pub balance_history_id: i64,
    pub operation_type: String,
    pub operation_type_code: i32,
    pub balance: f64,
    pub delta: f64,
    pub equity: Option<f64>,
    pub change_balance_timestamp_ms: i64,
    pub external_note: Option<String>,
    pub balance_version: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CTraderCashFlowBundle {
    pub account_id: i64,
    pub entries: Vec<CTraderCashFlowSnapshot>,
}

fn cash_flow_payload_to_snapshot(p: DepositWithdrawPayload) -> Result<CTraderCashFlowSnapshot> {
    let digits = required_money_digits(p.money_digits, "cashFlow.moneyDigits")?;
    Ok(CTraderCashFlowSnapshot {
        balance_history_id: p.balance_history_id,
        operation_type: change_balance_type_label(p.operation_type),
        operation_type_code: p.operation_type,
        balance: scaled_money(p.balance, digits)?,
        delta: scaled_money(p.delta, digits)?,
        equity: p
            .equity
            .map(|value| scaled_money(value, digits))
            .transpose()?,
        change_balance_timestamp_ms: p.change_balance_timestamp,
        external_note: p.external_note,
        balance_version: p.balance_version,
    })
}

pub fn parse_cash_flow_history_response(response_json: &str) -> Result<CTraderCashFlowBundle> {
    let envelope: CashFlowListEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader cash flow history response")?;
    if envelope.payload_type != CTRADER_OA_CASH_FLOW_HISTORY_LIST_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader cash flow payload type: {}",
            envelope.payload_type
        ));
    }
    Ok(CTraderCashFlowBundle {
        account_id: envelope.payload.ctid_trader_account_id,
        entries: envelope
            .payload
            .deposit_withdraw
            .into_iter()
            .map(cash_flow_payload_to_snapshot)
            .collect::<Result<Vec<_>>>()?,
    })
}

// ─── ProtoOAExpectedMarginRes (2140) — pre-trade margin ────────────────────

#[derive(Debug, Deserialize)]
struct ExpectedMarginEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: ExpectedMarginPayload,
}
#[derive(Debug, Deserialize)]
struct ExpectedMarginPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    #[serde(default)]
    margin: Vec<ExpectedMarginEntryPayload>,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
}
#[derive(Debug, Deserialize)]
struct ExpectedMarginEntryPayload {
    volume: i64,
    #[serde(rename = "buyMargin")]
    buy_margin: i64,
    #[serde(rename = "sellMargin")]
    sell_margin: i64,
}

/// One (volume → buy/sell margin) entry. `moneyDigits` is on the PARENT Res,
/// scaling buy/sellMargin; `volume` is cents → lots.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CTraderExpectedMarginEntry {
    pub volume_lots: f64,
    pub buy_margin: f64,
    pub sell_margin: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CTraderExpectedMarginBundle {
    pub account_id: i64,
    pub entries: Vec<CTraderExpectedMarginEntry>,
}

pub fn parse_expected_margin_response(response_json: &str) -> Result<CTraderExpectedMarginBundle> {
    let envelope: ExpectedMarginEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader expected margin response")?;
    if envelope.payload_type != CTRADER_OA_EXPECTED_MARGIN_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader expected margin payload type: {}",
            envelope.payload_type
        ));
    }
    let digits =
        required_money_digits(envelope.payload.money_digits, "expectedMargin.moneyDigits")?;
    Ok(CTraderExpectedMarginBundle {
        account_id: envelope.payload.ctid_trader_account_id,
        entries: envelope
            .payload
            .margin
            .into_iter()
            .map(|m| -> Result<CTraderExpectedMarginEntry> {
                Ok(CTraderExpectedMarginEntry {
                    volume_lots: volume_to_units(m.volume),
                    buy_margin: scaled_money(m.buy_margin, digits)?,
                    sell_margin: scaled_money(m.sell_margin, digits)?,
                })
            })
            .collect::<Result<Vec<_>>>()?,
    })
}

// ─── ProtoOAGetCtidProfileByTokenRes (2152) — who is logged in ─────────────
// Verified proto: ProtoOACtidProfile carries EXACTLY one field, userId.
// (The builder docstring's "nickname" is wrong; do NOT add a nickname field.)

#[derive(Debug, Deserialize)]
struct CtidProfileEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: CtidProfilePayload,
}
#[derive(Debug, Deserialize)]
struct CtidProfilePayload {
    profile: CtidProfileInner,
}
#[derive(Debug, Deserialize)]
struct CtidProfileInner {
    #[serde(rename = "userId")]
    user_id: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CTraderCtidProfileSnapshot {
    pub user_id: i64,
}

pub fn parse_ctid_profile_response(response_json: &str) -> Result<CTraderCtidProfileSnapshot> {
    let envelope: CtidProfileEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader cTID profile response")?;
    if envelope.payload_type != CTRADER_OA_GET_CTID_PROFILE_BY_TOKEN_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader cTID profile payload type: {}",
            envelope.payload_type
        ));
    }
    Ok(CTraderCtidProfileSnapshot {
        user_id: envelope.payload.profile.user_id,
    })
}

// ─── ProtoOAVersionRes (2105) — broker Open API version ────────────────────

#[derive(Debug, Deserialize)]
struct VersionEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: VersionPayload,
}
#[derive(Debug, Deserialize)]
struct VersionPayload {
    version: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CTraderServerVersionSnapshot {
    pub version: String,
}

pub fn parse_version_response(response_json: &str) -> Result<CTraderServerVersionSnapshot> {
    let envelope: VersionEnvelope =
        serde_json::from_str(response_json).context("failed to parse cTrader version response")?;
    if envelope.payload_type != CTRADER_OA_VERSION_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader version payload type: {}",
            envelope.payload_type
        ));
    }
    Ok(CTraderServerVersionSnapshot {
        version: envelope.payload.version,
    })
}

pub(crate) fn ensure_success_payload_type(
    response_json: &str,
    expected_payload_type: u32,
) -> Result<()> {
    let envelope = parse_open_api_envelope(response_json)?;
    if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "cTrader response failed: {}",
            parse_ctrader_error_payload(&envelope.payload)?
        ));
    }
    if envelope.payload_type != expected_payload_type {
        return Err(anyhow!(
            "unexpected cTrader payload type: expected {}, got {}",
            expected_payload_type,
            envelope.payload_type
        ));
    }
    Ok(())
}

/// Apply the exact broker-side decimal precision to a raw money value.
fn scaled_money(value: i64, digits: u32) -> Result<f64> {
    crate::app_services::ctrader_money::scale_ctrader_money_int(value, digits as i32)
}

fn scaled_unsigned_money(value: u64, digits: u32) -> Result<f64> {
    crate::app_services::ctrader_money::scale_ctrader_money_uint(value, digits as i32)
}

fn required_money_digits(value: Option<u32>, field: &str) -> Result<u32> {
    crate::app_services::ctrader_money::required_money_digits(value, field)
}

fn volume_to_units(value: i64) -> f64 {
    value as f64 / 100.0
}

fn account_type_label(value: i32) -> String {
    match value {
        0 => "HEDGED",
        1 => "NETTED",
        2 => "SPREAD_BETTING",
        other => return format!("UNKNOWN({other})"),
    }
    .to_string()
}

fn trade_side_label(value: i32) -> String {
    match value {
        1 => "BUY",
        2 => "SELL",
        other => return format!("UNKNOWN({other})"),
    }
    .to_string()
}

fn order_type_label(value: i32) -> String {
    match value {
        1 => "MARKET",
        2 => "LIMIT",
        3 => "STOP",
        4 => "STOP_LOSS_TAKE_PROFIT",
        5 => "MARKET_RANGE",
        6 => "STOP_LIMIT",
        other => return format!("UNKNOWN({other})"),
    }
    .to_string()
}

fn deal_status_label(value: i32) -> String {
    match value {
        2 => "FILLED",
        3 => "PARTIALLY_FILLED",
        4 => "REJECTED",
        5 => "INTERNALLY_REJECTED",
        6 => "ERROR",
        7 => "MISSED",
        other => return format!("UNKNOWN({other})"),
    }
    .to_string()
}

/// `ProtoOAOrderStatus` enum (2026-06-10). Order-history audit view.
fn order_status_label(value: i32) -> String {
    match value {
        1 => "ACCEPTED",
        2 => "FILLED",
        3 => "REJECTED",
        4 => "EXPIRED",
        5 => "CANCELLED",
        other => return format!("UNKNOWN({other})"),
    }
    .to_string()
}

/// `ProtoOATimeInForce` enum (2026-06-10).
fn time_in_force_label(value: i32) -> String {
    match value {
        1 => "GOOD_TILL_DATE",
        2 => "GOOD_TILL_CANCEL",
        3 => "IMMEDIATE_OR_CANCEL",
        4 => "FILL_OR_KILL",
        5 => "MARKET_ON_OPEN",
        other => return format!("UNKNOWN({other})"),
    }
    .to_string()
}

/// `ProtoOAChangeBalanceType` (cash-flow operation) — COSMETIC label only.
///
/// The enum has ~35 members; their exact numeric mapping was NOT fully
/// verified against the proto in this pass, and mislabelling a withdrawal as
/// a deposit would corrupt the operator's money view. So we deliberately map
/// ONLY the two values we are certain of (0 = deposit, 1 = withdraw) plus the
/// known 39, and return `CHANGE_BALANCE_TYPE(n)` for the rest — the SIGNED
/// `delta` is the authoritative direction for any money math, and the raw
/// `operation_type_code` is preserved on the snapshot for later precise
/// labelling once the full enum is transcribed from the proto.
fn change_balance_type_label(value: i32) -> String {
    match value {
        0 => "BALANCE_DEPOSIT",
        1 => "BALANCE_WITHDRAW",
        39 => "BALANCE_DEPOSIT_NEGATIVE_BALANCE_PROTECTION",
        other => return format!("CHANGE_BALANCE_TYPE({other})"),
    }
    .to_string()
}

fn current_unix_millis() -> Result<i64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| anyhow!("system clock is before unix epoch"))?
        .as_millis() as i64)
}

#[cfg(test)]
#[path = "ctrader_account_tests.rs"]
mod tests;
