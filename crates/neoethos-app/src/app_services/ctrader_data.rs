use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_QUOTE_TYPE_ASK, CTRADER_QUOTE_TYPE_BID, CTraderOpenApiJsonMessage,
    CTraderOpenApiTransport, ProductionCTraderOpenApiTransport, build_account_auth_request,
    build_application_auth_request, build_get_tick_data_request, build_get_trendbars_request,
    build_subscribe_live_trendbar_request, build_subscribe_spots_request,
    build_symbol_by_id_request, build_symbols_list_request,
    build_unsubscribe_live_trendbar_request, build_unsubscribe_spots_request,
    parse_ctrader_error_payload, parse_open_api_envelope, trendbar_period_value,
};
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderLightSymbolInfo {
    pub symbol_id: i64,
    pub symbol_name: String,
    pub enabled: bool,
    pub description: Option<String>,
    /// **Phase D.1b (2026-05-28)** — broker's classification of the
    /// symbol into a category (e.g. "FX Majors", "Spot Metals",
    /// "US Indices"). Joins to `ProtoOASymbolCategory.id` which in
    /// turn references `ProtoOAAssetClass.id`. We use this chain to
    /// filter the bootstrap catalog to forex-relevant classes only,
    /// dropping the 700+ equity symbols a forex-ai will never
    /// trade. `None` when the broker omits the field (some exotic
    /// instruments).
    pub symbol_category_id: Option<i64>,
    /// **Phase D.2a (2026-05-28)** — base asset id (e.g. 4 = EUR).
    /// Joins to `ProtoOAAsset.assetId` from the asset list. Used by
    /// `SymbolMetadataTable::load_from_broker_catalog` to populate
    /// `SymbolMetadata.base` without name-pattern hacks on
    /// symbolName. `None` for symbols where broker omits the field
    /// (rare).
    pub base_asset_id: Option<i64>,
    /// **Phase D.2a (2026-05-28)** — quote asset id (e.g. 8 = USD).
    /// Joins to `ProtoOAAsset.assetId`. Used by the loader to
    /// populate `SymbolMetadata.quote`.
    pub quote_asset_id: Option<i64>,
}

/// **Phase D.2a (2026-05-28)** — broker's per-asset record. Mirrors
/// `ProtoOAAsset`. `name` is the canonical 3-letter currency code
/// for FX (`"EUR"`, `"USD"`), metal code (`"XAU"`), or commodity
/// unit (`"Oz"` doesn't exist as an asset; gold quotes use USD
/// quote_asset_id). `digits` is the precision of the asset itself
/// (independent of any symbol that uses it).
#[derive(Debug, Clone, PartialEq)]
pub struct CTraderAssetInfo {
    pub asset_id: i64,
    pub name: String,
    pub display_name: Option<String>,
    pub digits: Option<i32>,
}

/// **Phase D.1b (2026-05-28)** — top-level asset class metadata.
/// Mirrors `ProtoOAAssetClass`. Names are broker-defined strings
/// like "Forex", "Metals", "Indices", "Commodities", "Stocks",
/// "Cryptocurrencies", "ETFs". The Phase D bootstrap filters by
/// these names case-insensitively.
#[derive(Debug, Clone, PartialEq)]
pub struct CTraderAssetClassInfo {
    pub id: i64,
    pub name: String,
    pub sorting_number: Option<f64>,
}

/// **Phase D.1b (2026-05-28)** — symbol category metadata. Mirrors
/// `ProtoOASymbolCategory`. Each category points to a parent
/// `asset_class_id` so the bootstrap can keep symbols whose
/// category belongs to a desired asset class.
#[derive(Debug, Clone, PartialEq)]
pub struct CTraderSymbolCategoryInfo {
    pub id: i64,
    pub asset_class_id: i64,
    pub name: String,
    pub sorting_number: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderSymbolsListResult {
    pub account_id: i64,
    pub symbols: Vec<CTraderLightSymbolInfo>,
    pub archived_symbols: Vec<String>,
}

// ─── Cycle-3 Phase A — full ProtoOASymbol projection ────────────────────────
//
// Reference: proto/OpenApiModelMessages.proto:113-155 (ProtoOASymbol).
// All proto field numbers, units, and semantics quoted verbatim from
// the official cTrader Open API spec at
// https://help.ctrader.com/open-api/model-messages/#protooasymbol.
//
// Until this commit the codebase parsed 5 of the 40 fields the broker
// exposes, ignoring per-symbol commission, swap, SL/TP min-distance,
// trading-mode gating, and conversion-fee — making every backtest cost
// model systematically wrong and surfacing broker rejections at trade
// time instead of at the pre-trade gate. This module now captures the
// full ProtoOASymbol so downstream consumers (cost model in
// crates/neoethos-search, pre-trade gate in trading/risk_gate, AI Dock
// rationale in the new UI) can read the same numbers the broker uses
// to settle real PnL.

/// `ProtoOATradingMode` — the symbol's overall tradeability mode.
///
/// Proto file: `OpenApiModelMessages.proto:222-227`. **2026-05-28
/// real-data correction**: the cTrader JSON proxy sends the enum as
/// the **protobuf integer discriminant** (e.g. `"tradingMode": 0`),
/// NOT a SCREAMING_SNAKE_CASE string. We discovered this by capturing
/// real `ProtoOASymbolByIdRes` payloads via `--capture-symbols` and
/// finding the string-tagged enum failed to deserialize every
/// response. The explicit discriminants below match the proto's
/// `enum ProtoOATradingMode` declared values exactly.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq,
    serde_repr::Deserialize_repr, serde_repr::Serialize_repr,
)]
#[repr(u8)]
pub enum TradingModeProto {
    /// Default. Both opening and closing are allowed.
    Enabled = 0,
    /// New market orders only; no pending orders.
    DisabledWithoutPendingsExecution = 1,
    /// Pending orders allowed; market orders blocked.
    DisabledWithPendingsExecution = 2,
    /// Only closing existing positions is allowed.
    CloseOnlyMode = 3,
}

/// `ProtoOACommissionType` — how to interpret the
/// `precise_trading_commission_rate` value.
///
/// Proto file: `OpenApiModelMessages.proto:202-207`. The proto
/// comment specifies units per variant:
/// - `UsdPerMillionUsd`: USD per million USD of notional volume
///   (typical FX, e.g. $50 per $1M).
/// - `UsdPerLot`: USD per 1 lot (CFDs, commodities, indices).
/// - `PercentageOfValue`: percentage of notional volume × 100,000
///   (i.e. value 5 means 0.005%); used for equities.
/// - `QuoteCcyPerLot`: quote-currency per 1 lot (CFDs in non-USD).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq,
    serde_repr::Deserialize_repr, serde_repr::Serialize_repr,
)]
#[repr(u8)]
pub enum CommissionType {
    UsdPerMillionUsd = 1,
    UsdPerLot = 2,
    PercentageOfValue = 3,
    QuoteCcyPerLot = 4,
}

/// `ProtoOAMinCommissionType` — whether the broker's minimum
/// commission is denominated in the deposit currency (`Currency`) or
/// the symbol's quote currency (`QuoteCurrency`).
///
/// Proto file: `OpenApiModelMessages.proto:216-219`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq,
    serde_repr::Deserialize_repr, serde_repr::Serialize_repr,
)]
#[repr(u8)]
pub enum MinCommissionType {
    Currency = 1,
    QuoteCurrency = 2,
}

/// `ProtoOASwapCalculationType` — units of the `swap_long`/`swap_short`
/// double fields. Each variant changes how a daily-swap charge is
/// computed:
/// - `Pips`: the raw value is in pips (most common for FX).
/// - `Percentage`: annual percentage (divide by 365 for daily).
/// - `Points`: the raw value is in points (= 10^-digits).
///
/// Proto file: `OpenApiModelMessages.proto:230-234`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq,
    serde_repr::Deserialize_repr, serde_repr::Serialize_repr,
)]
#[repr(u8)]
pub enum SwapCalculationType {
    Pips = 0,
    Percentage = 1,
    Points = 2,
}

/// `ProtoOASymbolDistanceType` — units for the min SL/TP/GSL distance
/// fields.
///
/// Proto file: `OpenApiModelMessages.proto:210-213`. cTrader's JSON
/// proxy uses the `SYMBOL_DISTANCE_IN_*` prefix.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq,
    serde_repr::Deserialize_repr, serde_repr::Serialize_repr,
)]
#[repr(u8)]
pub enum SymbolDistanceType {
    SymbolDistanceInPoints = 1,
    SymbolDistanceInPercentage = 2,
}

/// `ProtoOADayOfWeek` — used by `swap_rollover_3_days` to mark the
/// triple-swap weekday, and by `rollover_commission_3_days` for the
/// Shariah triple-rollover weekday.
///
/// Proto file: `OpenApiModelMessages.proto:184-193`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq,
    serde_repr::Deserialize_repr, serde_repr::Serialize_repr,
)]
#[repr(u8)]
pub enum DayOfWeek {
    None = 0,
    Monday = 1,
    Tuesday = 2,
    Wednesday = 3,
    Thursday = 4,
    Friday = 5,
    Saturday = 6,
    Sunday = 7,
}

/// One weekly-trading interval — start and end seconds counted from
/// Sunday 00:00 in the symbol's `schedule_time_zone`. Mirrors
/// `ProtoOAInterval` (`OpenApiModelMessages.proto:196-199`).
///
/// `end_second` is exclusive, `start_second` is inclusive. So a full
/// trading day might be `(0, 86400)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TradingInterval {
    pub start_second_from_sunday: u32,
    pub end_second_from_sunday: u32,
}

/// One holiday window. Mirrors `ProtoOAHoliday`
/// (`OpenApiModelMessages.proto:693-701`).
#[derive(Debug, Clone, PartialEq)]
pub struct HolidayWindow {
    pub holiday_id: i64,
    pub name: String,
    pub description: Option<String>,
    pub schedule_time_zone: String,
    /// Days since 1 Jan 1970. Multiply by 86_400_000 for Unix ms.
    pub days_since_epoch: i64,
    pub is_recurring: bool,
    /// Optional intra-day offsets (e.g. closes at noon).
    pub start_second_from_midnight: Option<i32>,
    pub end_second_from_midnight: Option<i32>,
}

/// Full financial-fields projection of `ProtoOASymbol` — everything
/// `CTraderSymbolInfo` doesn't already cover. Held as
/// `Option<SymbolFinancials>` on the parent struct so legacy paths
/// (light symbol list, cached symbol-by-name lookup) can keep working
/// even when these fields haven't been fetched yet.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolFinancials {
    // ── Commission (D.1 from batch-2 audit) ──────────────────────
    /// How to interpret `precise_trading_commission_rate`.
    pub commission_type: Option<CommissionType>,
    /// Commission base amount. For non-percentage types this is the
    /// raw rate × 10^8 (e.g. $50/million USD is stored as 50e8).
    /// For `PercentageOfValue` the multiplier is × 10^5 instead
    /// (per the proto comment on field 31).
    pub precise_trading_commission_rate: Option<i64>,
    /// Minimum commission per trade, ALWAYS × 10^8 (proto field 32).
    pub precise_min_commission: Option<i64>,
    /// Whether `precise_min_commission` is in the deposit currency
    /// (`Currency`) or the symbol's quote currency (`QuoteCurrency`).
    pub min_commission_type: Option<MinCommissionType>,
    /// Asset code for the min commission (default "USD").
    pub min_commission_asset: Option<String>,
    /// Per-trade conversion fee when symbol quote ≠ deposit currency.
    /// Stored as `1 = 0.01%`. A typical broker value is 50-100
    /// (= 0.5-1% silent profit cut on every closed trade).
    pub pnl_conversion_fee_rate: Option<i32>,

    // ── Swap (D.2 from batch-2 audit) ────────────────────────────
    /// SWAP charge for long positions, in units determined by
    /// `swap_calculation_type` (PIPS / PERCENTAGE / POINTS).
    pub swap_long: Option<f64>,
    pub swap_short: Option<f64>,
    pub swap_calculation_type: Option<SwapCalculationType>,
    /// Hours between swap charges. 24 = once per day (the common
    /// case); 12 = twice per day; 8 = thrice per day.
    pub swap_period_hours: Option<i32>,
    /// Minutes from 00:00 UTC when the first intraday swap is
    /// charged. Defines the rollover moment within the day.
    pub swap_time_minutes_from_utc_midnight: Option<i32>,
    /// Triple-swap weekday. Most brokers: WEDNESDAY (the MT4
    /// convention covers the weekend rollover); some FX brokers
    /// use FRIDAY instead.
    pub swap_rollover_3_days: Option<DayOfWeek>,
    /// Initial period count to skip before the first swap charge.
    pub skip_swap_periods: Option<i32>,
    /// If TRUE, swap is charged for all 7 weekdays (rare; some
    /// Islamic accounts may turn this on with the rollover field
    /// instead).
    pub charge_swap_at_weekends: Option<bool>,

    // ── Rollover (Shariah / swap-free accounts) ──────────────────
    /// Admin fee charged INSTEAD of swap on Shariah-compliant
    /// accounts. Stored in the deposit currency, charged daily per
    /// open lot.
    pub rollover_commission: Option<i64>,
    pub rollover_commission_3_days: Option<DayOfWeek>,
    pub skip_rollover_days: Option<i32>,

    // ── Distance constraints (D.3 from batch-2 audit) ────────────
    /// Minimum allowed distance between Stop Loss and current price.
    /// Units determined by `distance_set_in`. Sending an SL closer
    /// than this gets the order rejected with TRADING_BAD_STOPS.
    pub sl_distance_points: Option<u32>,
    pub tp_distance_points: Option<u32>,
    /// Same but for Guaranteed Stop Loss (limited-risk accounts
    /// only).
    pub gsl_distance_points: Option<u32>,
    /// Guaranteed stop-loss fee. Units not documented in the proto
    /// comment — assume same scaling as `precise_*` fields if you
    /// need to read it.
    pub gsl_charge: Option<i64>,
    pub distance_set_in: Option<SymbolDistanceType>,
    /// Per-symbol GSL availability flag (proto field 5 —
    /// `guaranteedStopLoss`).
    pub guaranteed_stop_loss_available: Option<bool>,

    // ── Trading-mode gating (D.4 from batch-2 audit) ─────────────
    /// Full proto enum — exposes more than the existing
    /// `is_trading_enabled` bool (e.g., CLOSE_ONLY_MODE).
    pub trading_mode: Option<TradingModeProto>,
    pub enable_short_selling: Option<bool>,

    // ── Schedule + holidays ──────────────────────────────────────
    pub schedule_time_zone: Option<String>,
    pub trading_intervals: Vec<TradingInterval>,
    pub holidays: Vec<HolidayWindow>,

    // ── Misc ─────────────────────────────────────────────────────
    pub max_exposure: Option<u64>,
    pub leverage_id: Option<i64>,
    pub measurement_units: Option<String>,
}

impl SymbolFinancials {
    /// Build the per-lot commission in the asset implied by
    /// `commission_type`, for a given notional USD volume of the trade
    /// (used by USD_PER_MILLION_USD) or lot count (for the other
    /// variants). Returns None if commission_type or rate is missing.
    ///
    /// For non-percentage variants the proto stores the rate × 10^8;
    /// for `PercentageOfValue` the multiplier is × 10^5. We undo the
    /// scaling so the returned value is in plain decimal units
    /// (USD per million USD, USD per lot, percentage, or quote-ccy
    /// per lot — interpretation matches the variant).
    ///
    /// Caller is responsible for applying `precise_min_commission` and
    /// the `pnl_conversion_fee_rate` separately — those are not
    /// folded in here so the breakdown stays auditable.
    pub fn commission_rate_decimal(&self) -> Option<f64> {
        let rate = self.precise_trading_commission_rate?;
        let kind = self.commission_type?;
        let divisor: f64 = match kind {
            CommissionType::PercentageOfValue => 1.0e5,
            _ => 1.0e8,
        };
        Some(rate as f64 / divisor)
    }

    /// Daily swap charge in the units implied by
    /// `swap_calculation_type`. The caller's cost model converts this
    /// to a notional charge using the symbol's pip_size and lot
    /// notional. Returns None if `swap_long` is missing.
    ///
    /// **2026-05-28 real-data correction**: for BTC/USD on the
    /// captured Demo account the broker sends `swap_long: 0.0,
    /// swap_short: 0.0` but **OMITS** `swap_calculation_type`. The
    /// previous implementation `let _kind = self.swap_calculation_type?`
    /// silently returned `None` in that case, so the backtest saw
    /// "unknown swap" instead of "zero swap" — same fail-loud
    /// regression we just fixed for the enum deserializer. Now we
    /// treat a missing calc_type as PIPS (the proto default per
    /// `ProtoOASwapCalculationType` declaration), matching the cTrader
    /// proxy's omitted-means-default convention. The fixture
    /// `ctrader_symbol_BTCUSD.raw.json` exercises this path.
    pub fn daily_swap_long(&self) -> Option<f64> {
        let raw = self.swap_long?;
        // Default to PIPS per proto when broker omits the field.
        let _kind = self
            .swap_calculation_type
            .unwrap_or(SwapCalculationType::Pips);
        Some(raw)
    }
    pub fn daily_swap_short(&self) -> Option<f64> {
        let raw = self.swap_short?;
        let _kind = self
            .swap_calculation_type
            .unwrap_or(SwapCalculationType::Pips);
        Some(raw)
    }

    /// True if the symbol allows opening NEW positions right now.
    /// `CloseOnlyMode` and the two `Disabled*` variants return false.
    pub fn can_open_new_position(&self) -> bool {
        matches!(self.trading_mode, Some(TradingModeProto::Enabled))
    }

    /// True if SHORT (sell-to-open) is permitted on this symbol.
    /// Defaults to true when the broker omitted the field — matches
    /// cTrader's default behavior.
    pub fn short_selling_allowed(&self) -> bool {
        self.enable_short_selling.unwrap_or(true)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderSymbolInfo {
    pub symbol_id: i64,
    pub symbol_name: String,
    pub display_name: String,
    pub digits: i32,
    pub pip_position: i32,
    pub is_archived: bool,
    pub is_trading_enabled: bool,
    pub min_volume: Option<i64>,
    pub max_volume: Option<i64>,
    pub step_volume: Option<i64>,
    pub lot_size: Option<i64>,
    /// **2026-05-27**: kept on the parent struct for backwards-compat
    /// with paths that read it directly. New code should prefer
    /// `financials.pnl_conversion_fee_rate` so the units and the
    /// "missing when symbol wasn't fetched in full" case are obvious.
    pub pnl_conversion_fee_rate: Option<i32>,
    /// Full ProtoOASymbol projection. `None` for light-symbol-list
    /// entries (where the broker hasn't sent us the financial
    /// fields); `Some(_)` after a successful
    /// `ProtoOASymbolByIdReq` fetch. See `SymbolFinancials` for the
    /// breakdown.
    pub financials: Option<SymbolFinancials>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderSymbolLookupRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
    pub account_id: String,
    pub symbol_name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderResolvedSymbol {
    pub account_id: i64,
    pub light_symbol: CTraderLightSymbolInfo,
    pub symbol: CTraderSymbolInfo,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistoricalBar {
    pub timestamp_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistoricalBarsResult {
    pub symbol_id: i64,
    pub timeframe: String,
    pub bars: Vec<HistoricalBar>,
    pub has_more: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistoricalTick {
    pub timestamp_ms: i64,
    pub price: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistoricalTicksResult {
    pub symbol_id: i64,
    pub ticks: Vec<HistoricalTick>,
    pub has_more: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderChartHistoryRequest {
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

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderChartHistoryResult {
    pub symbol: CTraderSymbolInfo,
    pub bars: Vec<HistoricalBar>,
    pub has_more: bool,
    pub bid_ticks: Vec<HistoricalTick>,
    pub ask_ticks: Vec<HistoricalTick>,
    pub live_subscription_plan: CTraderLiveSubscriptionPlan,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderHistoricalBarsFetchResult {
    pub symbol: CTraderSymbolInfo,
    pub bars: Vec<HistoricalBar>,
    pub has_more: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderLiveSubscriptionPlan {
    pub subscribe_spots: CTraderOpenApiJsonMessage,
    pub subscribe_trendbars: CTraderOpenApiJsonMessage,
    pub unsubscribe_spots: CTraderOpenApiJsonMessage,
    pub unsubscribe_trendbars: CTraderOpenApiJsonMessage,
}

#[derive(Debug, Deserialize)]
struct SymbolsListEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: SymbolsListPayload,
}

#[derive(Debug, Deserialize)]
struct SymbolsListPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    #[serde(default)]
    symbol: Vec<LightSymbolPayload>,
    #[serde(rename = "archivedSymbol", default)]
    archived_symbol: Vec<ArchivedSymbolPayload>,
}

#[derive(Debug, Deserialize)]
struct LightSymbolPayload {
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    #[serde(rename = "symbolName")]
    symbol_name: Option<String>,
    enabled: Option<bool>,
    description: Option<String>,
    // **Phase D.1b (2026-05-28)** — broker's category classifier;
    // joins to ProtoOASymbolCategory.id. Proto field 6 on
    // ProtoOALightSymbol; previously unused by our parser.
    #[serde(rename = "symbolCategoryId")]
    symbol_category_id: Option<i64>,
    // **Phase D.2a (2026-05-28)** — proto fields 4 & 5; joins to
    // ProtoOAAsset.assetId so the catalog loader can populate
    // `SymbolMetadata.base`/`quote` strings from the broker's
    // own asset table.
    #[serde(rename = "baseAssetId")]
    base_asset_id: Option<i64>,
    #[serde(rename = "quoteAssetId")]
    quote_asset_id: Option<i64>,
}

// ─── Phase D.2a — asset list wire shapes ────────────────────────────
#[derive(Debug, Deserialize)]
struct AssetListEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: AssetListPayload,
}

#[derive(Debug, Deserialize)]
struct AssetListPayload {
    #[serde(default)]
    asset: Vec<AssetEntry>,
}

#[derive(Debug, Deserialize)]
struct AssetEntry {
    #[serde(rename = "assetId")]
    asset_id: i64,
    name: Option<String>,
    #[serde(rename = "displayName")]
    display_name: Option<String>,
    digits: Option<i32>,
}

// ─── Phase D.1b — asset class + symbol category wire shapes ─────────
//
// The catalog filter chain is:
//   ProtoOAAssetClassListRes  → [ {id, name, sortingNumber}, ... ]
//   ProtoOASymbolCategoryListRes → [ {id, assetClassId, name, sortingNumber}, ... ]
//   ProtoOASymbolsListRes         → [ {symbolId, symbolName, symbolCategoryId, ...}, ... ]
//
// At bootstrap time we resolve the user's asset-class allow-list
// ("Forex", "Metals", "Indices", "Commodities") against the broker's
// own naming, then keep only LightSymbols whose category's parent
// class is in the allow-list. No name-pattern heuristics — the
// broker is the source of truth for classification too.

#[derive(Debug, Deserialize)]
struct AssetClassListEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: AssetClassListPayload,
}

#[derive(Debug, Deserialize)]
struct AssetClassListPayload {
    #[serde(default, rename = "assetClass")]
    asset_class: Vec<AssetClassEntry>,
}

#[derive(Debug, Deserialize)]
struct AssetClassEntry {
    id: i64,
    name: Option<String>,
    #[serde(rename = "sortingNumber")]
    sorting_number: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SymbolCategoryListEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: SymbolCategoryListPayload,
}

#[derive(Debug, Deserialize)]
struct SymbolCategoryListPayload {
    #[serde(default, rename = "symbolCategory")]
    symbol_category: Vec<SymbolCategoryEntry>,
}

#[derive(Debug, Deserialize)]
struct SymbolCategoryEntry {
    id: i64,
    #[serde(rename = "assetClassId")]
    asset_class_id: i64,
    name: Option<String>,
    #[serde(rename = "sortingNumber")]
    sorting_number: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ArchivedSymbolPayload {
    name: String,
}

#[derive(Debug, Deserialize)]
struct SymbolByIdEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: SymbolByIdPayload,
}

#[derive(Debug, Deserialize)]
struct SymbolByIdPayload {
    #[serde(default)]
    symbol: Vec<FullSymbolPayload>,
}

/// Wire-shape for one element of `ProtoOASymbolByIdRes.symbol`. Field
/// names mirror the proto JSON encoding exactly (camelCase). Every
/// optional that the proto marks `optional` is wrapped in `Option<>`
/// so a broker that doesn't populate a field doesn't blow up
/// deserialization.
#[derive(Debug, Deserialize)]
struct FullSymbolPayload {
    // ── Identity / pricing / volume (already wired pre-Phase-A) ──
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    digits: i32,
    #[serde(rename = "pipPosition")]
    pip_position: i32,
    #[serde(rename = "minVolume")]
    min_volume: Option<i64>,
    #[serde(rename = "maxVolume")]
    max_volume: Option<i64>,
    #[serde(rename = "stepVolume")]
    step_volume: Option<i64>,
    #[serde(rename = "lotSize")]
    lot_size: Option<i64>,
    #[serde(rename = "pnlConversionFeeRate")]
    pnl_conversion_fee_rate: Option<i32>,
    #[serde(rename = "tradingMode")]
    trading_mode: Option<TradingModeProto>,

    // ── Phase A — commission (D.1) ───────────────────────────────
    #[serde(rename = "commissionType")]
    commission_type: Option<CommissionType>,
    #[serde(rename = "preciseTradingCommissionRate")]
    precise_trading_commission_rate: Option<i64>,
    #[serde(rename = "preciseMinCommission")]
    precise_min_commission: Option<i64>,
    #[serde(rename = "minCommissionType")]
    min_commission_type: Option<MinCommissionType>,
    #[serde(rename = "minCommissionAsset")]
    min_commission_asset: Option<String>,

    // ── Phase A — swap (D.2) ─────────────────────────────────────
    #[serde(rename = "swapLong")]
    swap_long: Option<f64>,
    #[serde(rename = "swapShort")]
    swap_short: Option<f64>,
    #[serde(rename = "swapCalculationType")]
    swap_calculation_type: Option<SwapCalculationType>,
    #[serde(rename = "swapPeriod")]
    swap_period: Option<i32>,
    #[serde(rename = "swapTime")]
    swap_time: Option<i32>,
    #[serde(rename = "swapRollover3Days")]
    swap_rollover_3_days: Option<DayOfWeek>,
    // Proto field name is `skipSWAPPeriods` (uppercase SWAP).
    #[serde(rename = "skipSWAPPeriods")]
    skip_swap_periods: Option<i32>,
    #[serde(rename = "chargeSwapAtWeekends")]
    charge_swap_at_weekends: Option<bool>,

    // ── Phase A — rollover (Shariah accounts) ────────────────────
    #[serde(rename = "rolloverCommission")]
    rollover_commission: Option<i64>,
    #[serde(rename = "rolloverCommission3Days")]
    rollover_commission_3_days: Option<DayOfWeek>,
    #[serde(rename = "skipRolloverDays")]
    skip_rollover_days: Option<i32>,

    // ── Phase A — distance constraints (D.3) ─────────────────────
    #[serde(rename = "slDistance")]
    sl_distance: Option<u32>,
    #[serde(rename = "tpDistance")]
    tp_distance: Option<u32>,
    #[serde(rename = "gslDistance")]
    gsl_distance: Option<u32>,
    #[serde(rename = "gslCharge")]
    gsl_charge: Option<i64>,
    #[serde(rename = "distanceSetIn")]
    distance_set_in: Option<SymbolDistanceType>,
    #[serde(rename = "guaranteedStopLoss")]
    guaranteed_stop_loss: Option<bool>,

    // ── Phase A — trading-mode gating (D.4) ──────────────────────
    #[serde(rename = "enableShortSelling")]
    enable_short_selling: Option<bool>,

    // ── Phase A — schedule + holidays ────────────────────────────
    #[serde(rename = "scheduleTimeZone")]
    schedule_time_zone: Option<String>,
    #[serde(default)]
    schedule: Vec<IntervalPayload>,
    #[serde(default)]
    holiday: Vec<HolidayPayload>,

    // ── Phase A — misc ───────────────────────────────────────────
    #[serde(rename = "maxExposure")]
    max_exposure: Option<u64>,
    #[serde(rename = "leverageId")]
    leverage_id: Option<i64>,
    #[serde(rename = "measurementUnits")]
    measurement_units: Option<String>,
}

#[derive(Debug, Deserialize)]
struct IntervalPayload {
    #[serde(rename = "startSecond")]
    start_second: u32,
    #[serde(rename = "endSecond")]
    end_second: u32,
}

#[derive(Debug, Deserialize)]
struct HolidayPayload {
    #[serde(rename = "holidayId")]
    holiday_id: i64,
    name: String,
    description: Option<String>,
    #[serde(rename = "scheduleTimeZone")]
    schedule_time_zone: String,
    #[serde(rename = "holidayDate")]
    holiday_date: i64,
    #[serde(rename = "isRecurring")]
    is_recurring: bool,
    #[serde(rename = "startSecond")]
    start_second: Option<i32>,
    #[serde(rename = "endSecond")]
    end_second: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct TrendbarsEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: TrendbarsPayload,
}

#[derive(Debug, Deserialize)]
struct TrendbarsPayload {
    period: Value,
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    #[serde(rename = "hasMore")]
    has_more: Option<bool>,
    #[serde(default)]
    trendbar: Vec<TrendbarPayload>,
}

#[derive(Debug, Deserialize)]
struct TrendbarPayload {
    volume: Option<i64>,
    low: i64,
    #[serde(rename = "deltaOpen")]
    delta_open: Option<u64>,
    #[serde(rename = "deltaClose")]
    delta_close: Option<u64>,
    #[serde(rename = "deltaHigh")]
    delta_high: Option<u64>,
    #[serde(rename = "utcTimestampInMinutes")]
    utc_timestamp_in_minutes: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TickDataEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: TickDataPayload,
}

#[derive(Debug, Deserialize)]
struct TickDataPayload {
    #[serde(rename = "symbolId")]
    symbol_id: Option<i64>,
    #[serde(rename = "hasMore")]
    has_more: bool,
    #[serde(rename = "tickData", default)]
    tick_data: Vec<TickPayload>,
}

#[derive(Debug, Deserialize)]
struct TickPayload {
    timestamp: i64,
    tick: i64,
}

pub fn parse_symbols_list_response(response_json: &str) -> Result<CTraderSymbolsListResult> {
    let envelope: SymbolsListEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader symbols list response")?;
    if envelope.payload_type != CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader symbols list payload type: {}",
            envelope.payload_type
        ));
    }

    Ok(CTraderSymbolsListResult {
        account_id: envelope.payload.ctid_trader_account_id,
        symbols: envelope
            .payload
            .symbol
            .into_iter()
            .map(|symbol| CTraderLightSymbolInfo {
                symbol_id: symbol.symbol_id,
                symbol_name: symbol.symbol_name.unwrap_or_default(),
                enabled: symbol.enabled.unwrap_or(false),
                description: symbol.description,
                symbol_category_id: symbol.symbol_category_id,
                base_asset_id: symbol.base_asset_id,
                quote_asset_id: symbol.quote_asset_id,
            })
            .collect(),
        archived_symbols: envelope
            .payload
            .archived_symbol
            .into_iter()
            .map(|symbol| symbol.name)
            .collect(),
    })
}

/// **Phase D.1b (2026-05-28)** — parse `ProtoOAAssetClassListRes`.
/// Returns the broker's top-level asset class table (Forex, Metals,
/// Indices, Commodities, Stocks, Cryptocurrencies, ETFs, ...).
/// Used by the catalog bootstrap to map a user-supplied allow-list
/// of class names to broker-defined IDs.
pub fn parse_asset_class_list_response(
    response_json: &str,
) -> Result<Vec<CTraderAssetClassInfo>> {
    let envelope: AssetClassListEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader asset class list response")?;
    if envelope.payload_type
        != crate::app_services::ctrader_messages::CTRADER_OA_ASSET_CLASS_LIST_RESPONSE_PAYLOAD_TYPE
    {
        return Err(anyhow!(
            "unexpected cTrader asset class list payload type: {}",
            envelope.payload_type
        ));
    }
    Ok(envelope
        .payload
        .asset_class
        .into_iter()
        .map(|e| CTraderAssetClassInfo {
            id: e.id,
            name: e.name.unwrap_or_default(),
            sorting_number: e.sorting_number,
        })
        .collect())
}

/// **Phase D.2a (2026-05-28)** — parse `ProtoOAAssetListRes`.
/// Returns the broker's per-asset registry, used to map
/// `LightSymbol.{base,quote}AssetId` to a 3-letter currency code.
pub fn parse_asset_list_response(response_json: &str) -> Result<Vec<CTraderAssetInfo>> {
    let envelope: AssetListEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader asset list response")?;
    if envelope.payload_type
        != crate::app_services::ctrader_messages::CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE
    {
        return Err(anyhow!(
            "unexpected cTrader asset list payload type: {}",
            envelope.payload_type
        ));
    }
    Ok(envelope
        .payload
        .asset
        .into_iter()
        .map(|e| CTraderAssetInfo {
            asset_id: e.asset_id,
            name: e.name.unwrap_or_default(),
            display_name: e.display_name,
            digits: e.digits,
        })
        .collect())
}

/// **Phase D.1b (2026-05-28)** — parse `ProtoOASymbolCategoryListRes`.
/// Returns the broker's symbol category table. Each row joins
/// `LightSymbol.symbol_category_id` to a parent `asset_class_id`,
/// completing the filter chain
/// `symbol → category → asset_class → allow-list`.
pub fn parse_symbol_category_list_response(
    response_json: &str,
) -> Result<Vec<CTraderSymbolCategoryInfo>> {
    let envelope: SymbolCategoryListEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader symbol category list response")?;
    if envelope.payload_type
        != crate::app_services::ctrader_messages::CTRADER_OA_SYMBOL_CATEGORY_RESPONSE_PAYLOAD_TYPE
    {
        return Err(anyhow!(
            "unexpected cTrader symbol category list payload type: {}",
            envelope.payload_type
        ));
    }
    Ok(envelope
        .payload
        .symbol_category
        .into_iter()
        .map(|e| CTraderSymbolCategoryInfo {
            id: e.id,
            asset_class_id: e.asset_class_id,
            name: e.name.unwrap_or_default(),
            sorting_number: e.sorting_number,
        })
        .collect())
}

pub fn parse_symbol_by_id_response(response_json: &str) -> Result<Vec<CTraderSymbolInfo>> {
    let envelope: SymbolByIdEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader full symbol response")?;
    if envelope.payload_type != CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader symbol-by-id payload type: {}",
            envelope.payload_type
        ));
    }

    Ok(envelope
        .payload
        .symbol
        .into_iter()
        .map(|symbol| {
            let financials = SymbolFinancials {
                commission_type: symbol.commission_type,
                precise_trading_commission_rate: symbol.precise_trading_commission_rate,
                precise_min_commission: symbol.precise_min_commission,
                min_commission_type: symbol.min_commission_type,
                min_commission_asset: symbol.min_commission_asset,
                pnl_conversion_fee_rate: symbol.pnl_conversion_fee_rate,
                swap_long: symbol.swap_long,
                swap_short: symbol.swap_short,
                swap_calculation_type: symbol.swap_calculation_type,
                swap_period_hours: symbol.swap_period,
                swap_time_minutes_from_utc_midnight: symbol.swap_time,
                swap_rollover_3_days: symbol.swap_rollover_3_days,
                skip_swap_periods: symbol.skip_swap_periods,
                charge_swap_at_weekends: symbol.charge_swap_at_weekends,
                rollover_commission: symbol.rollover_commission,
                rollover_commission_3_days: symbol.rollover_commission_3_days,
                skip_rollover_days: symbol.skip_rollover_days,
                sl_distance_points: symbol.sl_distance,
                tp_distance_points: symbol.tp_distance,
                gsl_distance_points: symbol.gsl_distance,
                gsl_charge: symbol.gsl_charge,
                distance_set_in: symbol.distance_set_in,
                guaranteed_stop_loss_available: symbol.guaranteed_stop_loss,
                trading_mode: symbol.trading_mode,
                enable_short_selling: symbol.enable_short_selling,
                schedule_time_zone: symbol.schedule_time_zone,
                trading_intervals: symbol
                    .schedule
                    .into_iter()
                    .map(|iv| TradingInterval {
                        start_second_from_sunday: iv.start_second,
                        end_second_from_sunday: iv.end_second,
                    })
                    .collect(),
                holidays: symbol
                    .holiday
                    .into_iter()
                    .map(|h| HolidayWindow {
                        holiday_id: h.holiday_id,
                        name: h.name,
                        description: h.description,
                        schedule_time_zone: h.schedule_time_zone,
                        days_since_epoch: h.holiday_date,
                        is_recurring: h.is_recurring,
                        start_second_from_midnight: h.start_second,
                        end_second_from_midnight: h.end_second,
                    })
                    .collect(),
                max_exposure: symbol.max_exposure,
                leverage_id: symbol.leverage_id,
                measurement_units: symbol.measurement_units,
            };
            CTraderSymbolInfo {
                symbol_id: symbol.symbol_id,
                symbol_name: String::new(),
                display_name: String::new(),
                digits: symbol.digits,
                pip_position: symbol.pip_position,
                is_archived: false,
                is_trading_enabled: matches!(
                    symbol.trading_mode,
                    Some(TradingModeProto::Enabled)
                ),
                min_volume: symbol.min_volume,
                max_volume: symbol.max_volume,
                step_volume: symbol.step_volume,
                lot_size: symbol.lot_size,
                pnl_conversion_fee_rate: symbol.pnl_conversion_fee_rate,
                financials: Some(financials),
            }
        })
        .collect())
}

pub fn parse_trendbars_response(
    response_json: &str,
    symbol: &CTraderSymbolInfo,
) -> Result<HistoricalBarsResult> {
    let envelope: TrendbarsEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader trendbars response")?;
    if envelope.payload_type != CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader trendbars payload type: {}",
            envelope.payload_type
        ));
    }

    let timeframe = trendbar_period_label(&envelope.payload.period)?;
    let bars = envelope
        .payload
        .trendbar
        .into_iter()
        .map(|trendbar| HistoricalBar {
            timestamp_ms: i64::from(trendbar.utc_timestamp_in_minutes.unwrap_or_default()) * 60_000,
            open: relative_price_to_absolute(
                trendbar.low + trendbar.delta_open.unwrap_or_default() as i64,
                symbol.digits,
            ),
            high: relative_price_to_absolute(
                trendbar.low + trendbar.delta_high.unwrap_or_default() as i64,
                symbol.digits,
            ),
            low: relative_price_to_absolute(trendbar.low, symbol.digits),
            close: relative_price_to_absolute(
                trendbar.low + trendbar.delta_close.unwrap_or_default() as i64,
                symbol.digits,
            ),
            volume: trendbar.volume,
        })
        .collect();

    Ok(HistoricalBarsResult {
        symbol_id: envelope.payload.symbol_id,
        timeframe,
        bars,
        has_more: envelope.payload.has_more.unwrap_or(false),
        warnings: Vec::new(),
    })
}

pub fn parse_tick_data_response(
    response_json: &str,
    symbol: &CTraderSymbolInfo,
) -> Result<HistoricalTicksResult> {
    let envelope: TickDataEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader tick data response")?;
    if envelope.payload_type != CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader tick-data payload type: {}",
            envelope.payload_type
        ));
    }

    let mut ticks = Vec::with_capacity(envelope.payload.tick_data.len());
    let mut previous_timestamp = None;
    for tick in envelope.payload.tick_data {
        let timestamp_ms = match previous_timestamp {
            None => tick.timestamp,
            Some(previous) => previous - tick.timestamp,
        };
        previous_timestamp = Some(timestamp_ms);
        ticks.push(HistoricalTick {
            timestamp_ms,
            price: relative_price_to_absolute(tick.tick, symbol.digits),
        });
    }
    ticks.sort_by_key(|tick| tick.timestamp_ms);

    Ok(HistoricalTicksResult {
        symbol_id: match envelope.payload.symbol_id {
            Some(symbol_id) if symbol_id != symbol.symbol_id => {
                return Err(anyhow!(
                    "unexpected cTrader tick-data symbol id: {}",
                    symbol_id
                ));
            }
            Some(symbol_id) => symbol_id,
            None => symbol.symbol_id,
        },
        ticks,
        has_more: envelope.payload.has_more,
    })
}

pub fn load_chart_history_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderChartHistoryRequest,
) -> Result<CTraderChartHistoryResult> {
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
    let account_id = resolved.account_id;
    let light_symbol = &resolved.light_symbol;
    let symbol = &resolved.symbol;

    let trendbar_period = trendbar_period_value(&request.timeframe)?;
    let live_subscription_plan = CTraderLiveSubscriptionPlan {
        subscribe_spots: build_subscribe_spots_request(
            account_id,
            &[light_symbol.symbol_id],
            true,
            "subscribe-spots-1",
        ),
        subscribe_trendbars: build_subscribe_live_trendbar_request(
            account_id,
            light_symbol.symbol_id,
            trendbar_period,
            "subscribe-live-trendbar-1",
        ),
        unsubscribe_spots: build_unsubscribe_spots_request(
            account_id,
            &[light_symbol.symbol_id],
            "unsubscribe-spots-1",
        ),
        unsubscribe_trendbars: build_unsubscribe_live_trendbar_request(
            account_id,
            light_symbol.symbol_id,
            trendbar_period,
            "unsubscribe-live-trendbar-1",
        ),
    };

    let detail_responses = transport.send_sequence(&[
        build_get_trendbars_request(
            account_id,
            light_symbol.symbol_id,
            trendbar_period,
            request.from_timestamp_ms,
            request.to_timestamp_ms,
            request.count,
            "trendbars-1",
        ),
        build_get_tick_data_request(
            account_id,
            light_symbol.symbol_id,
            CTRADER_QUOTE_TYPE_BID,
            request.from_timestamp_ms,
            request.to_timestamp_ms,
            "ticks-bid-1",
        ),
        build_get_tick_data_request(
            account_id,
            light_symbol.symbol_id,
            CTRADER_QUOTE_TYPE_ASK,
            request.from_timestamp_ms,
            request.to_timestamp_ms,
            "ticks-ask-1",
        ),
    ])?;

    if detail_responses.len() != 3 {
        return Err(anyhow!(
            "expected 3 cTrader detail responses, received {}",
            detail_responses.len()
        ));
    }
    let trendbars = parse_trendbars_response(&detail_responses[0], symbol)?;
    let bid_ticks = parse_tick_data_response(&detail_responses[1], symbol)?;
    let ask_ticks = parse_tick_data_response(&detail_responses[2], symbol)?;
    Ok(CTraderChartHistoryResult {
        symbol: resolved.symbol.clone(),
        bars: trendbars.bars,
        has_more: trendbars.has_more,
        bid_ticks: bid_ticks.ticks,
        ask_ticks: ask_ticks.ticks,
        live_subscription_plan,
    })
}

pub fn load_chart_history(
    request: &CTraderChartHistoryRequest,
) -> Result<CTraderChartHistoryResult> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    load_chart_history_with_transport(&transport, request)
}

pub fn load_historical_bars_only_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderChartHistoryRequest,
) -> Result<CTraderHistoricalBarsFetchResult> {
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
    let trendbar_period = trendbar_period_value(&request.timeframe)?;
    // `ProductionCTraderOpenApiTransport::send_sequence` opens a fresh
    // WSS connection per call and cTrader requires ProtoOAApplicationAuthReq
    // + ProtoOAAccountAuthReq on every new socket — otherwise the next
    // data-bearing request (trendbars here) comes back as ProtoOAErrorRes,
    // which parse_trendbars_response then chokes on with the unhelpful
    // "failed to parse cTrader trendbars response". Mirror the
    // re-auth pattern already used by `resolve_symbol_with_transport` for
    // the symbol-by-id call.
    let auth_responses = transport.send_sequence(&[
        build_application_auth_request(&request.client_id, &request.client_secret, "app-auth-1"),
        build_account_auth_request(resolved.account_id, &request.access_token, "account-auth-1"),
        build_get_trendbars_request(
            resolved.account_id,
            resolved.light_symbol.symbol_id,
            trendbar_period,
            request.from_timestamp_ms,
            request.to_timestamp_ms,
            request.count,
            "trendbars-1",
        ),
    ])?;

    if auth_responses.len() < 3 {
        // Same partial-response error walking we do in resolve_symbol —
        // send_sequence early-exits on ProtoOAErrorRes so the cTrader
        // error code lives in the last envelope we got back.
        for response in &auth_responses {
            let envelope = parse_open_api_envelope(response)?;
            if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                return Err(anyhow!(
                    "cTrader trendbars sequence failed (step {}): {}",
                    auth_responses.len(),
                    parse_ctrader_error_payload(&envelope.payload)?
                ));
            }
        }
        return Err(anyhow!(
            "expected 3 cTrader auth/trendbars responses, received {}",
            auth_responses.len()
        ));
    }

    let trendbars = parse_trendbars_response(&auth_responses[2], &resolved.symbol)?;
    Ok(CTraderHistoricalBarsFetchResult {
        symbol: resolved.symbol.clone(),
        bars: trendbars.bars,
        has_more: trendbars.has_more,
    })
}

#[allow(dead_code)]
pub fn load_historical_bars_only(
    request: &CTraderChartHistoryRequest,
) -> Result<CTraderHistoricalBarsFetchResult> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    load_historical_bars_only_with_transport(&transport, request)
}

pub fn resolve_symbol_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderSymbolLookupRequest,
) -> Result<CTraderResolvedSymbol> {
    let account_id = request
        .account_id
        .parse::<i64>()
        .context("cTrader account id must be numeric")?;

    let auth_responses = transport.send_sequence(&[
        build_application_auth_request(&request.client_id, &request.client_secret, "app-auth-1"),
        build_account_auth_request(account_id, &request.access_token, "account-auth-1"),
        build_symbols_list_request(account_id, false, "symbols-1"),
    ])?;

    if auth_responses.len() < 3 {
        // `send_sequence` early-exits the moment it sees an
        // `ERROR_RESPONSE` envelope (so the rest of the messages in
        // the batch never get sent). The partial set we got back
        // therefore carries the cTrader error code in its last
        // envelope. Walk the partial set and surface the first
        // error we find verbatim — otherwise the operator only ever
        // sees the un-actionable "received N" count and the actual
        // error code (e.g. `CH_ACCESS_TOKEN_INVALID`,
        // `ACCOUNT_NOT_AUTHORIZED`) stays trapped in the wire log.
        for response in &auth_responses {
            let envelope = parse_open_api_envelope(response)?;
            if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                return Err(anyhow!(
                    "cTrader auth/symbol failed (step {}): {}",
                    auth_responses.len(),
                    parse_ctrader_error_payload(&envelope.payload)?
                ));
            }
        }
        // Fall back to the count mismatch if none of the partial
        // responses was an error envelope (would mean the socket
        // closed cleanly mid-sequence, which is itself worth
        // surfacing).
        return Err(anyhow!(
            "expected 3 cTrader auth/symbol responses, received {}",
            auth_responses.len()
        ));
    }

    ensure_success_payload_type(
        &auth_responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(
        &auth_responses[1],
        CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;

    let symbols = parse_symbols_list_response(&auth_responses[2])?;
    let requested_key = normalize_symbol_key(&request.symbol_name);
    let light_symbol = symbols
        .symbols
        .into_iter()
        .find(|symbol| normalize_symbol_key(&symbol.symbol_name) == requested_key)
        .ok_or_else(|| {
            anyhow!(
                "cTrader symbol '{}' was not found for this account",
                request.symbol_name
            )
        })?;

    // v0.5.1.1 — `ProductionCTraderOpenApiTransport::send_sequence`
    // opens a fresh WSS connection on every call. cTrader Open API
    // requires `ProtoOAApplicationAuthReq` + `ProtoOAAccountAuthReq`
    // on every connection before any data-bearing request, otherwise
    // the next request comes back as `ProtoOAErrorRes` (payloadType
    // 2142). Re-authenticate at the head of this sequence so the
    // symbol-by-id call lands on an authenticated socket. Same fix
    // applies to the trendbars sequence in `ctrader_history.rs`.
    let detail_responses = transport.send_sequence(&[
        build_application_auth_request(&request.client_id, &request.client_secret, "app-auth-2"),
        build_account_auth_request(account_id, &request.access_token, "account-auth-2"),
        build_symbol_by_id_request(account_id, &[light_symbol.symbol_id], "symbol-by-id-1"),
    ])?;
    if detail_responses.len() < 3 {
        for response in &detail_responses {
            let envelope = parse_open_api_envelope(response)?;
            if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                return Err(anyhow!(
                    "cTrader symbol-by-id sequence failed (step {}): {}",
                    detail_responses.len(),
                    parse_ctrader_error_payload(&envelope.payload)?
                ));
            }
        }
        return Err(anyhow!(
            "expected 3 cTrader symbol-by-id auth/data responses, received {}",
            detail_responses.len()
        ));
    }
    ensure_success_payload_type(
        &detail_responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(
        &detail_responses[1],
        CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;

    let mut symbol = parse_symbol_by_id_response(&detail_responses[2])?
        .into_iter()
        .find(|symbol| symbol.symbol_id == light_symbol.symbol_id)
        .ok_or_else(|| {
            anyhow!(
                "cTrader full symbol metadata missing for symbol {}",
                light_symbol.symbol_id
            )
        })?;
    symbol.symbol_name = light_symbol.symbol_name.clone();
    symbol.display_name = light_symbol
        .description
        .clone()
        .filter(|description| !description.trim().is_empty())
        .unwrap_or_else(|| light_symbol.symbol_name.clone());

    Ok(CTraderResolvedSymbol {
        account_id,
        light_symbol,
        symbol,
    })
}

pub fn resolve_symbol(request: &CTraderSymbolLookupRequest) -> Result<CTraderResolvedSymbol> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    resolve_symbol_with_transport(&transport, request)
}

fn ensure_success_payload_type(response_json: &str, expected_payload_type: u32) -> Result<()> {
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

fn relative_price_to_absolute(relative: i64, digits: i32) -> f64 {
    round_to_digits(relative as f64 / 100000.0, digits)
}

fn round_to_digits(value: f64, digits: i32) -> f64 {
    let factor = 10_f64.powi(digits);
    (value * factor).round() / factor
}

fn trendbar_period_label(value: &Value) -> Result<String> {
    if let Some(label) = value.as_str() {
        return Ok(label.to_string());
    }
    let period = value
        .as_i64()
        .context("cTrader trendbar period is missing")?;
    // cTrader emits its own M2/M4/M10 codes (2/4/6), but those are
    // outside our canonical 12-timeframe set; we reject rather than
    // returning a label that downstream pipelines do not know how to
    // resample, train, or evaluate against.
    let label = match period {
        1 => "M1",
        3 => "M3",
        5 => "M5",
        7 => "M15",
        8 => "M30",
        9 => "H1",
        10 => "H4",
        11 => "H12",
        12 => "D1",
        13 => "W1",
        14 => "MN1",
        2 | 4 | 6 => {
            return Err(anyhow!(
                "cTrader trendbar period {} (M2/M4/M10) is outside the canonical timeframe set",
                period
            ));
        }
        other => return Err(anyhow!("unsupported cTrader trendbar period {}", other)),
    };
    Ok(label.to_string())
}

// **2026-05-27 Phase A**: previously the wire-shape held
// `trading_mode: Option<Value>` and this helper interrogated the JSON
// untyped. Now `FullSymbolPayload.trading_mode` is `Option<TradingModeProto>`
// (proper serde enum) and the parser uses `matches!(.., Some(Enabled))`
// directly. Helper retired to remove an inconsistent fallback path.

fn normalize_symbol_key(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_uppercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::ctrader_messages::CTraderOpenApiJsonMessage;

    struct StubTransport {
        sent: std::sync::Mutex<Vec<CTraderOpenApiJsonMessage>>,
        responses: std::sync::Mutex<Vec<anyhow::Result<String>>>,
    }

    impl StubTransport {
        fn with_responses(responses: Vec<anyhow::Result<String>>) -> Self {
            Self {
                sent: std::sync::Mutex::new(Vec::new()),
                responses: std::sync::Mutex::new(responses),
            }
        }

        fn sent_len(&self) -> usize {
            self.sent.lock().expect("sent lock").len()
        }
    }

    impl CTraderOpenApiTransport for StubTransport {
        fn send_sequence(
            &self,
            messages: &[CTraderOpenApiJsonMessage],
        ) -> anyhow::Result<Vec<String>> {
            self.sent
                .lock()
                .expect("sent lock")
                .extend(messages.iter().cloned());
            let mut responses = self.responses.lock().expect("responses lock");
            let mut output = Vec::with_capacity(messages.len());
            for _ in messages {
                output.push(responses.remove(0)?);
            }
            Ok(output)
        }
    }

    #[test]
    fn symbols_list_response_parses_lightweight_symbols() {
        let response = serde_json::json!({
            "clientMsgId": "symbols-list-1",
            "payloadType": 2115,
            "payload": {
                "ctidTraderAccountId": 7001,
                "symbol": [
                    {
                        "symbolId": 1,
                        "symbolName": "EUR/USD",
                        "enabled": true,
                        "description": "Euro vs Dollar"
                    },
                    {
                        "symbolId": 2,
                        "symbolName": "GBP/USD",
                        "enabled": false
                    }
                ],
                "archivedSymbol": [
                    {
                        "name": "AUD/USD"
                    }
                ]
            }
        });

        let result = parse_symbols_list_response(&response.to_string()).expect("symbols response");

        assert_eq!(result.account_id, 7001);
        assert_eq!(result.symbols.len(), 2);
        assert_eq!(result.symbols[0].symbol_id, 1);
        assert_eq!(result.symbols[0].symbol_name, "EUR/USD");
        assert!(result.symbols[0].enabled);
        assert_eq!(result.archived_symbols, vec!["AUD/USD"]);
    }

    #[test]
    fn symbol_by_id_response_parses_full_symbol_metadata() {
        // **2026-05-28 real-data correction**: previously this test
        // used a synthetic payload with `"tradingMode": "ENABLED"`
        // (string), asserted only 4 of the 30+ fields, and "passed"
        // — masking the production bug where the cTrader proxy
        // actually sends `"tradingMode": 0` (integer). Confirmed by
        // capturing real `ProtoOASymbolByIdRes` payloads via the new
        // `--capture-symbols` CLI. Test now uses the integer enum
        // encoding the broker really emits, asserting both the basic
        // identity fields and the `SymbolFinancials` projection.
        let response = serde_json::json!({
            "clientMsgId": "symbol-by-id-1",
            "payloadType": 2117,
            "payload": {
                "symbol": [
                    {
                        "symbolId": 1,
                        "digits": 5,
                        "pipPosition": 4,
                        "tradingMode": 0,
                        "commissionType": 1,
                        "preciseTradingCommissionRate": 4500000000_i64,
                        "swapCalculationType": 0,
                        "swapLong": -2.445,
                        "swapShort": -0.105,
                        "swapRollover3Days": 3,
                        "distanceSetIn": 1,
                        "enableShortSelling": true,
                        "minCommissionType": 2,
                        "pnlConversionFeeRate": 0
                    }
                ]
            }
        });

        let symbols = parse_symbol_by_id_response(&response.to_string()).expect("full symbols");

        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol_id, 1);
        assert_eq!(symbols[0].digits, 5);
        assert_eq!(symbols[0].pip_position, 4);
        assert!(symbols[0].is_trading_enabled);
        let financials = symbols[0]
            .financials
            .as_ref()
            .expect("financials projection must be present");
        assert_eq!(financials.trading_mode, Some(TradingModeProto::Enabled));
        assert_eq!(financials.commission_type, Some(CommissionType::UsdPerMillionUsd));
        assert_eq!(financials.precise_trading_commission_rate, Some(4_500_000_000));
        assert_eq!(financials.commission_rate_decimal(), Some(45.0));
        assert_eq!(
            financials.swap_calculation_type,
            Some(SwapCalculationType::Pips)
        );
        assert_eq!(financials.swap_long, Some(-2.445));
        assert_eq!(financials.swap_short, Some(-0.105));
        assert_eq!(financials.daily_swap_long(), Some(-2.445));
        assert_eq!(financials.daily_swap_short(), Some(-0.105));
        assert_eq!(financials.swap_rollover_3_days, Some(DayOfWeek::Wednesday));
        assert_eq!(
            financials.distance_set_in,
            Some(SymbolDistanceType::SymbolDistanceInPoints)
        );
        assert_eq!(
            financials.min_commission_type,
            Some(MinCommissionType::QuoteCurrency)
        );
        assert_eq!(financials.pnl_conversion_fee_rate, Some(0));
        assert!(financials.short_selling_allowed());
        assert!(financials.can_open_new_position());
    }

    #[test]
    fn symbol_by_id_response_parses_real_eurusd_broker_capture() {
        // Loads the verbatim payload captured from cTrader Demo
        // (account 47367144, 2026-05-28) via `--capture-symbols
        // EURUSD`. This is the parser's regression guard against the
        // ground-truth wire format the broker actually emits.
        // Updating the parser without re-running the capture is
        // forbidden — the fixture is the source of truth.
        let raw = include_str!("../../tests/fixtures/ctrader_symbol_EURUSD.raw.json");
        let symbols =
            parse_symbol_by_id_response(raw).expect("real cTrader EURUSD payload must parse");
        assert_eq!(symbols.len(), 1);
        let s = &symbols[0];
        assert_eq!(s.symbol_id, 1);
        assert_eq!(s.digits, 5);
        assert_eq!(s.pip_position, 4);
        assert!(s.is_trading_enabled);
        assert_eq!(s.lot_size, Some(10_000_000));
        let f = s.financials.as_ref().expect("financials must be populated");
        // Commission: $45/M USD (proto rate × 10^8 = 4_500_000_000).
        assert_eq!(f.commission_type, Some(CommissionType::UsdPerMillionUsd));
        assert_eq!(f.precise_trading_commission_rate, Some(4_500_000_000));
        assert_eq!(f.commission_rate_decimal(), Some(45.0));
        // Swap: PIPS-typed, both sides charge (negative = cost).
        assert_eq!(f.swap_calculation_type, Some(SwapCalculationType::Pips));
        assert!((f.swap_long.unwrap() - (-2.445)).abs() < 1e-9);
        assert!((f.swap_short.unwrap() - (-0.105)).abs() < 1e-9);
        assert_eq!(f.swap_period_hours, Some(24));
        assert_eq!(f.swap_time_minutes_from_utc_midnight, Some(1259));
        assert_eq!(f.swap_rollover_3_days, Some(DayOfWeek::Wednesday));
        // No fee for USD-quoted symbol on this USD-deposit account.
        assert_eq!(f.pnl_conversion_fee_rate, Some(0));
        // Trading mode + short selling.
        assert_eq!(f.trading_mode, Some(TradingModeProto::Enabled));
        assert_eq!(f.enable_short_selling, Some(true));
        // No minimum SL/TP distance — broker reports 0 for FX majors.
        assert_eq!(f.sl_distance_points, Some(0));
        assert_eq!(f.tp_distance_points, Some(0));
        assert_eq!(
            f.distance_set_in,
            Some(SymbolDistanceType::SymbolDistanceInPoints)
        );
        // 5 trading intervals (Mon-Fri), no holidays in this payload.
        assert_eq!(f.trading_intervals.len(), 5);
        assert!(f.holidays.is_empty());
    }

    #[test]
    fn symbol_by_id_response_parses_real_btcusd_with_missing_swap_calc_type() {
        // BTCUSD on this demo account ships with
        //   swap_long: 0.0
        //   swap_short: 0.0
        //   swap_calculation_type: <field omitted>
        // The fix in `daily_swap_long/short()` treats a missing
        // calc-type as PIPS (proto default) so the helper returns
        // Some(0.0) instead of None — which would have silently
        // zeroed swap in the backtest cost model.
        let raw = include_str!("../../tests/fixtures/ctrader_symbol_BTCUSD.raw.json");
        let symbols = parse_symbol_by_id_response(raw).expect("BTCUSD payload must parse");
        let s = &symbols[0];
        let f = s.financials.as_ref().expect("financials must be populated");
        assert_eq!(f.swap_calculation_type, None);
        assert_eq!(f.swap_long, Some(0.0));
        assert_eq!(f.swap_short, Some(0.0));
        // The fix: daily_swap_* returns Some(0.0), not None.
        assert_eq!(f.daily_swap_long(), Some(0.0));
        assert_eq!(f.daily_swap_short(), Some(0.0));
        // BTCUSD has zero commission on this demo broker.
        assert_eq!(f.precise_trading_commission_rate, Some(0));
        assert_eq!(f.commission_rate_decimal(), Some(0.0));
    }

    #[test]
    fn symbol_by_id_response_parses_real_xauusd_distinct_lot_and_commission() {
        // XAUUSD differs from FX in:
        //   - lot_size = 10_000 (vs FX 10_000_000) — exactly what
        //     A.4 (broker_api lots→wire) needed to NOT silently fall
        //     back to the 10M default.
        //   - commission = $25/M USD (vs FX $45/M USD).
        //   - measurement_units = "Oz" (ounces).
        //   - schedule_time_zone = "UTC" (vs FX "America/New_York").
        let raw = include_str!("../../tests/fixtures/ctrader_symbol_XAUUSD.raw.json");
        let symbols = parse_symbol_by_id_response(raw).expect("XAUUSD payload must parse");
        let s = &symbols[0];
        assert_eq!(s.lot_size, Some(10_000));
        assert_eq!(s.digits, 2);
        let f = s.financials.as_ref().expect("financials must be populated");
        assert_eq!(f.precise_trading_commission_rate, Some(2_500_000_000));
        assert_eq!(f.commission_rate_decimal(), Some(25.0));
        assert_eq!(f.measurement_units.as_deref(), Some("Oz"));
        assert_eq!(f.schedule_time_zone.as_deref(), Some("UTC"));
        // Swap is PIPS-typed, both sides charge (gold isn't free to hold).
        assert_eq!(f.swap_calculation_type, Some(SwapCalculationType::Pips));
        assert!(f.swap_long.unwrap() < 0.0); // charge, not credit
        assert!(f.swap_short.unwrap() < 0.0);
    }

    #[test]
    fn trendbars_response_normalizes_relative_prices_and_timestamps() {
        let symbol = CTraderSymbolInfo {
            symbol_id: 1,
            symbol_name: "EUR/USD".to_string(),
            display_name: "EUR/USD".to_string(),
            digits: 5,
            pip_position: 4,
            is_archived: false,
            is_trading_enabled: true,
            min_volume: None,
            max_volume: None,
            step_volume: None,
            lot_size: None,
            pnl_conversion_fee_rate: None,
            financials: None,
        };
        let response = serde_json::json!({
            "clientMsgId": "trendbars-1",
            "payloadType": 2138,
            "payload": {
                "period": "M15",
                "symbolId": 1,
                "hasMore": false,
                "trendbar": [
                    {
                        "volume": 12,
                        "low": 110000,
                        "deltaOpen": 25,
                        "deltaClose": 75,
                        "deltaHigh": 140,
                        "utcTimestampInMinutes": 28333333
                    }
                ]
            }
        });

        let result =
            parse_trendbars_response(&response.to_string(), &symbol).expect("trendbars response");

        assert_eq!(result.symbol_id, 1);
        assert_eq!(result.timeframe, "M15");
        assert_eq!(result.bars.len(), 1);
        assert_eq!(result.bars[0].timestamp_ms, 28_333_333_i64 * 60_000);
        assert!((result.bars[0].low - 1.10000).abs() < 1e-9);
        assert!((result.bars[0].open - 1.10025).abs() < 1e-9);
        assert!((result.bars[0].close - 1.10075).abs() < 1e-9);
        assert!((result.bars[0].high - 1.10140).abs() < 1e-9);
        assert_eq!(result.bars[0].volume, Some(12));
        assert!(!result.has_more);
    }

    #[test]
    fn tick_data_response_normalizes_relative_prices_and_descending_timestamps() {
        let symbol = CTraderSymbolInfo {
            symbol_id: 1,
            symbol_name: "EUR/USD".to_string(),
            display_name: "EUR/USD".to_string(),
            digits: 5,
            pip_position: 4,
            is_archived: false,
            is_trading_enabled: true,
            min_volume: None,
            max_volume: None,
            step_volume: None,
            lot_size: None,
            pnl_conversion_fee_rate: None,
            financials: None,
        };
        let response = serde_json::json!({
            "clientMsgId": "ticks-1",
            "payloadType": 2146,
            "payload": {
                "hasMore": true,
                "tickData": [
                    {
                        "timestamp": 1_700_000_000_000i64,
                        "tick": 110120
                    },
                    {
                        "timestamp": 250,
                        "tick": 110100
                    }
                ]
            }
        });

        let result = parse_tick_data_response(&response.to_string(), &symbol).expect("tick data");

        assert_eq!(result.symbol_id, 1);
        assert_eq!(result.ticks.len(), 2);
        assert!(result.ticks[0].timestamp_ms < result.ticks[1].timestamp_ms);
        assert_eq!(result.ticks[0].timestamp_ms, 1_699_999_999_750);
        assert_eq!(result.ticks[1].timestamp_ms, 1_700_000_000_000);
        assert!((result.ticks[0].price - 1.10100).abs() < 1e-9);
        assert!((result.ticks[1].price - 1.10120).abs() < 1e-9);
        assert!(result.has_more);
    }

    #[test]
    fn tick_data_response_rejects_mismatched_symbol_id() {
        let symbol = CTraderSymbolInfo {
            symbol_id: 1,
            symbol_name: "EUR/USD".to_string(),
            display_name: "EUR/USD".to_string(),
            digits: 5,
            pip_position: 4,
            is_archived: false,
            is_trading_enabled: true,
            min_volume: None,
            max_volume: None,
            step_volume: None,
            lot_size: None,
            pnl_conversion_fee_rate: None,
            financials: None,
        };
        let response = serde_json::json!({
            "clientMsgId": "ticks-1",
            "payloadType": 2146,
            "payload": {
                "symbolId": 2,
                "hasMore": false,
                "tickData": [
                    {
                        "timestamp": 1_700_000_000_000i64,
                        "tick": 110120
                    }
                ]
            }
        });

        let err = parse_tick_data_response(&response.to_string(), &symbol)
            .expect_err("mismatched symbol id should fail");
        assert!(
            err.to_string()
                .contains("unexpected cTrader tick-data symbol id"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn chart_history_backend_loads_symbol_metadata_then_historical_bars_and_ticks() {
        // After symbol resolution the v0.5.1.1 fix re-auths before fetching
        // symbol-by-id and trendbars (fresh WSS connection per call), so the
        // full sequence is 9 messages: initial auth (2) + symbols-list (1) +
        // re-auth (2) + symbol-by-id (1) + trendbars (1) + ticks×2 (2).
        let transport = StubTransport::with_responses(vec![
            Ok(r#"{"clientMsgId":"app-auth-1","payloadType":2101,"payload":{}}"#.to_string()),
            Ok(r#"{"clientMsgId":"account-auth-1","payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
            Ok(r#"{"clientMsgId":"symbols-1","payloadType":2115,"payload":{"ctidTraderAccountId":712345,"symbol":[{"symbolId":14,"symbolName":"EURUSD","enabled":true,"description":"Euro vs Dollar"}]}}"#.to_string()),
            // Re-auth on the second WSS connection (before symbol-by-id):
            Ok(r#"{"clientMsgId":"app-auth-2","payloadType":2101,"payload":{}}"#.to_string()),
            Ok(r#"{"clientMsgId":"account-auth-2","payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
            Ok(r#"{"clientMsgId":"symbol-by-id-1","payloadType":2117,"payload":{"symbol":[{"symbolId":14,"digits":5,"pipPosition":4,"tradingMode":"ENABLED"}]}}"#.to_string()),
            Ok(r#"{"clientMsgId":"trendbars-1","payloadType":2138,"payload":{"period":"M5","symbolId":14,"trendbar":[{"volume":9,"low":109950,"deltaOpen":50,"deltaClose":125,"deltaHigh":225,"utcTimestampInMinutes":28500000}],"hasMore":false}}"#.to_string()),
            Ok(r#"{"clientMsgId":"ticks-bid-1","payloadType":2146,"payload":{"symbolId":14,"hasMore":false,"tickData":[{"timestamp":1710000000000,"tick":109990},{"timestamp":200,"tick":109970}]}}"#.to_string()),
            Ok(r#"{"clientMsgId":"ticks-ask-1","payloadType":2146,"payload":{"symbolId":14,"hasMore":false,"tickData":[{"timestamp":1710000000000,"tick":110010},{"timestamp":200,"tick":109990}]}}"#.to_string()),
        ]);

        let result = load_chart_history_with_transport(
            &transport,
            &CTraderChartHistoryRequest {
                client_id: "client".to_string(),
                client_secret: "secret".to_string(),
                access_token: "token".to_string(),
                environment: CTraderEnvironment::Demo,
                account_id: "712345".to_string(),
                symbol_name: "EURUSD".to_string(),
                timeframe: "M5".to_string(),
                from_timestamp_ms: 1_709_000_000_000,
                to_timestamp_ms: 1_710_000_000_000,
                count: Some(96),
            },
        )
        .expect("chart history");

        assert_eq!(result.symbol.symbol_name, "EURUSD");
        assert_eq!(result.symbol.symbol_id, 14);
        assert_eq!(result.symbol.digits, 5);
        assert_eq!(result.bars.len(), 1);
        assert_eq!(result.bars[0].open, 1.1);
        assert_eq!(result.bars[0].close, 1.10075);
        assert_eq!(result.bid_ticks.len(), 2);
        assert_eq!(result.ask_ticks.len(), 2);
        assert_eq!(
            result.live_subscription_plan.subscribe_spots.payload_type,
            crate::app_services::ctrader_messages::CTRADER_OA_SUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE
        );
        assert_eq!(
            result.live_subscription_plan.subscribe_trendbars.payload_type,
            crate::app_services::ctrader_messages::CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE
        );
        assert_eq!(
            result.live_subscription_plan.unsubscribe_spots.payload_type,
            crate::app_services::ctrader_messages::CTRADER_OA_UNSUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE
        );
        assert_eq!(
            result.live_subscription_plan.unsubscribe_trendbars.payload_type,
            crate::app_services::ctrader_messages::CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE
        );
        assert_eq!(transport.sent_len(), 9);
    }

    #[test]
    fn chart_history_backend_rejects_unknown_symbol_name() {
        let transport = StubTransport::with_responses(vec![
            Ok(r#"{"clientMsgId":"app-auth-1","payloadType":2101,"payload":{}}"#.to_string()),
            Ok(r#"{"clientMsgId":"account-auth-1","payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
            Ok(
                r#"{"clientMsgId":"symbols-1","payloadType":2115,"payload":{"ctidTraderAccountId":712345,"symbol":[{"symbolId":14,"symbolName":"GBPUSD","enabled":true,"description":"Cable"}]}}"#
                    .to_string(),
            ),
        ]);

        let err = load_chart_history_with_transport(
            &transport,
            &CTraderChartHistoryRequest {
                client_id: "client".to_string(),
                client_secret: "secret".to_string(),
                access_token: "token".to_string(),
                environment: CTraderEnvironment::Demo,
                account_id: "712345".to_string(),
                symbol_name: "EURUSD".to_string(),
                timeframe: "M5".to_string(),
                from_timestamp_ms: 1_709_000_000_000,
                to_timestamp_ms: 1_710_000_000_000,
                count: Some(96),
            },
        )
        .expect_err("unknown symbol must fail");

        assert!(err.to_string().contains("EURUSD"));
    }

    #[test]
    fn bars_only_backend_loads_symbol_metadata_then_trendbars_without_ticks() {
        // Every production send_sequence opens a fresh WSS connection, so
        // symbol list, symbol detail, and trendbars each re-auth.
        let transport = StubTransport::with_responses(vec![
            Ok(r#"{"clientMsgId":"app-auth-1","payloadType":2101,"payload":{}}"#.to_string()),
            Ok(r#"{"clientMsgId":"account-auth-1","payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
            Ok(r#"{"clientMsgId":"symbols-1","payloadType":2115,"payload":{"ctidTraderAccountId":712345,"symbol":[{"symbolId":14,"symbolName":"EURUSD","enabled":true,"description":"Euro vs Dollar"}]}}"#.to_string()),
            // Re-auth on the second WSS connection (before symbol-by-id):
            Ok(r#"{"clientMsgId":"app-auth-2","payloadType":2101,"payload":{}}"#.to_string()),
            Ok(r#"{"clientMsgId":"account-auth-2","payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
            Ok(r#"{"clientMsgId":"symbol-by-id-1","payloadType":2117,"payload":{"symbol":[{"symbolId":14,"digits":5,"pipPosition":4,"tradingMode":"ENABLED"}]}}"#.to_string()),
            // Re-auth on the third WSS connection (before trendbars):
            Ok(r#"{"clientMsgId":"app-auth-3","payloadType":2101,"payload":{}}"#.to_string()),
            Ok(r#"{"clientMsgId":"account-auth-3","payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
            Ok(r#"{"clientMsgId":"trendbars-1","payloadType":2138,"payload":{"period":"M15","symbolId":14,"trendbar":[{"volume":9,"low":109950,"deltaOpen":50,"deltaClose":125,"deltaHigh":225,"utcTimestampInMinutes":28500000}],"hasMore":false}}"#.to_string()),
        ]);

        let result = load_historical_bars_only_with_transport(
            &transport,
            &CTraderChartHistoryRequest {
                client_id: "client".to_string(),
                client_secret: "secret".to_string(),
                access_token: "token".to_string(),
                environment: CTraderEnvironment::Demo,
                account_id: "712345".to_string(),
                symbol_name: "EURUSD".to_string(),
                timeframe: "M15".to_string(),
                from_timestamp_ms: 1_709_000_000_000,
                to_timestamp_ms: 1_710_000_000_000,
                count: Some(96),
            },
        )
        .expect("bars-only history");

        assert_eq!(result.symbol.symbol_name, "EURUSD");
        assert_eq!(result.bars.len(), 1);
        assert_eq!(result.bars[0].close, 1.10075);
        assert!(!result.has_more);
        assert_eq!(transport.sent_len(), 9);
    }
}
