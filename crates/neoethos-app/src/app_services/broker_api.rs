//! Pure helpers that turn `(broker_credentials.toml + keyring token)`
//! + a request into a live cTrader Open API result, without going
//! through `TradingSession`. Used by:
//!
//!   - `server::symbols_control`   (GET /broker/symbols)
//!   - `server::data_control`      (POST /data/fetch)
//!
//! Both endpoints need the same setup dance: load broker settings,
//! pull the access token, materialise the Spotware host. Keeping that
//! in one place keeps the route modules thin.

use anyhow::{Result, anyhow};
use std::path::PathBuf;

use crate::app_services::bootstrap_writer::write_bootstrap_vortex;
use crate::app_services::broker_config::CTraderBrokerEnvironment;
use crate::app_services::broker_persistence::load_broker_settings;
use crate::app_services::ctrader_bootstrap::NormalizedBar;
use crate::app_services::ctrader_data::{
    CTraderChartHistoryRequest, CTraderHistoricalBarsFetchResult, CTraderLightSymbolInfo,
    CTraderResolvedSymbol, CTraderSymbolLookupRequest, CTraderSymbolsListResult, HistoricalBar,
    load_historical_bars_only, parse_asset_class_list_response,
    parse_symbol_category_list_response, parse_symbols_list_response, resolve_symbol,
};
use crate::app_services::ctrader_execution::{
    CTraderExecutionBackend, CTraderExecutionOutcome, CTraderExecutionRequest,
    CTraderExecutionRuntimeRequest, ProductionCTraderExecutionBackend,
};
use crate::app_services::ctrader_messages::{
    CTraderCancelOrderRequest, CTraderClosePositionRequest, CTraderNewOrderRequest,
    CTraderOrderType, CTraderTradeSide,
};
use crate::app_services::ctrader_messages::{
    CTraderOpenApiTransport, ProductionCTraderOpenApiTransport, build_account_auth_request,
    build_application_auth_request, build_asset_class_list_request,
    build_symbol_category_list_request, build_symbols_list_request,
};
use crate::app_services::secure_store::production_ctrader_token_store;
use crate::app_services::trading::CTraderEnvironment;

/// What `/broker/symbols` ultimately returns over the wire — kept here
/// so the server module just shovels it to JSON.
#[derive(Debug, Clone)]
pub struct BrokerSymbolsBundle {
    pub account_id: i64,
    pub environment: &'static str,
    pub symbols: Vec<CTraderLightSymbolInfo>,
    pub archived_symbols: Vec<String>,
    /// F-341: `symbol_id → canonical asset bucket` ("forex" | "metals" |
    /// "indices" | "commodities"). Built from the broker's own
    /// asset-class / symbol-category tables. Empty when the broker's
    /// classification RPCs failed (in which case `symbols` is the
    /// unfiltered list — we never blank the Markets tab over a
    /// classification hiccup).
    pub asset_class_by_id: std::collections::HashMap<i64, String>,
}

/// What `/broker/accounts` returns. Sourced from
/// `ProtoOAGetAccountListByAccessTokenReq` (payload 2149/2150) — the
/// authoritative list of accounts the user granted access to during
/// OAuth. Used by the Settings screen's account picker so the user
/// doesn't have to type a numeric cTID by hand (and end up with a
/// stale ID that returns CH_ACCESS_TOKEN_INVALID).
#[derive(Debug, Clone)]
pub struct BrokerAccountsBundle {
    pub environment: &'static str,
    pub permission_scope: String,
    pub accounts: Vec<BrokerAccountInfo>,
}

#[derive(Debug, Clone)]
pub struct BrokerAccountInfo {
    pub account_id: String,
    pub broker_title: String,
    pub account_name: String,
    pub trader_login: Option<i64>,
    pub is_live: Option<bool>,
    pub enabled_for_execution: bool,
}

/// Bundled outcome of a historical fetch.
#[derive(Debug, Clone)]
pub struct HistoricalDownloadOutcome {
    pub symbol: String,
    pub timeframe: String,
    pub bar_count: usize,
    pub has_more: bool,
    pub written_path: PathBuf,
}

/// Resolve broker credentials + token bundle into the four primitives
/// every downstream call needs: client_id, client_secret, access_token,
/// account_id_string, environment.
struct ResolvedCreds {
    client_id: String,
    client_secret: String,
    access_token: String,
    account_id_str: String,
    environment: CTraderEnvironment,
    env_label: &'static str,
}

fn resolve_creds() -> Result<ResolvedCreds> {
    let settings = load_broker_settings();
    let ct = &settings.ctrader;
    if ct.client_id.is_empty() || ct.client_secret.is_empty() {
        return Err(anyhow!(
            "cTrader client_id / client_secret are empty in \
             broker_credentials.toml; the wizard / --reauth must run first"
        ));
    }
    let account = ct
        .accounts
        .first()
        .ok_or_else(|| anyhow!("no cTrader account configured"))?;

    let bundle = production_ctrader_token_store()
        .load_token_bundle_with_legacy_fallback()
        .map_err(|e| anyhow!("token bundle load failed: {e}"))?
        .ok_or_else(|| {
            anyhow!(
                "no cTrader token bundle saved yet — run \
                 `neoethos-app --reauth` (or click Re-authenticate \
                 in Broker Setup) first"
            )
        })?;

    let (env, env_label) = match ct.environment {
        CTraderBrokerEnvironment::Demo => (CTraderEnvironment::Demo, "Demo"),
        CTraderBrokerEnvironment::Live => (CTraderEnvironment::Live, "Live"),
    };

    Ok(ResolvedCreds {
        client_id: ct.client_id.clone(),
        client_secret: ct.client_secret.clone(),
        access_token: bundle.access_token,
        account_id_str: account.account_id.clone(),
        environment: env,
        env_label,
    })
}

/// Hit `ProtoOAGetAccountListByAccessTokenReq` (payload 2149/2150) and
/// return every account the user granted access to during OAuth.
///
/// Differs from `resolve_creds` in one key way: it does NOT require an
/// account_id to already be configured. That's the whole point — we
/// call this BEFORE the user has picked an account, so the Settings
/// dropdown can show them what's available without making them type a
/// numeric cTID by hand. client_id/secret + access_token are enough.
///
/// Blocking; callers must wrap in `spawn_blocking`.
pub fn fetch_broker_accounts_blocking() -> Result<BrokerAccountsBundle> {
    use crate::app_services::ctrader_live_auth::{
        CTraderAccountDiscoveryBackend, CTraderAccountDiscoveryRequest,
        ProductionCTraderLiveAuthBackend,
    };

    let settings = load_broker_settings();
    let ct = &settings.ctrader;
    if ct.client_id.is_empty() || ct.client_secret.is_empty() {
        return Err(anyhow!(
            "cTrader client_id / client_secret are empty in \
             broker_credentials.toml. Save them in Settings first."
        ));
    }

    let bundle = production_ctrader_token_store()
        .load_token_bundle_with_legacy_fallback()
        .map_err(|e| anyhow!("token bundle load failed: {e}"))?
        .ok_or_else(|| {
            anyhow!(
                "no cTrader token bundle saved yet — open Broker Setup \
                 and click Re-authenticate first"
            )
        })?;

    let (env, env_label) = match ct.environment {
        CTraderBrokerEnvironment::Demo => (CTraderEnvironment::Demo, "Demo"),
        CTraderBrokerEnvironment::Live => (CTraderEnvironment::Live, "Live"),
    };

    let request = CTraderAccountDiscoveryRequest {
        client_id: ct.client_id.clone(),
        client_secret: ct.client_secret.clone(),
        access_token: bundle.access_token,
        environment: env,
    };

    // `ProductionCTraderLiveAuthBackend` is a unit struct — no ::new
    // or ::default() needed; instantiate directly. The discovery call
    // does its own ProtoOAApplicationAuth handshake internally, so we
    // don't need to wire the transport here.
    let backend = ProductionCTraderLiveAuthBackend;
    let result = backend
        .discover_accounts(&request)
        .map_err(|e| anyhow!("cTrader account-list call failed: {e}"))?;

    let accounts: Vec<BrokerAccountInfo> = result
        .accounts
        .into_iter()
        .map(|a| BrokerAccountInfo {
            account_id: a.account_id,
            broker_title: a.broker_title,
            account_name: a.account_name,
            trader_login: a.trader_login,
            is_live: a.is_live,
            enabled_for_execution: a.enabled_for_execution,
        })
        .collect();

    Ok(BrokerAccountsBundle {
        environment: env_label,
        permission_scope: result.permission_scope,
        accounts,
    })
}

/// Hit the cTrader symbols-list endpoint and return the parsed bundle.
///
/// Blocking — the transport uses synchronous WSS + reqwest::blocking.
/// Callers must wrap in `spawn_blocking`.
pub fn fetch_broker_symbols_blocking() -> Result<BrokerSymbolsBundle> {
    let creds = resolve_creds()?;
    let account_id: i64 = creds
        .account_id_str
        .parse()
        .map_err(|_| anyhow!("account_id '{}' is not numeric", creds.account_id_str))?;

    let transport = ProductionCTraderOpenApiTransport::new(creds.environment.endpoint_host());
    // F-341: one connection, five requests — symbols list + the broker's
    // own asset-class and symbol-category tables. The latter two let us
    // restrict the catalog to forex/metals/indices/commodities (dropping
    // the broker's 700+ equities & ETFs the engine never trades) using
    // the broker's classification, not name-pattern guesses.
    let responses = transport.send_sequence(&[
        build_application_auth_request(&creds.client_id, &creds.client_secret, "app-auth-1"),
        build_account_auth_request(account_id, &creds.access_token, "account-auth-1"),
        build_symbols_list_request(account_id, false, "symbols-1"),
        build_asset_class_list_request(account_id, "asset-classes-1"),
        build_symbol_category_list_request(account_id, "symbol-categories-1"),
    ])?;
    if responses.len() < 3 {
        return Err(anyhow!(
            "expected ≥3 cTrader symbols-list responses, received {}",
            responses.len()
        ));
    }

    let CTraderSymbolsListResult {
        account_id,
        symbols,
        archived_symbols,
    } = parse_symbols_list_response(&responses[2])?;

    // Build `category_id → canonical bucket` from the broker tables.
    // Best-effort: if either RPC is missing or unparseable we log and
    // fall through to the unfiltered list (an empty bucket map), so a
    // classification hiccup never blanks the Markets tab.
    let category_bucket: std::collections::HashMap<i64, &'static str> = (|| {
        let classes = parse_asset_class_list_response(responses.get(3)?).ok()?;
        let categories = parse_symbol_category_list_response(responses.get(4)?).ok()?;
        // class_id → canonical bucket, keeping only the forex-ai classes.
        let class_bucket: std::collections::HashMap<i64, &'static str> = classes
            .iter()
            .filter(|c| crate::app_services::capture_symbols::is_forex_ai_asset_class(&c.name))
            .map(|c| (c.id, canonical_asset_bucket(&c.name)))
            .collect();
        Some(
            categories
                .iter()
                .filter_map(|cat| {
                    class_bucket
                        .get(&cat.asset_class_id)
                        .map(|bucket| (cat.id, *bucket))
                })
                .collect(),
        )
    })()
    .unwrap_or_default();

    if category_bucket.is_empty() {
        // Classification unavailable — return everything, untagged. The
        // UI picker falls back to its own name heuristics in this case.
        tracing::warn!(
            "broker symbol classification unavailable; returning all {} symbols unfiltered",
            symbols.len()
        );
        return Ok(BrokerSymbolsBundle {
            account_id,
            environment: creds.env_label,
            symbols,
            archived_symbols,
            asset_class_by_id: std::collections::HashMap::new(),
        });
    }

    // Keep only symbols whose category resolves to a forex-ai bucket;
    // tag each kept symbol with that bucket for the UI category chips.
    let total_raw = symbols.len();
    let mut asset_class_by_id: std::collections::HashMap<i64, String> =
        std::collections::HashMap::new();
    let filtered: Vec<CTraderLightSymbolInfo> = symbols
        .into_iter()
        .filter(|s| {
            match s
                .symbol_category_id
                .and_then(|cid| category_bucket.get(&cid))
            {
                Some(bucket) => {
                    asset_class_by_id.insert(s.symbol_id, (*bucket).to_string());
                    true
                }
                // Unknown / uncategorised → drop (matches the bootstrap's
                // conservative "no category = not forex" stance).
                None => false,
            }
        })
        .collect();

    tracing::info!(
        "broker symbols classified: kept {} of {} (forex/metals/indices/commodities)",
        filtered.len(),
        total_raw
    );

    Ok(BrokerSymbolsBundle {
        account_id,
        environment: creds.env_label,
        symbols: filtered,
        archived_symbols,
        asset_class_by_id,
    })
}

/// Map a broker asset-class name onto one of the four canonical buckets
/// the UI groups by. Order matters: "metal" / "indic" / "commodit" are
/// checked before the forex default so e.g. "Spot Metals" lands in
/// `metals` rather than the catch-all. Only called for names that
/// already passed [`is_forex_ai_asset_class`].
fn canonical_asset_bucket(class_name: &str) -> &'static str {
    let lower = class_name.to_ascii_lowercase();
    if lower.contains("metal") {
        "metals"
    } else if lower.contains("indic") || lower.contains("index") {
        "indices"
    } else if lower.contains("commodit")
        || lower.contains("energ")
        || lower.contains("oil")
        || lower.contains("gas")
    {
        "commodities"
    } else {
        // forex / fx / currencies — the remaining keep-list classes.
        "forex"
    }
}

/// Download historical bars for [from_ms, to_ms] and write the result
/// into the local data dir. Auto-chunked: cTrader caps each
/// ProtoOAGetTrendbarsReq at ~5000 bars, so for wide windows we loop
/// — sliding `to_ms` backwards by the timeframe's natural span until
/// we cover the requested range. Accumulated bars are deduped and
/// sorted by timestamp before the single vortex write.
///
/// Blocking; callers must wrap in `spawn_blocking`.
pub fn download_history_blocking(
    symbol: &str,
    timeframe: &str,
    from_ms: i64,
    to_ms: i64,
    data_root: &std::path::Path,
) -> Result<HistoricalDownloadOutcome> {
    if to_ms <= from_ms {
        return Err(anyhow!(
            "invalid range: from_ms ({from_ms}) must be < to_ms ({to_ms})"
        ));
    }

    let creds = resolve_creds()?;
    let chunk_ms = timeframe_chunk_ms(timeframe);

    // Walk the window in `chunk_ms`-wide slices, latest first. Stops
    // either when we cross from_ms or when a slice returns 0 bars
    // (broker has nothing earlier — markets weren't trading, etc).
    let mut all_bars: Vec<HistoricalBar> = Vec::new();
    let mut cursor_to = to_ms;
    let mut has_more_overall = false;
    let max_chunks = 100; // hard cap so a misconfigured range can't loop forever
    let mut chunk_count = 0;
    while cursor_to > from_ms && chunk_count < max_chunks {
        let cursor_from = (cursor_to - chunk_ms).max(from_ms);
        let request = CTraderChartHistoryRequest {
            client_id: creds.client_id.clone(),
            client_secret: creds.client_secret.clone(),
            access_token: creds.access_token.clone(),
            environment: creds.environment,
            account_id: creds.account_id_str.clone(),
            symbol_name: symbol.to_string(),
            timeframe: timeframe.to_string(),
            from_timestamp_ms: cursor_from,
            to_timestamp_ms: cursor_to,
            count: None,
        };
        let CTraderHistoricalBarsFetchResult { bars, has_more, .. } =
            load_historical_bars_only(&request)?;
        if bars.is_empty() {
            // No more data going further back in time — stop.
            break;
        }
        if has_more {
            // The broker still has more inside this chunk than fit
            // in the response. Carry that flag through so the UI can
            // hint the user to widen their range or split it further.
            has_more_overall = true;
        }
        all_bars.extend(bars);
        cursor_to = cursor_from;
        chunk_count += 1;
    }

    // Dedupe + sort. Multiple chunks can overlap by 1 bar at the
    // boundary; dedupe on timestamp keeps the dataset clean.
    all_bars.sort_by_key(|b| b.timestamp_ms);
    all_bars.dedup_by_key(|b| b.timestamp_ms);

    let normalized = bars_to_normalized(&all_bars);
    let written_path = write_bootstrap_vortex(data_root, symbol, timeframe, &normalized)?;

    Ok(HistoricalDownloadOutcome {
        symbol: symbol.to_string(),
        timeframe: timeframe.to_string(),
        bar_count: all_bars.len(),
        has_more: has_more_overall,
        written_path,
    })
}

/// How wide a single ProtoOAGetTrendbarsReq slice should be for the
/// given timeframe. cTrader caps each response at ~5000 bars; we
/// stay below that with the values below so we never bump the cap.
///
///   M1  →  3 days   (4320 bars)
///   M3  →  9 days   (4320 bars)
///   M5  →  15 days  (4320 bars)
///   M15 →  45 days  (4320 bars)
///   M30 →  90 days
///   H1  →  180 days (4320 bars)
///   H4  →  720 days
///   H12 →  6 years
///   D1  →  12 years
///   W1/MN1 → no chunking needed in practice (one shot covers
///            available history)
fn timeframe_chunk_ms(tf: &str) -> i64 {
    let day_ms: i64 = 24 * 60 * 60 * 1000;
    match tf.trim().to_ascii_uppercase().as_str() {
        "M1" => 3 * day_ms,
        "M3" => 9 * day_ms,
        "M5" => 15 * day_ms,
        "M15" => 45 * day_ms,
        "M30" => 90 * day_ms,
        "H1" => 180 * day_ms,
        "H4" => 720 * day_ms,
        "H12" => 6 * 365 * day_ms,
        "D1" => 12 * 365 * day_ms,
        // For W1 / MN1 the broker's full coverage is usually <500
        // bars, so one big slice covers everything.
        _ => 50 * 365 * day_ms,
    }
}

/// Fetch the most recent `limit` OHLCV bars for `symbol`/`timeframe`
/// straight from the cTrader broker (`ProtoOAGetTrendbarsReq`) with NO
/// disk write — the chart's broker-passthrough path (the authoritative,
/// *current* source). Returns bars sorted oldest→newest, trimmed to the
/// trailing `limit`. Opens a fresh WSS connection + re-auths, same as the
/// history-download path, so callers must run it on a blocking task.
pub fn fetch_recent_chart_bars_blocking(
    symbol: &str,
    timeframe: &str,
    limit: usize,
) -> Result<Vec<HistoricalBar>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let creds = resolve_creds()?;
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let step_ms = chart_bar_step_ms(timeframe);
    // Window wide enough to contain `limit` bars with generous headroom
    // for weekends / holidays / illiquid gaps (markets aren't open 24/7,
    // so a tight window would starve the requested count). cTrader caps a
    // single response at ~5000 bars and `count` bounds the result, so one
    // request covers a chart (limit ≤ MAX_LIMIT = 2000).
    let span_ms = step_ms
        .saturating_mul(limit as i64)
        .saturating_mul(3)
        .max(step_ms);
    let from_ms = now_ms.saturating_sub(span_ms);
    let request = CTraderChartHistoryRequest {
        client_id: creds.client_id.clone(),
        client_secret: creds.client_secret.clone(),
        access_token: creds.access_token.clone(),
        environment: creds.environment,
        account_id: creds.account_id_str.clone(),
        symbol_name: symbol.to_string(),
        timeframe: timeframe.to_string(),
        from_timestamp_ms: from_ms,
        to_timestamp_ms: now_ms,
        count: Some(limit as u32),
    };
    let CTraderHistoricalBarsFetchResult { mut bars, .. } =
        load_historical_bars_only(&request)?;
    bars.sort_by_key(|b| b.timestamp_ms);
    bars.dedup_by_key(|b| b.timestamp_ms);
    // The broker may return a few more than requested — keep trailing N.
    if bars.len() > limit {
        bars.drain(0..bars.len() - limit);
    }
    Ok(bars)
}

/// Fetch up to `limit` OHLCV bars ENDING strictly before `before_ms`,
/// straight from the broker with **NO disk write** — the chart's
/// scroll-back pagination path. This is the TradingView model: when the
/// operator pans left past the oldest loaded candle, the client asks for
/// the next page of older history, holds it only in memory, and never
/// persists it. Two years of scroll-back therefore costs zero disk — the
/// local Vortex cache is only ever written by the explicit Data
/// Bootstrap / discovery auto-fetch paths, never by viewing a chart.
///
/// Returns bars sorted oldest→newest, every one with
/// `timestamp_ms < before_ms`, so the client can splice the result onto
/// the front of its list without overlap. Empty result ⇒ the broker has
/// nothing older (we've reached the start of its coverage). Opens a fresh
/// WSS connection + re-auths, so callers must run it on a blocking task.
pub fn fetch_chart_bars_before_blocking(
    symbol: &str,
    timeframe: &str,
    before_ms: i64,
    limit: usize,
) -> Result<Vec<HistoricalBar>> {
    if limit == 0 || before_ms <= 0 {
        return Ok(Vec::new());
    }
    let creds = resolve_creds()?;
    let step_ms = chart_bar_step_ms(timeframe);
    // Same generous headroom as the recent-bars path: markets aren't open
    // 24/7, so the wall-clock window must be wider than `limit × step` to
    // actually contain `limit` bars. `count` bounds the response so the
    // wide window never over-fetches.
    let span_ms = step_ms
        .saturating_mul(limit as i64)
        .saturating_mul(3)
        .max(step_ms);
    let from_ms = before_ms.saturating_sub(span_ms).max(0);
    let request = CTraderChartHistoryRequest {
        client_id: creds.client_id.clone(),
        client_secret: creds.client_secret.clone(),
        access_token: creds.access_token.clone(),
        environment: creds.environment,
        account_id: creds.account_id_str.clone(),
        symbol_name: symbol.to_string(),
        timeframe: timeframe.to_string(),
        from_timestamp_ms: from_ms,
        to_timestamp_ms: before_ms,
        count: Some(limit as u32),
    };
    let CTraderHistoricalBarsFetchResult { mut bars, .. } =
        load_historical_bars_only(&request)?;
    bars.sort_by_key(|b| b.timestamp_ms);
    bars.dedup_by_key(|b| b.timestamp_ms);
    // Drop any bar at/after the cursor so the page is strictly older.
    bars.retain(|b| b.timestamp_ms < before_ms);
    if bars.len() > limit {
        let cut = bars.len() - limit;
        bars.drain(0..cut);
    }
    Ok(bars)
}

/// Duration of a single bar for the canonical timeframe, in ms. Used to
/// size the broker fetch window in [`fetch_recent_chart_bars_blocking`]
/// and [`fetch_chart_bars_before_blocking`].
fn chart_bar_step_ms(tf: &str) -> i64 {
    let m: i64 = 60 * 1000;
    match tf.trim().to_ascii_uppercase().as_str() {
        "M1" => m,
        "M3" => 3 * m,
        "M5" => 5 * m,
        "M15" => 15 * m,
        "M30" => 30 * m,
        "H1" => 60 * m,
        "H4" => 240 * m,
        "H12" => 720 * m,
        "D1" => 1440 * m,
        "W1" => 7 * 1440 * m,
        "MN1" => 30 * 1440 * m,
        _ => m,
    }
}

/// Side of a manual market order. Mirrors `CTraderTradeSide` but kept
/// here so the server module doesn't depend on the cTrader-internal
/// enum directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide {
    Buy,
    Sell,
}

impl From<OrderSide> for CTraderTradeSide {
    fn from(s: OrderSide) -> Self {
        match s {
            OrderSide::Buy => CTraderTradeSide::Buy,
            OrderSide::Sell => CTraderTradeSide::Sell,
        }
    }
}

/// Submit a Market order for `symbol` with the given side + lot size
/// + SL/TP **in pips relative to fill price** (cTrader rejects
/// absolute SL/TP on Market orders with "SL/TP in absolute values are
/// allowed only for LIMIT/STOP/STOP_LIMIT"). Pass `None` to leave the
/// bracket off — but the UI MUST hard-require at least one for
/// risk-control reasons.
///
/// `stop_loss_pips` / `take_profit_pips` are positive distances:
///   - BUY:  SL price = fill - sl_pips * 1 pip
///           TP price = fill + tp_pips * 1 pip
///   - SELL: mirror.
///
/// Blocking — wraps `ProductionCTraderExecutionBackend::execute`
/// which uses sync WSS. Callers must `spawn_blocking`.
pub fn submit_market_order_blocking(
    symbol: &str,
    side: OrderSide,
    volume_lots: f64,
    stop_loss_pips: Option<f64>,
    take_profit_pips: Option<f64>,
    comment: Option<String>,
) -> Result<CTraderExecutionOutcome> {
    if !(volume_lots.is_finite() && volume_lots > 0.0) {
        return Err(anyhow!(
            "volume_lots must be a finite positive number (got {volume_lots})"
        ));
    }
    for (name, val) in [
        ("stop_loss_pips", stop_loss_pips),
        ("take_profit_pips", take_profit_pips),
    ] {
        if let Some(v) = val {
            if !v.is_finite() || v <= 0.0 {
                return Err(anyhow!(
                    "{name} must be a finite positive number when set (got {v})"
                ));
            }
        }
    }
    let creds = resolve_creds()?;

    // Resolve the symbol so we know its id + lot_size for volume
    // conversion.
    //
    // **2026-05-26 fix v2 (Κωνσταντίνος)**: cTrader's
    // `ProtoOASymbol.lot_size` is documented as "Lot size in
    // 1/100 of a unit" — i.e., it's ALREADY in cents (centi-units
    // of base currency). For EURUSD the broker returns
    // 10_000_000 = 100,000 EUR × 100 cents. The prior code further
    // multiplied by `* 100.0` on top of that, which made every
    // order 100× larger than the operator requested — a default
    // 0.01-lot click opened a 1.0-lot position (100k EUR exposure
    // instead of 1k), and on cTrader Demo the silent inflation went
    // unnoticed until live close-position rejection surfaced the
    // volume mismatch.
    //
    // Verified empirically against this Demo account (47367144,
    // 2026-05-26): user typed 0.01 → backend computed
    // 0.01 × 10_000_000 × 100 = 10_000_000 → broker stored a
    // 1.0-lot position with `tradeData.volume = 10_000_000`. Removing
    // the spurious `× 100` makes 0.01 × 10_000_000 = 100_000 wire,
    // which is exactly 0.01 lot (1,000 EUR exposure × 100 cents).
    //
    // **2026-05-27 — A.4 fix (Cycle-3 Phase A)**: route the
    // conversion through `SymbolMetadata::lots_to_wire_volume` so
    // (a) overflow + non-finite inputs are caught by the helper's
    // explicit guards rather than the silent `as i64` saturation,
    // and (b) we no longer silently fall back to 10_000_000 cents
    // when the broker forgot `lotSize`. That fallback was correct
    // for FX majors but **1000× wrong for XAU** (gold has
    // `lotSize=100`) and similarly wrong for indices/CFDs. A
    // missing-catalog entry is now a hard failure — operator sees
    // the bug instead of placing a wildly mis-sized order.
    let resolved: CTraderResolvedSymbol = resolve_symbol(&CTraderSymbolLookupRequest {
        client_id: creds.client_id.clone(),
        client_secret: creds.client_secret.clone(),
        access_token: creds.access_token.clone(),
        environment: creds.environment,
        account_id: creds.account_id_str.clone(),
        symbol_name: symbol.to_string(),
    })?;
    let lot_size = resolved.symbol.lot_size.ok_or_else(|| {
        anyhow!(
            "broker omitted lotSize for {symbol}; refusing to fall back \
             to a synthetic 10,000,000-cents default (would be 1000× wrong \
             for XAU/XAG/index symbols). Re-fetch /broker/symbols or check \
             the cTrader symbol catalog endpoint."
        )
    })?;
    let meta = neoethos_core::symbol_metadata::resolve(symbol).ok_or_else(|| {
        anyhow!(
            "no SymbolMetadata for {symbol} — wire-volume conversion needs \
             pip_size/contract_size to bounds-check the result. Populate \
             data/symbol_metadata.json (or its env override) from the \
             ProtoOASymbol records before trading."
        )
    })?;
    let volume_units = meta.lots_to_wire_volume(volume_lots, Some(lot_size)).ok_or_else(
        || {
            anyhow!(
                "could not derive cTrader wire volume for {symbol}: \
                 lots={volume_lots}, lot_size_cents={lot_size}. \
                 Inputs must be finite, positive, and within i64 range."
            )
        },
    )?;
    if let Some(min) = resolved.symbol.min_volume {
        if volume_units < min {
            return Err(anyhow!(
                "volume {volume_units} is below broker min_volume {min} \
                 for {symbol}"
            ));
        }
    }
    if let Some(max) = resolved.symbol.max_volume {
        if volume_units > max {
            return Err(anyhow!(
                "volume {volume_units} exceeds broker max_volume {max} \
                 for {symbol}"
            ));
        }
    }

    // cTrader relative_stop_loss is in 1e-5 base-price units.
    // For 5-digit FX (EURUSD etc): 1 pip = 0.0001 = 10 * 1e-5, so
    //   relative_units = pips * 10.
    // For 3-digit JPY pairs: 1 pip = 0.01 = 1000 * 1e-5, so
    //   relative_units = pips * 1000.
    // Generally: relative_units = pips * 10^(digits - 4).
    let digits = resolved.symbol.digits.max(0) as u32;
    // For 5-digit FX, pip_in_units = 10 (i.e. 10 * 1e-5 = 0.0001).
    // We map every "pip" to `10^(digits-4)` 1e-5 units, then clamp
    // at 1 so 3-digit JPY (digits=3) still resolves to something sane
    // — though cTrader's standard contracts are 5/3 digits anyway.
    let pip_relative_units: f64 = if digits >= 4 {
        10f64.powi((digits - 4) as i32 + 1)
    } else {
        1.0
    };
    let relative_stop_loss = stop_loss_pips.map(|p| (p * pip_relative_units).round() as i64);
    let relative_take_profit = take_profit_pips.map(|p| (p * pip_relative_units).round() as i64);

    let new_order = CTraderNewOrderRequest {
        account_id: resolved.account_id,
        symbol_id: resolved.light_symbol.symbol_id,
        order_type: CTraderOrderType::Market,
        trade_side: side.into(),
        volume: volume_units,
        limit_price: None,
        stop_price: None,
        time_in_force: None,
        expiration_timestamp_ms: None,
        // For Market orders, ABSOLUTE SL/TP fields are rejected by
        // cTrader ("SL/TP in absolute values are allowed only for
        // LIMIT/STOP/STOP_LIMIT"). Use the `relative_*` fields instead,
        // expressed in 1e-5 base-price units derived above.
        stop_loss: None,
        take_profit: None,
        comment,
        base_slippage_price: None,
        slippage_in_points: None,
        label: Some("neoethos-ui".to_string()),
        position_id: None,
        client_order_id: None,
        relative_stop_loss,
        relative_take_profit,
        guaranteed_stop_loss: None,
        trailing_stop_loss: None,
        stop_trigger_method: None,
    };

    let backend = ProductionCTraderExecutionBackend::default();
    let runtime_request = CTraderExecutionRuntimeRequest {
        client_id: creds.client_id,
        client_secret: creds.client_secret,
        access_token: creds.access_token,
        environment: creds.environment,
        account_id: creds.account_id_str,
        request: CTraderExecutionRequest::NewOrder(Box::new(new_order)),
    };
    backend.execute(&runtime_request)
}

/// Close an open position (full close — pass the position's own
/// volume). Used by the Trade Watch screen's per-row close button.
pub fn close_position_blocking(position_id: i64, volume: i64) -> Result<CTraderExecutionOutcome> {
    let creds = resolve_creds()?;
    let account_id: i64 = creds
        .account_id_str
        .parse()
        .map_err(|_| anyhow!("account_id '{}' is not numeric", creds.account_id_str))?;
    let runtime_request = CTraderExecutionRuntimeRequest {
        client_id: creds.client_id,
        client_secret: creds.client_secret,
        access_token: creds.access_token,
        environment: creds.environment,
        account_id: creds.account_id_str,
        request: CTraderExecutionRequest::ClosePosition(CTraderClosePositionRequest {
            account_id,
            position_id,
            volume,
        }),
    };
    ProductionCTraderExecutionBackend::default().execute(&runtime_request)
}

/// Cancel a pending order (not a filled position — use
/// `close_position_blocking` for that).
pub fn cancel_order_blocking(order_id: i64) -> Result<CTraderExecutionOutcome> {
    let creds = resolve_creds()?;
    let account_id: i64 = creds
        .account_id_str
        .parse()
        .map_err(|_| anyhow!("account_id '{}' is not numeric", creds.account_id_str))?;
    let runtime_request = CTraderExecutionRuntimeRequest {
        client_id: creds.client_id,
        client_secret: creds.client_secret,
        access_token: creds.access_token,
        environment: creds.environment,
        account_id: creds.account_id_str,
        request: CTraderExecutionRequest::CancelOrder(CTraderCancelOrderRequest {
            account_id,
            order_id,
        }),
    };
    ProductionCTraderExecutionBackend::default().execute(&runtime_request)
}

fn bars_to_normalized(bars: &[HistoricalBar]) -> Vec<NormalizedBar> {
    bars.iter()
        .map(|b| NormalizedBar {
            // The vortex writer stores nanosecond timestamps. cTrader
            // gives us milliseconds, multiply once here so downstream
            // chart loads don't get confused about units.
            timestamp_ns: b.timestamp_ms.saturating_mul(1_000_000),
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
            volume: b.volume.unwrap_or(0) as f64,
        })
        .collect()
}
