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

use anyhow::{Context, Result, anyhow};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::app_services::broker_config::CTraderBrokerEnvironment;
use crate::app_services::broker_deal_economics::{
    BrokerSymbolVolumeScaleEvidenceV1, MAX_EXACT_BROKER_VOLUME, broker_lots_from_wire_volume_v1,
};
use crate::app_services::broker_persistence::load_broker_settings;
use crate::app_services::ctrader_account::{
    CTraderAccountRuntimeRequest, CTraderAccountRuntimeSnapshot, CTraderCashFlowBundle,
    CTraderCtidProfileSnapshot, CTraderDealListBundle, CTraderExpectedMarginBundle,
    CTraderOrderHistoryBundle, CTraderServerVersionSnapshot, ensure_success_payload_type,
    load_account_runtime, parse_cash_flow_history_response, parse_ctid_profile_response,
    parse_deal_list_bundle_response, parse_expected_margin_response, parse_order_list_response,
    parse_reconcile_response, parse_trader_response, parse_version_response,
    reconcile_broker_unrealized_pnl,
};
use crate::app_services::ctrader_auth::CTraderTokenBundle;
use crate::app_services::ctrader_data::{
    CTraderChartHistoryRequest, CTraderHistoricalBarsFetchResult, CTraderLightSymbolInfo,
    CTraderResolvedSymbol, CTraderSymbolLookupRequest, CTraderSymbolsListResult, HistoricalBar,
    load_historical_bars_only, parse_asset_class_list_response, parse_symbol_by_id_response,
    parse_symbol_category_list_response, parse_symbols_list_response, resolve_symbol,
};
use crate::app_services::ctrader_execution::{
    CTraderExecutionBackend, CTraderExecutionOutcome, CTraderExecutionRequest,
    CTraderExecutionRuntimeRequest, ProductionCTraderExecutionBackend,
};
use crate::app_services::ctrader_live_auth::{
    CTraderEnvironment, CTraderLiveAuthBackend, CTraderTokenRefreshRequest,
    ProductionCTraderLiveAuthBackend,
};
use crate::app_services::ctrader_messages::{
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_CASH_FLOW_HISTORY_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_EXPECTED_MARGIN_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_CTID_PROFILE_BY_TOKEN_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_MARGIN_CALL_LIST_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ORDER_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_VERSION_RESPONSE_PAYLOAD_TYPE,
    CTraderDealListRequest, CTraderMarginCallListSnapshot, CTraderOpenApiTransport,
    ProductionCTraderOpenApiTransport, build_account_auth_request, build_application_auth_request,
    build_asset_class_list_request, build_cash_flow_history_list_request, build_deal_list_request,
    build_expected_margin_request, build_get_ctid_profile_by_token_request,
    build_get_position_unrealized_pnl_request, build_margin_call_list_request,
    build_order_list_request, build_reconcile_request, build_symbol_by_id_request,
    build_symbol_category_list_request, build_symbols_list_request, build_trader_request,
    build_version_request, parse_get_position_unrealized_pnl_response,
    parse_margin_call_list_response,
};
use crate::app_services::ctrader_messages::{
    CTraderAmendOrderRequest, CTraderAmendPositionSltpRequest, CTraderCancelOrderRequest,
    CTraderClosePositionRequest, CTraderNewOrderRequest, CTraderOrderType, CTraderTimeInForce,
    CTraderTradeSide,
};
use crate::app_services::secure_store::production_ctrader_token_store;

/// cTrader's bounded recent-trendbar response ceiling. Callers that need more
/// history must use the paged canonical-history capture path instead of
/// pretending a one-page live snapshot is complete.
pub(crate) const CTRADER_RECENT_TRENDBAR_LIMIT: usize = 5_000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RecentTrendbarPurpose {
    DisplayOnly,
    CanonicalEvidence,
}

impl RecentTrendbarPurpose {
    fn validate_pagination(self, has_more: bool) -> Result<()> {
        anyhow::ensure!(
            !has_more || self == Self::DisplayOnly,
            "cTrader reported hasMore; an incomplete page cannot become a canonical live-feature snapshot"
        );
        Ok(())
    }
}

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
    /// Unix-millis of the oldest bar the broker actually returned across all
    /// chunks (None when 0 bars came back). Lets the UI show real depth.
    pub oldest_ms: Option<i64>,
    pub dataset_identity: String,
    pub generation: String,
    pub durable_commit_id: String,
}

/// One exact, in-memory cTrader trendbar response together with the broker
/// identity and request bounds needed to publish it through the canonical
/// Vortex boundary. Unlike a bare `Vec<HistoricalBar>`, this cannot silently
/// lose the environment/account/symbol/timeframe authority of its source.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecentBrokerTrendbarSnapshot {
    pub(crate) identity: neoethos_data::CanonicalDatasetIdentity,
    pub(crate) requested_from_ms: i64,
    pub(crate) requested_to_ms: i64,
    pub(crate) retrieved_unix_ms: u64,
    pub(crate) bars: Vec<HistoricalBar>,
}

impl RecentBrokerTrendbarSnapshot {
    pub(crate) fn identity(&self) -> &neoethos_data::CanonicalDatasetIdentity {
        &self.identity
    }

    pub(crate) fn bars(&self) -> &[HistoricalBar] {
        &self.bars
    }

    fn into_bars(self) -> Vec<HistoricalBar> {
        self.bars
    }

    pub(crate) fn validate(&self) -> Result<()> {
        self.validate_with_empty_policy(false)
    }

    fn validate_with_empty_policy(&self, allow_empty: bool) -> Result<()> {
        anyhow::ensure!(
            self.identity.is_broker_real(),
            "recent trendbar snapshot is not bound to a broker identity"
        );
        anyhow::ensure!(
            self.identity.bar_timestamp_convention()
                == neoethos_data::BarTimestampConvention::BarOpen,
            "recent trendbar snapshot does not use canonical bar-open timestamps"
        );
        anyhow::ensure!(
            self.requested_from_ms < self.requested_to_ms,
            "recent trendbar snapshot request range is empty or descending"
        );
        if self.bars.is_empty() {
            anyhow::ensure!(
                allow_empty,
                "recent trendbar snapshot contains zero broker rows"
            );
            return Ok(());
        }
        validate_broker_bar_order(&self.bars, "recent cTrader trendbar snapshot")?;
        for (row, bar) in self.bars.iter().enumerate() {
            anyhow::ensure!(
                bar.timestamp_ms >= self.requested_from_ms
                    && bar.timestamp_ms < self.requested_to_ms,
                "recent trendbar row {row} timestamp {} is outside request [{}, {})",
                bar.timestamp_ms,
                self.requested_from_ms,
                self.requested_to_ms
            );
            anyhow::ensure!(
                [bar.open, bar.high, bar.low, bar.close]
                    .into_iter()
                    .all(f64::is_finite),
                "recent trendbar row {row} contains a non-finite OHLC price"
            );
            anyhow::ensure!(
                bar.high >= bar.open
                    && bar.high >= bar.close
                    && bar.high >= bar.low
                    && bar.low <= bar.open
                    && bar.low <= bar.close,
                "recent trendbar row {row} has incoherent OHLC bounds"
            );
            anyhow::ensure!(
                bar.volume.is_none_or(|volume| volume >= 0),
                "recent trendbar row {row} has negative tick volume"
            );
        }
        Ok(())
    }
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

/// Refresh the access token when it is within this many seconds of expiry
/// (or already expired). cTrader access tokens live ~30 min; 120 s of slack
/// means a call never goes out on a token about to die mid-request.
const TOKEN_REFRESH_WINDOW_SECS: i64 = 120;

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Load the stored token bundle and **silently refresh** it via the
/// `refresh_token` grant when it is expired or about to expire — NO browser,
/// NO user interaction. The refreshed bundle is persisted back to the keyring.
///
/// This is what makes the broker connection automatic: the interactive OAuth
/// (`run_reauth_flow_blocking`) is only ever needed ONCE to mint the first
/// refresh_token, or again if the broker revokes the refresh_token. Every
/// normal launch and every API call after that auto-refreshes here.
///
/// Blocking (does a token-endpoint HTTP POST when refreshing); callers already
/// run broker work inside `spawn_blocking`.
/// `pub(crate)`: the spot-streamer's reconnect loop also needs a fresh
/// token before every (re)connect — its spawn-time token dies after ~30
/// minutes and a reconnect with a stale token would fail auth forever.
pub(crate) fn ensure_fresh_token_bundle(
    client_id: &str,
    client_secret: &str,
) -> Result<CTraderTokenBundle> {
    let store = production_ctrader_token_store();
    let bundle = store
        .load_token_bundle_with_legacy_fallback()
        .map_err(|e| anyhow!("token bundle load failed: {e}"))?
        .ok_or_else(|| {
            anyhow!(
                "no cTrader token bundle saved yet — run Re-authenticate \
                 in Broker Setup once (only needed the first time)"
            )
        })?;

    if !bundle.needs_refresh_at(now_unix(), TOKEN_REFRESH_WINDOW_SECS) {
        return Ok(bundle);
    }
    if bundle.refresh_token.trim().is_empty() {
        // No refresh_token to spend — return the (stale) bundle; the call may
        // 401 and the operator will be prompted to re-authenticate once.
        tracing::warn!(
            target: "neoethos_app::auth",
            "token expired and no refresh_token present — re-authentication required"
        );
        return Ok(bundle);
    }

    let req = CTraderTokenRefreshRequest {
        client_id: client_id.to_string(),
        client_secret: client_secret.to_string(),
        refresh_token: bundle.refresh_token.clone(),
        scope: if bundle.scope.trim().is_empty() {
            "trading".to_string()
        } else {
            bundle.scope.clone()
        },
    };
    match ProductionCTraderLiveAuthBackend.refresh_token_bundle(&req) {
        Ok(fresh) => {
            if let Err(e) = store.save_token_bundle(&fresh) {
                tracing::warn!(
                    target: "neoethos_app::auth",
                    error = %e,
                    "access token refreshed but failed to persist; using in-memory copy"
                );
            } else {
                tracing::info!(
                    target: "neoethos_app::auth",
                    "cTrader access token silently refreshed (no re-auth needed)"
                );
            }
            Ok(fresh)
        }
        Err(e) => {
            tracing::warn!(
                target: "neoethos_app::auth",
                error = %e,
                "silent token refresh failed; falling back to stored token \
                 (re-authentication may be required if the refresh_token was revoked)"
            );
            Ok(bundle)
        }
    }
}

/// The error a caller sees when the cTrader environment on disk no longer
/// matches the one it was admitted against. Shared by every refusal so the
/// operator sees ONE recognisable message in the log.
fn environment_changed_error(expected_is_live: bool, now_is_live: bool) -> anyhow::Error {
    anyhow!(
        "REFUSING to contact the broker: the cTrader environment changed from {} to {} \
         since this engine was admitted by the demo forward-test gate. No order is sent. \
         Restart the engine to re-run the gate against the environment now selected.",
        if expected_is_live { "Live" } else { "Demo" },
        if now_is_live { "Live" } else { "Demo" },
    )
}

/// Resolve the cTrader credentials, optionally refusing unless the environment
/// on disk still matches the one the caller was admitted against.
///
/// **2026-08-09, and the reason this parameter exists.** `live_trading::run`
/// samples the environment once at the top of each bar iteration. The first
/// repair added a separate `assert_environment()` call immediately before the
/// broker call — better, but still two reads of `broker_credentials.toml` with a
/// gap between them, so a flip landing inside that gap was checked against the
/// OLD file and routed with the NEW one. That is a narrowing, not a guarantee,
/// and the previous version of this comment said so.
///
/// It is closed here instead: the comparison happens against `settings` — the
/// SAME read that produces `environment` on the returned [`ResolvedCreds`]. The
/// environment that is checked is by construction the environment the request is
/// sent to; there is no window between them at all.
///
/// `expected_is_live: None` means "no admission decision to honour" — the
/// read-only account/chart/history calls and the operator's own manual orders,
/// which are deliberately not gated (the operator ruled that manual trading
/// respects the operator).
fn resolve_creds_expecting(expected_is_live: Option<bool>) -> Result<ResolvedCreds> {
    let settings = load_broker_settings();
    let ct = &settings.ctrader;
    if let Some(expected) = expected_is_live {
        let now_is_live = matches!(ct.environment, CTraderBrokerEnvironment::Live);
        if now_is_live != expected {
            return Err(environment_changed_error(expected, now_is_live));
        }
    }
    if ct.client_id.is_empty() || ct.client_secret.is_empty() {
        return Err(anyhow!(
            "cTrader client_id / client_secret are empty in \
             broker_credentials.toml. Save them in Broker Setup, then run Re-authenticate."
        ));
    }
    // Prefer the account explicitly marked for execution (mirrors the spot
    // streamer's selection). Falling back to `.first()` blindly is how a stale
    // non-granted account id (e.g. a Live id while the token only grants Demo
    // accounts) ended up routing every request to CANT_ROUTE_REQUEST.
    let account = ct
        .accounts
        .iter()
        .find(|a| a.enabled_for_execution)
        .or_else(|| ct.accounts.first())
        .ok_or_else(|| anyhow!("no cTrader account configured"))?;

    let bundle = ensure_fresh_token_bundle(&ct.client_id, &ct.client_secret)?;

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

/// Resolve credentials with no admission decision to honour — read-only broker
/// calls (account runtime, chart bars, symbol catalogue, history) and the
/// operator's own manual orders.
fn resolve_creds() -> Result<ResolvedCreds> {
    resolve_creds_expecting(None)
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

    let bundle = ensure_fresh_token_bundle(&ct.client_id, &ct.client_secret)?;

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
    // Resilient: retry transient cold-connection failures and surface the
    // real cTrader error instead of a misleading "received N" count. Needs
    // the first 3 responses (app-auth, account-auth, symbols); asset-class +
    // symbol-category are best-effort classification on top.
    let responses = crate::app_services::ctrader_messages::send_sequence_resilient(
        &transport,
        &[
            build_application_auth_request(&creds.client_id, &creds.client_secret, "app-auth-1"),
            build_account_auth_request(account_id, &creds.access_token, "account-auth-1"),
            build_symbols_list_request(account_id, false, "symbols-1"),
            build_asset_class_list_request(account_id, "asset-classes-1"),
            build_symbol_category_list_request(account_id, "symbol-categories-1"),
        ],
        3,
        "cTrader symbols list",
    )?;

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

/// Download and publish one exact cTrader historical generation through the
/// shared model-free broker-history service. The shared service owns the
/// persistent authenticated socket, bounded one-page spool, exact identity/CAS
/// checks, cancellation gate and receipt derived directly from publication.
pub fn download_history_blocking(
    symbol: &str,
    timeframe: &str,
    from_ms: i64,
    to_ms: i64,
    data_root: &std::path::Path,
    dataset_selection: Option<&neoethos_data::SelectedDatasetGenerationV1>,
    active_fetch: &neoethos_broker_history::ProcessHistoricalCapture,
) -> Result<HistoricalDownloadOutcome> {
    let timeframe = parse_canonical_timeframe(timeframe)?;
    let target = dataset_selection.map_or(
        neoethos_broker_history::HistoricalCaptureTarget::NewIdentity,
        |selected| {
            neoethos_broker_history::HistoricalCaptureTarget::SelectedGeneration(selected.clone())
        },
    );
    let request = neoethos_broker_history::HistoricalCaptureRequest {
        symbol: symbol.to_owned(),
        timeframe,
        from_ms,
        to_ms,
        data_root: data_root.to_path_buf(),
        target,
    };
    let credentials = neoethos_broker_history::load_production_historical_credentials()?;
    let outcome =
        neoethos_broker_history::capture_historical_generation(request, credentials, active_fetch)?;
    let dataset_identity = outcome.selected_generation.identity().to_path_component();
    let generation = outcome.selected_generation.generation_id().to_owned();

    Ok(HistoricalDownloadOutcome {
        symbol: outcome.symbol,
        timeframe: outcome.timeframe.to_string(),
        bar_count: outcome.bar_count,
        has_more: false,
        written_path: outcome.written_path,
        oldest_ms: Some(outcome.oldest_ms),
        dataset_identity,
        generation,
        durable_commit_id: outcome.durable_commit_id,
    })
}
/// Fetch the most recent `limit` OHLCV bars for `symbol`/`timeframe`
/// straight from the cTrader broker (`ProtoOAGetTrendbarsReq`) with NO
/// disk write — the chart's broker-passthrough path (the authoritative,
/// *current* source). Returns bars sorted oldest→newest, trimmed to the
/// trailing `limit`. This one-page chart call owns one authenticated socket;
/// the full download path instead keeps one such socket across every page.
/// Callers must run either synchronous path on a blocking task.
pub fn fetch_recent_chart_bars_blocking(
    symbol: &str,
    timeframe: &str,
    limit: usize,
) -> Result<Vec<HistoricalBar>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let as_of_ms = current_unix_time_ms_i64()?;
    Ok(fetch_recent_broker_trendbar_snapshot_at_inner(
        symbol,
        timeframe,
        limit,
        as_of_ms,
        RecentTrendbarPurpose::DisplayOnly,
    )?
    .into_bars())
}

/// Fetch one recent broker snapshot and retain its exact cTrader identity and
/// request bounds so live feature construction can publish verified canonical
/// artifacts instead of fabricating an empty authority map.
pub(crate) fn fetch_recent_broker_trendbar_snapshot_blocking(
    symbol: &str,
    timeframe: &str,
    limit: usize,
) -> Result<RecentBrokerTrendbarSnapshot> {
    let as_of_ms = current_unix_time_ms_i64()?;
    fetch_recent_broker_trendbar_snapshot_at_blocking(symbol, timeframe, limit, as_of_ms)
}

/// Same operation with a caller-owned inclusive cTrader `toTimestamp`. The
/// parity harness uses one shared value for its long and short requests and
/// then proves that the short response is the exact tail of the long response.
pub(crate) fn fetch_recent_broker_trendbar_snapshot_at_blocking(
    symbol: &str,
    timeframe: &str,
    limit: usize,
    as_of_ms: i64,
) -> Result<RecentBrokerTrendbarSnapshot> {
    fetch_recent_broker_trendbar_snapshot_at_inner(
        symbol,
        timeframe,
        limit,
        as_of_ms,
        RecentTrendbarPurpose::CanonicalEvidence,
    )
}

fn fetch_recent_broker_trendbar_snapshot_at_inner(
    symbol: &str,
    timeframe: &str,
    limit: usize,
    as_of_ms: i64,
    purpose: RecentTrendbarPurpose,
) -> Result<RecentBrokerTrendbarSnapshot> {
    anyhow::ensure!(limit > 0, "recent cTrader trendbar limit must be positive");
    anyhow::ensure!(
        limit <= CTRADER_RECENT_TRENDBAR_LIMIT,
        "recent cTrader trendbar limit {limit} exceeds the one-page ceiling {CTRADER_RECENT_TRENDBAR_LIMIT}; use paged canonical history capture"
    );
    anyhow::ensure!(as_of_ms >= 0, "recent cTrader as-of timestamp is negative");
    let creds = resolve_creds()?;
    let from_ms = chart_fetch_from_ms(timeframe, as_of_ms, limit)?;
    // Window wide enough to contain `limit` bars with generous headroom
    // for weekends / holidays / illiquid gaps (markets aren't open 24/7,
    // so a tight window would starve the requested count). cTrader caps a
    // single response at ~5000 bars and `count` bounds the result, so one
    // request covers a chart (limit ≤ MAX_LIMIT = 2000).
    let request = CTraderChartHistoryRequest {
        client_id: creds.client_id.clone(),
        client_secret: creds.client_secret.clone(),
        access_token: creds.access_token.clone(),
        environment: creds.environment,
        account_id: creds.account_id_str.clone(),
        symbol_name: symbol.to_string(),
        timeframe: timeframe.to_string(),
        from_timestamp_ms: from_ms,
        to_timestamp_ms: as_of_ms,
        count: Some(u32::try_from(limit).context("recent cTrader trendbar limit exceeds u32")?),
    };
    let CTraderHistoricalBarsFetchResult {
        symbol: resolved_symbol,
        symbol_id,
        timeframe: resolved_timeframe,
        mut bars,
        has_more,
    } = load_historical_bars_only(&request)?;
    // `count` deliberately bounds a display page. hasMore is pagination, not
    // corrupt OHLC data. Only the display caller may accept that page; the
    // canonical-feature callers still require complete response evidence.
    purpose.validate_pagination(has_more)?;
    validate_broker_bar_order(&bars, "recent cTrader chart response")?;
    // The broker may return a few more than requested — keep trailing N.
    if bars.len() > limit {
        bars.drain(0..bars.len() - limit);
    }
    let requested_timeframe = parse_canonical_timeframe(timeframe)?;
    anyhow::ensure!(
        resolved_timeframe == requested_timeframe,
        "recent cTrader trendbar response timeframe {resolved_timeframe} disagrees with request {requested_timeframe}"
    );
    let dataset_environment = match creds.environment {
        CTraderEnvironment::Live => neoethos_data::CTraderEnvironment::Live,
        CTraderEnvironment::Demo => neoethos_data::CTraderEnvironment::Demo,
    };
    let account_id = creds
        .account_id_str
        .parse::<i64>()
        .context("configured cTrader account id is not a signed 64-bit integer")?;
    let identity = neoethos_data::CanonicalDatasetIdentity::ctrader(
        dataset_environment,
        creds.environment.endpoint_host(),
        account_id,
        symbol_id,
        resolved_symbol.symbol_name,
        resolved_timeframe,
        neoethos_data::BarTimestampConvention::BarOpen,
    )
    .map_err(|error| anyhow!(error.to_string()))?;
    let requested_to_ms = as_of_ms
        .checked_add(1)
        .context("recent cTrader half-open upper bound overflows i64")?;
    let snapshot = RecentBrokerTrendbarSnapshot {
        identity,
        requested_from_ms: from_ms,
        requested_to_ms,
        retrieved_unix_ms: u64::try_from(current_unix_time_ms_i64()?)
            .context("recent cTrader retrieval timestamp is negative")?,
        bars,
    };
    snapshot.validate_with_empty_policy(purpose == RecentTrendbarPurpose::DisplayOnly)?;
    Ok(snapshot)
}

fn current_unix_time_ms_i64() -> Result<i64> {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_millis();
    i64::try_from(millis).context("Unix timestamp milliseconds exceed i64")
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
    // Broker toTimestamp is inclusive; the UI cursor is exclusive. Reuse
    // the same bounded fetch and OHLC/identity checks as the initial page.
    Ok(fetch_recent_broker_trendbar_snapshot_at_inner(
        symbol,
        timeframe,
        limit,
        before_ms - 1,
        RecentTrendbarPurpose::DisplayOnly,
    )?
    .into_bars())
}

/// Duration of a single bar for the canonical timeframe, in ms. Used to
/// size the broker fetch window in [`fetch_recent_chart_bars_blocking`]
/// and [`fetch_chart_bars_before_blocking`].
fn chart_fetch_from_ms(tf: &str, to_ms: i64, limit: usize) -> Result<i64> {
    let timeframe = parse_canonical_timeframe(tf)?;
    let Some(step_ms) = timeframe.fixed_duration_ms() else {
        // The official `count` field asks for the trailing N bars back from
        // `toTimestamp`. An epoch lower bound avoids inventing fixed D1/W1/MN1
        // durations while `count` still bounds the response.
        return Ok(0);
    };
    let limit = i64::try_from(limit).context("chart bar limit exceeds i64")?;
    let span_ms = step_ms
        .checked_mul(limit)
        .and_then(|value| value.checked_mul(3))
        .context("chart fetch window overflows i64 milliseconds")?;
    Ok(to_ms.saturating_sub(span_ms.max(step_ms)).max(0))
}

fn parse_canonical_timeframe(tf: &str) -> Result<neoethos_core::CanonicalTimeframe> {
    tf.trim()
        .to_ascii_uppercase()
        .parse()
        .map_err(|_| anyhow!("unsupported cTrader timeframe {tf:?}"))
}

fn validate_broker_bar_order(bars: &[HistoricalBar], context: &str) -> Result<()> {
    for (row, pair) in bars.windows(2).enumerate() {
        if pair[1].timestamp_ms <= pair[0].timestamp_ms {
            return Err(anyhow!(
                "{context} is not strictly increasing at rows {row}/{}: {} -> {}; \
                 refusing sort/dedup repair",
                row + 1,
                pair[0].timestamp_ms,
                pair[1].timestamp_ms
            ));
        }
    }
    Ok(())
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
/// Everything a new-order (market OR pending) submission needs after the
/// shared, money-critical prep: resolved account/symbol ids, the lots→wire
/// volume (bounds-checked against the broker's min/max/step) and tick-aligned
/// relative SL/TP. Extracted so the market and pending paths compute volume +
/// SL/TP precision through ONE code path — a bug fixed here is fixed for both.
struct PreparedNewOrder {
    creds: ResolvedCreds,
    account_id: i64,
    symbol_id: i64,
    volume_units: i64,
    volume_scale_evidence: BrokerSymbolVolumeScaleEvidenceV1,
    relative_stop_loss: Option<i64>,
    relative_take_profit: Option<i64>,
    broker_allows_new_positions: bool,
    broker_allows_short_selling: bool,
}

fn ensure_broker_allows_new_order(
    symbol: &str,
    side: OrderSide,
    broker_allows_new_positions: bool,
    broker_allows_short_selling: bool,
) -> Result<()> {
    if !broker_allows_new_positions {
        return Err(anyhow!(
            "broker full-symbol metadata does not report tradingMode=ENABLED for {symbol}; refusing to open a new position"
        ));
    }
    if side == OrderSide::Sell && !broker_allows_short_selling {
        return Err(anyhow!(
            "broker full-symbol metadata forbids short selling for {symbol}; refusing the sell-to-open order"
        ));
    }
    Ok(())
}

fn wire_volume_from_broker_lot_size(volume_lots: f64, lot_size_cents: i64) -> Result<i64> {
    if !volume_lots.is_finite() || volume_lots <= 0.0 {
        return Err(anyhow!(
            "volume_lots must be finite and positive (got {volume_lots})"
        ));
    }
    if lot_size_cents <= 0 {
        return Err(anyhow!(
            "broker lotSize must be positive (got {lot_size_cents})"
        ));
    }
    let raw = volume_lots * lot_size_cents as f64;
    let rounded = raw.round();
    let tolerance = raw.abs().max(1.0) * f64::EPSILON * 8.0;
    if !rounded.is_finite()
        || rounded <= 0.0
        || rounded >= i64::MAX as f64
        || (raw - rounded).abs() > tolerance
    {
        return Err(anyhow!(
            "volume {volume_lots} lots is not exactly representable by broker lotSize={lot_size_cents} or exceeds the supported range"
        ));
    }
    Ok(rounded as i64)
}

fn relative_distance_from_broker_symbol(pips: f64, digits: i32, pip_position: i32) -> Result<i64> {
    if !pips.is_finite() || pips <= 0.0 {
        return Err(anyhow!(
            "pip distance must be finite and positive (got {pips})"
        ));
    }
    if !(0..=5).contains(&digits) {
        return Err(anyhow!(
            "broker symbol digits={digits} cannot be represented by cTrader relative-distance units"
        ));
    }
    if !(0..=digits).contains(&pip_position) {
        return Err(anyhow!(
            "broker symbol pipPosition={pip_position} is inconsistent with digits={digits}"
        ));
    }

    // ProtoOASymbol.pipPosition defines one pip as 10^-pipPosition price
    // units. cTrader relativeStopLoss/relativeTakeProfit use 1/100000 price
    // units, while `digits` defines the symbol's minimum price tick.
    let raw_relative_units = pips * 10.0_f64.powi(5 - pip_position);
    let tick_units = 10_i64.pow((5 - digits) as u32);
    let tick_count = (raw_relative_units / tick_units as f64).round();
    let snapped = tick_count * tick_units as f64;
    let tolerance = raw_relative_units.abs().max(1.0) * f64::EPSILON * 8.0;
    if !snapped.is_finite()
        || snapped <= 0.0
        || snapped >= i64::MAX as f64
        || (snapped - raw_relative_units).abs() > tolerance
    {
        return Err(anyhow!(
            "pip distance {pips} is not exactly aligned to broker digits={digits}, pipPosition={pip_position}"
        ));
    }
    Ok(snapped as i64)
}

// Preserve the existing entry/resting-amend halt; close/reduce routes do not use it.
fn ensure_entry_margin_halt_clear() -> Result<()> {
    crate::app_services::margin_call::ensure_spawned();
    if let Some(halt) = crate::app_services::margin_call::active_halt() {
        return Err(anyhow!(
            "REFUSING to open a position: {}. No order is sent. Reduce exposure \
             (closing positions and amending stops are still allowed) and restart \
             the backend to clear the halt once the account is healthy.",
            halt.describe()
        ));
    }
    Ok(())
}

fn validate_order_volume_and_brackets(
    volume_lots: Option<f64>,
    stop_loss_pips: Option<f64>,
    take_profit_pips: Option<f64>,
) -> Result<()> {
    if let Some(volume_lots) = volume_lots {
        if !(volume_lots.is_finite() && volume_lots > 0.0) {
            return Err(anyhow!(
                "volume_lots must be a finite positive number (got {volume_lots})"
            ));
        }
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
    Ok(())
}

/// Validate inputs, resolve the symbol, convert lots→wire volume (bounds-checked)
/// and derive tick-aligned relative SL/TP exclusively from the resolved
/// `ProtoOASymbol` contract.
fn prepare_new_order(
    symbol: &str,
    volume_lots: f64,
    stop_loss_pips: Option<f64>,
    take_profit_pips: Option<f64>,
    expected_is_live: Option<bool>,
    expected_account_id: Option<i64>,
) -> Result<PreparedNewOrder> {
    // ── #238: the margin-call halt, applied at the single choke point ──────
    // Both order-OPENING paths (`submit_market_order_blocking` and
    // `submit_pending_order_blocking`) pass through here, so one check covers
    // the autopilot, the manual route and the MCP sidecar alike.
    //
    // BEHAVIOUR CHANGE, stated plainly: while the broker says this account is
    // in (or at) margin call, NO new position may be opened by ANY route.
    // Closing a position, cancelling a resting order and amending an existing
    // SL/TP are all deliberately still permitted — those reduce exposure, and
    // taking them away during a margin call would be the opposite of safe.
    //
    // This does NOT contradict the operator's #190 ruling that the manual path
    // respects the operator. That ruling is about POLICY — sizing, brackets,
    // daily slots. A margin call is not a policy opinion; it is the broker
    // stating that it is about to liquidate. The halt is cleared by restarting
    // the backend (see `margin_call::clear_halt`), which is deliberate: it can
    // never leave the operator permanently locked out.
    // Stopgap start (see `margin_call::ensure_spawned`): the watchdog belongs
    // in `main.rs` next to the other background services, which is outside the
    // scope of this change. Starting it here guarantees that any process which
    // actually opens a position is being watched. Idempotent and cheap — one
    // relaxed atomic swap after the first call.
    ensure_entry_margin_halt_clear()?;

    validate_order_volume_and_brackets(Some(volume_lots), stop_loss_pips, take_profit_pips)?;
    // THE LAST READ BEFORE REAL MONEY. `creds.environment` — the environment
    // this order is actually sent to — comes out of the same
    // `broker_credentials.toml` read that validates `expected_is_live`, so a
    // Demo→Live flip cannot land between the check and the send.
    let creds = resolve_creds_expecting(expected_is_live)?;
    prepare_new_order_from_resolved_creds(
        creds,
        symbol,
        volume_lots,
        stop_loss_pips,
        take_profit_pips,
        expected_account_id,
        resolve_symbol,
    )
}

// One captured credential/account identity for both entry and no-volume amendments.
fn resolve_order_symbol_from_creds(
    creds: &ResolvedCreds,
    symbol: &str,
    expected_account_id: Option<i64>,
    resolve: impl FnOnce(&CTraderSymbolLookupRequest) -> Result<CTraderResolvedSymbol>,
) -> Result<CTraderResolvedSymbol> {
    let account_id = creds
        .account_id_str
        .parse::<i64>()
        .context("configured cTrader account_id must be numeric")?;
    anyhow::ensure!(
        account_id > 0,
        "configured cTrader account_id must be positive; no order is sent"
    );
    if let Some(expected) = expected_account_id {
        anyhow::ensure!(
            expected > 0 && account_id == expected,
            "REFUSING to open a position: the resolved cTrader account does not match \
             the admitted account. No symbol lookup or order is sent."
        );
    }
    let resolved = resolve(&CTraderSymbolLookupRequest {
        client_id: creds.client_id.clone(),
        client_secret: creds.client_secret.clone(),
        access_token: creds.access_token.clone(),
        environment: creds.environment,
        account_id: creds.account_id_str.clone(),
        symbol_name: symbol.to_string(),
    })?;
    anyhow::ensure!(
        resolved.account_id == account_id,
        "REFUSING to prepare an order: resolved symbol metadata belongs to a different \
         cTrader account than the checked credentials. No order is sent."
    );
    // No-volume amendments skip volume-scale evidence, not its symbol identity
    // sanity checks. The resolver already binds the full/light/requested symbol.
    anyhow::ensure!(
        resolved.light_symbol.symbol_id > 0
            && !resolved.light_symbol.symbol_name.trim().is_empty()
            && resolved.light_symbol.symbol_name.trim() == resolved.light_symbol.symbol_name,
        "broker returned incomplete or invalid symbol identity; no order is sent"
    );
    Ok(resolved)
}

/// Continue from the one immutable credential snapshot used for authentication
/// and submission. Check the admitted account before the injected symbol lookup:
/// switching accounts inside the same environment must not send an order either.
fn prepare_new_order_from_resolved_creds(
    creds: ResolvedCreds,
    symbol: &str,
    volume_lots: f64,
    stop_loss_pips: Option<f64>,
    take_profit_pips: Option<f64>,
    expected_account_id: Option<i64>,
    resolve: impl FnOnce(&CTraderSymbolLookupRequest) -> Result<CTraderResolvedSymbol>,
) -> Result<PreparedNewOrder> {
    let resolved = resolve_order_symbol_from_creds(&creds, symbol, expected_account_id, resolve)?;

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
    // The conversion below consumes that exact broker field with checked
    // arithmetic. Missing/invalid lotSize, min/max, or stepVolume fails before
    // order submission; there is no built-in FX/XAU/CFD default.
    let financials = resolved.symbol.financials.as_ref().ok_or_else(|| {
        anyhow!(
            "broker omitted full financial/trading metadata for {symbol}; refusing to prepare an order"
        )
    })?;
    let broker_allows_new_positions = financials.can_open_new_position();
    let broker_allows_short_selling = financials.short_selling_allowed();
    let lot_size = resolved.symbol.lot_size.ok_or_else(|| {
        anyhow!(
            "broker omitted lotSize for {symbol}; refusing to fall back \
             to a synthetic 10,000,000-cents default (would be 1000× wrong \
             for XAU/XAG/index symbols). Re-fetch /broker/symbols or check \
             the cTrader symbol catalog endpoint."
        )
    })?;
    let canonical_environment = match creds.environment {
        CTraderEnvironment::Demo => "demo",
        CTraderEnvironment::Live => "live",
    };
    let volume_scale_evidence = BrokerSymbolVolumeScaleEvidenceV1::new(
        canonical_environment,
        resolved.account_id,
        resolved.light_symbol.symbol_id,
        resolved.light_symbol.symbol_name.clone(),
        lot_size,
    )?;
    let volume_units = wire_volume_from_broker_lot_size(volume_lots, lot_size)?;
    let min = resolved.symbol.min_volume.ok_or_else(|| {
        anyhow!("broker omitted minVolume for {symbol}; refusing to prepare an order")
    })?;
    let max = resolved.symbol.max_volume.ok_or_else(|| {
        anyhow!("broker omitted maxVolume for {symbol}; refusing to prepare an order")
    })?;
    let step = resolved.symbol.step_volume.ok_or_else(|| {
        anyhow!("broker omitted stepVolume for {symbol}; refusing to prepare an order")
    })?;
    for (field, value) in [("minVolume", min), ("maxVolume", max), ("stepVolume", step)] {
        anyhow::ensure!(
            value > 0,
            "broker returned invalid {field} {value} for {symbol}"
        );
    }
    anyhow::ensure!(
        min <= max,
        "broker returned inverted volume bounds minVolume={min}, maxVolume={max} for {symbol}"
    );
    if volume_units < min {
        return Err(anyhow!(
            "volume {volume_units} is below broker min_volume {min} for {symbol}"
        ));
    }
    if volume_units > max {
        return Err(anyhow!(
            "volume {volume_units} exceeds broker max_volume {max} for {symbol}"
        ));
    }
    if volume_units % step != 0 {
        return Err(anyhow!(
            "volume {volume_units} is not aligned to broker stepVolume {step} for {symbol}"
        ));
    }

    let (relative_stop_loss, relative_take_profit) =
        relative_brackets_from_broker_symbol(&resolved, stop_loss_pips, take_profit_pips)?;

    Ok(PreparedNewOrder {
        creds,
        account_id: resolved.account_id,
        symbol_id: resolved.light_symbol.symbol_id,
        volume_units,
        volume_scale_evidence,
        relative_stop_loss,
        relative_take_profit,
        broker_allows_new_positions,
        broker_allows_short_selling,
    })
}

fn relative_brackets_from_broker_symbol(
    resolved: &CTraderResolvedSymbol,
    stop_loss_pips: Option<f64>,
    take_profit_pips: Option<f64>,
) -> Result<(Option<i64>, Option<i64>)> {
    // cTrader `relativeStopLoss` / `relativeTakeProfit` is the price *distance*
    // expressed in 1/100000 of a price unit, and the broker REJECTS any value
    // that isn't aligned to the symbol's price precision (10^-digits) with
    // "Relative stop loss has invalid precision".
    //
    // `pipPosition` defines the pip itself and `digits` defines the tick. An
    // operator distance that cannot be represented exactly on that broker
    // grid is rejected rather than rounded to a different risk distance.
    let relative_stop_loss = stop_loss_pips
        .map(|pips| {
            relative_distance_from_broker_symbol(
                pips,
                resolved.symbol.digits,
                resolved.symbol.pip_position,
            )
        })
        .transpose()?;
    let relative_take_profit = take_profit_pips
        .map(|pips| {
            relative_distance_from_broker_symbol(
                pips,
                resolved.symbol.digits,
                resolved.symbol.pip_position,
            )
        })
        .transpose()?;
    Ok((relative_stop_loss, relative_take_profit))
}

///
/// `expected_is_live` is the broker environment the CALLER was admitted
/// against. `Some(false)`/`Some(true)` refuses the order outright if the
/// environment on disk has since changed; `None` means the caller has no
/// admission decision to honour (the operator's manual Buy/Sell button).
/// `expected_account_id` additionally pins the admitted account within that
/// environment, before symbol lookup, using the same credentials as submission.
#[allow(clippy::too_many_arguments)]
pub fn submit_market_order_blocking(
    symbol: &str,
    side: OrderSide,
    volume_lots: f64,
    stop_loss_pips: Option<f64>,
    take_profit_pips: Option<f64>,
    comment: Option<String>,
    expected_is_live: Option<bool>,
    expected_account_id: Option<i64>,
) -> Result<CTraderExecutionOutcome> {
    let prep = prepare_new_order(
        symbol,
        volume_lots,
        stop_loss_pips,
        take_profit_pips,
        expected_is_live,
        expected_account_id,
    )?;
    ensure_broker_allows_new_order(
        symbol,
        side,
        prep.broker_allows_new_positions,
        prep.broker_allows_short_selling,
    )?;

    execute_prepared_market_order(prep, side, comment, None, |request| {
        ProductionCTraderExecutionBackend::default().execute(request)
    })
}

// One wire construction/execute path for manual and context-bound entries.
fn execute_prepared_market_order(
    prep: PreparedNewOrder,
    side: OrderSide,
    comment: Option<String>,
    client_order_id: Option<String>,
    execute: impl FnOnce(&CTraderExecutionRuntimeRequest) -> Result<CTraderExecutionOutcome>,
) -> Result<CTraderExecutionOutcome> {
    let new_order = CTraderNewOrderRequest {
        account_id: prep.account_id,
        symbol_id: prep.symbol_id,
        order_type: CTraderOrderType::Market,
        trade_side: side.into(),
        volume: prep.volume_units,
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
        client_order_id,
        relative_stop_loss: prep.relative_stop_loss,
        relative_take_profit: prep.relative_take_profit,
        guaranteed_stop_loss: None,
        trailing_stop_loss: None,
        stop_trigger_method: None,
    };

    let runtime_request = CTraderExecutionRuntimeRequest {
        client_id: prep.creds.client_id,
        client_secret: prep.creds.client_secret,
        access_token: prep.creds.access_token,
        environment: prep.creds.environment,
        account_id: prep.creds.account_id_str,
        request: CTraderExecutionRequest::NewOrder(Box::new(new_order)),
    };
    let mut outcome = execute(&runtime_request)?;
    outcome.volume_scale_evidence = Some(prep.volume_scale_evidence);
    Ok(outcome)
}

/// Place a PENDING (conditional) order that the broker holds and fills only
/// when the market reaches `trigger_price`:
///   - `Limit` → fills at `trigger_price` or better (BUY below / SELL above market),
///   - `Stop`  → fills once price trades through `trigger_price` (breakout entries).
///
/// This is the user-facing "execute when the criteria are met" primitive. The
/// order lives broker-side (survives app restarts) so the fill happens even if
/// NeoEthos is closed. SL/TP are pip distances from the trigger, snapped to the
/// symbol's tick via the shared [`prepare_new_order`] path. `expiry_unix_ms`
/// switches the order to Good-Till-Date; otherwise it is Good-Till-Cancel.
///
/// The trigger price's relationship to the current market (limit-below /
/// stop-above etc.) is enforced by cTrader; an invalid side/price combination
/// is surfaced verbatim as the broker's rejection rather than guessed at here.
#[allow(clippy::too_many_arguments)]
pub fn submit_pending_order_blocking(
    symbol: &str,
    side: OrderSide,
    order_type: CTraderOrderType,
    volume_lots: f64,
    trigger_price: f64,
    stop_loss_pips: Option<f64>,
    take_profit_pips: Option<f64>,
    expiry_unix_ms: Option<i64>,
    comment: Option<String>,
    expected_is_live: Option<bool>,
) -> Result<CTraderExecutionOutcome> {
    if !matches!(order_type, CTraderOrderType::Limit | CTraderOrderType::Stop) {
        return Err(anyhow!(
            "pending order type must be Limit or Stop (got {})",
            order_type.label()
        ));
    }
    if !(trigger_price.is_finite() && trigger_price > 0.0) {
        return Err(anyhow!(
            "trigger_price must be a finite positive number (got {trigger_price})"
        ));
    }

    let prep = prepare_new_order(
        symbol,
        volume_lots,
        stop_loss_pips,
        take_profit_pips,
        expected_is_live,
        None,
    )?;
    ensure_broker_allows_new_order(
        symbol,
        side,
        prep.broker_allows_new_positions,
        prep.broker_allows_short_selling,
    )?;

    // Limit orders carry `limit_price`; stop orders carry `stop_price`.
    let (limit_price, stop_price) = match order_type {
        CTraderOrderType::Limit => (Some(trigger_price), None),
        CTraderOrderType::Stop => (None, Some(trigger_price)),
        // Unreachable: guarded above.
        _ => (None, None),
    };

    // Default GTC (rests until filled/cancelled); GTD when an expiry is given.
    let (time_in_force, expiration_timestamp_ms) = match expiry_unix_ms {
        Some(ms) if ms > 0 => (Some(CTraderTimeInForce::GoodTillDate), Some(ms)),
        _ => (Some(CTraderTimeInForce::GoodTillCancel), None),
    };

    let new_order = CTraderNewOrderRequest {
        account_id: prep.account_id,
        symbol_id: prep.symbol_id,
        order_type,
        trade_side: side.into(),
        volume: prep.volume_units,
        limit_price,
        stop_price,
        time_in_force,
        expiration_timestamp_ms,
        // For pending orders the broker accepts relative SL/TP (distance from the
        // order's entry), same encoding the market path uses — reuse it.
        stop_loss: None,
        take_profit: None,
        comment,
        base_slippage_price: None,
        slippage_in_points: None,
        label: Some("neoethos-ui".to_string()),
        position_id: None,
        client_order_id: None,
        relative_stop_loss: prep.relative_stop_loss,
        relative_take_profit: prep.relative_take_profit,
        guaranteed_stop_loss: None,
        trailing_stop_loss: None,
        stop_trigger_method: None,
    };

    let backend = ProductionCTraderExecutionBackend::default();
    let runtime_request = CTraderExecutionRuntimeRequest {
        client_id: prep.creds.client_id,
        client_secret: prep.creds.client_secret,
        access_token: prep.creds.access_token,
        environment: prep.creds.environment,
        account_id: prep.creds.account_id_str,
        request: CTraderExecutionRequest::NewOrder(Box::new(new_order)),
    };
    backend.execute(&runtime_request)
}

/// Close an open position (full close — pass the position's own
/// volume). Used by the Trade Watch screen's per-row close button.
///
/// `expected_is_live` — see [`submit_market_order_blocking`]. An engine's
/// weekend force-close and auto-cull flatten also pass `expected_account_id`.
/// Both expectations are checked against the same resolved credentials used
/// to send the request, including an account switch inside one environment.
/// The operator's manual Trade Watch button passes `None` for both.
pub fn close_position_blocking(
    position_id: i64,
    volume: i64,
    expected_is_live: Option<bool>,
    expected_account_id: Option<i64>,
) -> Result<CTraderExecutionOutcome> {
    let creds = resolve_creds_expecting(expected_is_live)?;
    let runtime_request =
        close_request_from_resolved_creds(creds, position_id, volume, expected_account_id)?;
    ProductionCTraderExecutionBackend::default().execute(&runtime_request)
}

/// Build and validate from one immutable credential snapshot. This pure step
/// cannot accidentally check one account and route with a second disk read.
fn close_request_from_resolved_creds(
    creds: ResolvedCreds,
    position_id: i64,
    volume: i64,
    expected_account_id: Option<i64>,
) -> Result<CTraderExecutionRuntimeRequest> {
    let account_id: i64 = creds
        .account_id_str
        .parse()
        .map_err(|_| anyhow!("account_id '{}' is not numeric", creds.account_id_str))?;
    if let Some(expected) = expected_account_id {
        anyhow::ensure!(
            expected > 0 && account_id == expected,
            "REFUSING to close position {position_id}: cTrader account changed from admitted account \
             {expected} to resolved account {account_id}. No close order is sent."
        );
    }
    Ok(CTraderExecutionRuntimeRequest {
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
    })
}

/// Load the live account runtime (balance, equity inputs, open positions,
/// pending orders) from cTrader for the active account. Resolves creds with
/// the automatic silent-refresh path, so a normal launch never needs re-auth.
/// Blocking; callers must wrap in `spawn_blocking`.
pub fn fetch_account_runtime_blocking() -> Result<CTraderAccountRuntimeSnapshot> {
    let creds = resolve_creds()?;
    let request = CTraderAccountRuntimeRequest {
        client_id: creds.client_id,
        client_secret: creds.client_secret,
        access_token: creds.access_token,
        environment: creds.environment,
        account_id: creds.account_id_str,
        return_protection_orders: true,
    };
    load_account_runtime(&request)
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

/// Modify a RESTING (pending) order — `ProtoOAAmendOrderReq` (2109).
///
/// **The capability the UI did not have (audit #236, wired 2026-08-10.)**
/// `build_amend_order_request` had existed since the message layer was written,
/// with no `CTraderExecutionRequest` variant to carry it and no caller, so the
/// Actions screen could place a resting order and cancel it but never change
/// it. Correcting a trigger price meant cancel + re-place: two broker round
/// trips, a different order id, and a window in which the level the operator was
/// waiting for has no order behind it at all.
///
/// **Every parameter is `Option`, and `None` means LEAVE UNCHANGED** — that is
/// the proto's own contract for the optional fields. At least one must be
/// `Some`, or there is nothing to amend and this refuses rather than sending a
/// no-op the broker would answer with a success.
///
/// `trigger_price` is written to `limit_price` or `stop_price` according to
/// `order_type`, which must be the order's OWN type: cTrader rejects a limit
/// price on a stop order. Volume is lots and SL/TP are pip DISTANCES, converted
/// by the shared placement helpers (volume only when explicitly set), so an amend
/// and a placement can never disagree about what "0.01 lots" or "20 pips" mean
/// on this symbol.
///
/// **Why it goes through the margin-call halt.** Amending a resting order is
/// NOT the exposure-reducing action that closing and stop-moving are: the order
/// can still fill, and this call can raise its volume. So it takes the same
/// choke point as placing one. During a halt the operator can still CANCEL the
/// order, which is the safe direction, and the refusal says so.
#[allow(clippy::too_many_arguments)]
pub fn amend_order_blocking(
    order_id: i64,
    symbol: &str,
    order_type: CTraderOrderType,
    volume_lots: Option<f64>,
    trigger_price: Option<f64>,
    stop_loss_pips: Option<f64>,
    take_profit_pips: Option<f64>,
    expiry_unix_ms: Option<i64>,
    expected_is_live: Option<bool>,
) -> Result<CTraderExecutionOutcome> {
    if volume_lots.is_none()
        && trigger_price.is_none()
        && stop_loss_pips.is_none()
        && take_profit_pips.is_none()
        && expiry_unix_ms.is_none()
    {
        return Err(anyhow!(
            "amend_order needs at least one of volumeLots / triggerPrice / stopLossPips / \
             takeProfitPips / expiryUnixMs — every field was omitted, so there is nothing to \
             change and no order was sent"
        ));
    }
    if !matches!(order_type, CTraderOrderType::Limit | CTraderOrderType::Stop) {
        return Err(anyhow!(
            "amend_order applies to RESTING limit/stop orders only (got {order_type:?}). A \
             filled position's stop and target are changed with amend_position_sltp"
        ));
    }
    if let Some(price) = trigger_price {
        // Deliberately NOT floored at zero: commodity prices can be negative
        // (WTI settled at -$37.63 on 2020-04-20) and XTIUSD is watchlisted. What
        // is refused is a value that is not a number.
        if !price.is_finite() {
            return Err(anyhow!(
                "triggerPrice must be a finite number (got {price})"
            ));
        }
    }
    if let Some(ms) = expiry_unix_ms {
        if ms <= 0 {
            return Err(anyhow!(
                "expiryUnixMs must be a positive epoch-millisecond timestamp when set (got {ms}). \
                 To make the order Good-Till-Cancel, omit it — this call cannot clear an expiry \
                 that is already set"
            ));
        }
    }

    // Keep the existing resting-order halt and the same one-snapshot environment
    // check. Only an explicitly amended volume requires entry-volume evidence.
    ensure_entry_margin_halt_clear()?;
    validate_order_volume_and_brackets(volume_lots, stop_loss_pips, take_profit_pips)?;
    let creds = resolve_creds_expecting(expected_is_live)?;
    let transport = ProductionCTraderOpenApiTransport::new(creds.environment.endpoint_host());
    let runtime_request = prepare_amend_order_from_resolved_creds(
        creds,
        order_id,
        symbol,
        order_type,
        volume_lots,
        trigger_price,
        stop_loss_pips,
        take_profit_pips,
        expiry_unix_ms,
        None,
        |request| {
            resolve_amend_order_symbol_with_transport(
                &transport,
                request,
                order_id,
                order_type,
                resolve_symbol,
            )
        },
    )?;
    ProductionCTraderExecutionBackend::default().execute(&runtime_request)
}

// Bind the requested amendment to the broker's own order before any pip or
// lot conversion. This reuses the existing details parser and symbol resolver;
// it does not create a second execution path or require historical fills.
fn resolve_amend_order_symbol_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderSymbolLookupRequest,
    order_id: i64,
    order_type: CTraderOrderType,
    resolve: impl FnOnce(&CTraderSymbolLookupRequest) -> Result<CTraderResolvedSymbol>,
) -> Result<CTraderResolvedSymbol> {
    let account_id = request
        .account_id
        .parse::<i64>()
        .context("cTrader amendment account id must be numeric")?;
    anyhow::ensure!(
        account_id > 0,
        "cTrader amendment account id must be positive"
    );
    anyhow::ensure!(order_id > 0, "cTrader amendment order id must be positive");
    anyhow::ensure!(
        matches!(order_type, CTraderOrderType::Limit | CTraderOrderType::Stop),
        "cTrader amendment requires a Limit or Stop order"
    );

    // All authentication and lookup inputs originate in the one credential
    // snapshot also carried by the eventual execution request.
    let responses = transport.send_sequence(&[
        build_application_auth_request(
            &request.client_id,
            &request.client_secret,
            "amend-details-app-auth",
        ),
        build_account_auth_request(
            account_id,
            &request.access_token,
            "amend-details-account-auth",
        ),
        crate::app_services::ctrader_messages::build_order_details_request(
            account_id,
            order_id,
            "amend-order-details",
        ),
    ])?;
    // A broker error may end the sequence before three responses. Preserve its
    // original code/description before reporting length or missing identity.
    for (response, expected_type) in responses.iter().zip([
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
        CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
        crate::app_services::ctrader_messages::CTRADER_OA_ORDER_DETAILS_RESPONSE_PAYLOAD_TYPE,
    ]) {
        ensure_success_payload_type(response, expected_type)?;
    }
    anyhow::ensure!(
        responses.len() == 3,
        "expected 3 cTrader amendment order-details responses, received {}",
        responses.len()
    );
    require_response_account(&responses[1], account_id, "amendment authentication")?;
    require_response_account(&responses[2], account_id, "amendment order details")?;
    let details =
        crate::app_services::ctrader_account::parse_order_details_response(&responses[2])?;
    anyhow::ensure!(
        details.account_id == account_id && details.order.order_id == order_id,
        "cTrader amendment order-details identity differs from requested account/order"
    );
    anyhow::ensure!(
        details.order.symbol_id > 0,
        "cTrader amendment order details contain an invalid symbol id"
    );
    anyhow::ensure!(
        details.order.order_type == order_type.label(),
        "cTrader amendment order type {} differs from requested {}",
        details.order.order_type,
        order_type.label()
    );
    // ProtoOAOrder.orderStatus is required. The reusable display snapshot does
    // not retain it, so check the original envelope, not a caller/default value.
    // Only ACCEPTED (1) is still pending; filled/cancelled/expired orders cannot
    // be amended. A later fill can still race this read: the broker remains the
    // authority at submission and its rejection is propagated unchanged.
    let envelope = crate::app_services::ctrader_messages::parse_open_api_envelope(&responses[2])?;
    anyhow::ensure!(
        envelope.payload["order"]
            .get("orderStatus")
            .and_then(serde_json::Value::as_i64)
            == Some(1),
        "cTrader amendment order is not pending or omitted a valid orderStatus"
    );

    let resolved = resolve(request)?;
    anyhow::ensure!(
        resolved.account_id == details.account_id
            && resolved.light_symbol.symbol_id == details.order.symbol_id
            && resolved.symbol.symbol_id == details.order.symbol_id,
        "cTrader amendment order symbol/account differs from requested symbol metadata"
    );
    Ok(resolved)
}

// Build the actual pending-order amendment after the wrapper's field/halt checks.
// Symbol lookup is injected for tests; submission stays in the existing backend.
#[allow(clippy::too_many_arguments)]
fn prepare_amend_order_from_resolved_creds(
    creds: ResolvedCreds,
    order_id: i64,
    symbol: &str,
    order_type: CTraderOrderType,
    volume_lots: Option<f64>,
    trigger_price: Option<f64>,
    stop_loss_pips: Option<f64>,
    take_profit_pips: Option<f64>,
    expiry_unix_ms: Option<i64>,
    expected_account_id: Option<i64>,
    resolve: impl FnOnce(&CTraderSymbolLookupRequest) -> Result<CTraderResolvedSymbol>,
) -> Result<CTraderExecutionRuntimeRequest> {
    let (creds, account_id, volume, relative_stop_loss, relative_take_profit) =
        if let Some(volume_lots) = volume_lots {
            let prep = prepare_new_order_from_resolved_creds(
                creds,
                symbol,
                volume_lots,
                stop_loss_pips,
                take_profit_pips,
                expected_account_id,
                resolve,
            )?;
            (
                prep.creds,
                prep.account_id,
                Some(prep.volume_units),
                prep.relative_stop_loss,
                prep.relative_take_profit,
            )
        } else {
            let resolved =
                resolve_order_symbol_from_creds(&creds, symbol, expected_account_id, resolve)?;
            let (relative_stop_loss, relative_take_profit) =
                relative_brackets_from_broker_symbol(&resolved, stop_loss_pips, take_profit_pips)?;
            (
                creds,
                resolved.account_id,
                None,
                relative_stop_loss,
                relative_take_profit,
            )
        };

    let (limit_price, stop_price) = match order_type {
        CTraderOrderType::Limit => (trigger_price, None),
        _ => (None, trigger_price),
    };

    let amend = CTraderAmendOrderRequest {
        account_id,
        order_id,
        volume,
        limit_price,
        stop_price,
        expiration_timestamp_ms: expiry_unix_ms,
        // Absolute SL/TP are the position-amend path's currency; a resting
        // order's bracket is expressed as a distance from its own entry, the
        // same encoding `submit_pending_order_blocking` writes.
        stop_loss: None,
        take_profit: None,
        slippage_in_points: None,
        relative_stop_loss,
        relative_take_profit,
        guaranteed_stop_loss: None,
        trailing_stop_loss: None,
        stop_trigger_method: None,
    };

    Ok(CTraderExecutionRuntimeRequest {
        client_id: creds.client_id,
        client_secret: creds.client_secret,
        access_token: creds.access_token,
        environment: creds.environment,
        account_id: creds.account_id_str,
        request: CTraderExecutionRequest::AmendOrder(Box::new(amend)),
    })
}

/// Modify the stop-loss / take-profit of an ALREADY-OPEN position
/// (`ProtoOAAmendPositionSLTPReq`, 2026-06-10). `stop_loss` / `take_profit`
/// are ABSOLUTE prices (cTrader's position-amend proto is price-based, unlike
/// the pip-relative new-order path); `None` leaves that bracket untouched. At
/// least one of the two must be provided, or there is nothing to amend.
///
/// This is the capability that lets the bot trail a winner or pull a stop to
/// breakeven without closing and re-opening the position.
///
/// **UNBOUND to any admission decision — that is what this name now means.**
/// This is the operator's own manual amend (`POST /positions/protection`),
/// which the operator ruled is his. Anything that was ADMITTED against a
/// specific broker environment — the autopilot's trailing stop above all —
/// must call [`amend_position_sltp_expecting`] instead and pass the
/// environment it was admitted against.
///
/// See #199: this was the one order-path call still resolving credentials
/// without the environment check that `resolve_creds_expecting` added to the
/// other four paths, so a Demo→Live flip mid-iteration could send a
/// stop-modification to the environment the caller was never admitted to.
pub fn amend_position_sltp_blocking(
    position_id: i64,
    stop_loss: Option<f64>,
    take_profit: Option<f64>,
    trailing_stop_loss: Option<bool>,
) -> Result<CTraderExecutionOutcome> {
    amend_position_sltp_expecting(
        position_id,
        stop_loss,
        take_profit,
        trailing_stop_loss,
        None,
    )
}

/// [`amend_position_sltp_blocking`] bound to the broker environment the caller
/// was admitted against.
///
/// `expected_is_live: Some(_)` REFUSES the amend outright — no request leaves
/// the process — when `broker_credentials.toml` now names the other
/// environment. `None` reproduces the old, unbound behaviour and is correct
/// only for the operator's manual route.
///
/// Logs failures independently of its caller. The autonomous loop additionally
/// requires a matching broker replacement event before advancing its local
/// confirmed stop. A transport error is not proof the broker left it unchanged.
pub fn amend_position_sltp_expecting(
    position_id: i64,
    stop_loss: Option<f64>,
    take_profit: Option<f64>,
    trailing_stop_loss: Option<bool>,
    expected_is_live: Option<bool>,
) -> Result<CTraderExecutionOutcome> {
    amend_position_sltp_expecting_account(
        position_id,
        stop_loss,
        take_profit,
        trailing_stop_loss,
        expected_is_live,
        None,
    )
}

/// Protective amendment bound to the autonomous loop's exact account as well
/// as environment. Does not impose entry/historical prerequisites on protection.
pub fn amend_position_sltp_expecting_account(
    position_id: i64,
    stop_loss: Option<f64>,
    take_profit: Option<f64>,
    trailing_stop_loss: Option<bool>,
    expected_is_live: Option<bool>,
    expected_account_id: Option<i64>,
) -> Result<CTraderExecutionOutcome> {
    let result = amend_position_sltp_inner(
        position_id,
        stop_loss,
        take_profit,
        trailing_stop_loss,
        expected_is_live,
        expected_account_id,
    );
    match &result {
        Ok(outcome) => {
            tracing::debug!(
                target: "neoethos_app::broker_api",
                position_id,
                ?stop_loss,
                ?take_profit,
                status = ?outcome.status,
                "position SL/TP amend returned"
            );
        }
        Err(e) => {
            tracing::error!(
                target: "neoethos_app::broker_api",
                position_id,
                ?stop_loss,
                ?take_profit,
                ?expected_is_live,
                ?expected_account_id,
                error = %e,
                "POSITION SL/TP AMEND UNCONFIRMED — local confirmed protection was not \
                 advanced. If sending had begun, a transport failure does not prove the \
                 broker left the stop unchanged; reconcile the actual position."
            );
        }
    }
    result
}

fn amend_position_sltp_inner(
    position_id: i64,
    stop_loss: Option<f64>,
    take_profit: Option<f64>,
    trailing_stop_loss: Option<bool>,
    expected_is_live: Option<bool>,
    expected_account_id: Option<i64>,
) -> Result<CTraderExecutionOutcome> {
    if stop_loss.is_none() && take_profit.is_none() {
        return Err(anyhow!(
            "amend_position_sltp requires at least one of stopLoss / takeProfit"
        ));
    }
    let creds = resolve_creds_expecting(expected_is_live)?;
    let runtime_request = amend_position_sltp_request_from_creds(
        creds,
        position_id,
        stop_loss,
        take_profit,
        trailing_stop_loss,
        expected_account_id,
    )?;
    ProductionCTraderExecutionBackend::default().execute(&runtime_request)
}

fn amend_position_sltp_request_from_creds(
    creds: ResolvedCreds,
    position_id: i64,
    stop_loss: Option<f64>,
    take_profit: Option<f64>,
    trailing_stop_loss: Option<bool>,
    expected_account_id: Option<i64>,
) -> Result<CTraderExecutionRuntimeRequest> {
    let account_id: i64 = creds
        .account_id_str
        .parse()
        .map_err(|_| anyhow!("account_id '{}' is not numeric", creds.account_id_str))?;
    if let Some(expected) = expected_account_id {
        anyhow::ensure!(
            expected > 0 && account_id == expected,
            "REFUSING position protection amendment: resolved cTrader account differs from the admitted account"
        );
    }
    Ok(CTraderExecutionRuntimeRequest {
        client_id: creds.client_id,
        client_secret: creds.client_secret,
        access_token: creds.access_token,
        environment: creds.environment,
        account_id: creds.account_id_str,
        request: CTraderExecutionRequest::AmendPositionSltp(CTraderAmendPositionSltpRequest {
            account_id,
            position_id,
            stop_loss,
            take_profit,
            guaranteed_stop_loss: None,
            trailing_stop_loss,
            stop_loss_trigger_method: None,
        }),
    })
}

// ─── #238: the margin-call feed finally has a reader ───────────────────────
//
// `ctrader_messages::build_margin_call_list_request` shipped in the 2026-06-10
// API-completeness pass and had ZERO callers outside its own unit test until
// today. Every breaker in `live_trading` keys off balance and REALISED P&L, so
// an UNREALISED margin emergency — the one that ends with the broker
// liquidating the account — reached the operator only if he happened to be
// watching cTrader's own platform.
//
// What is polled, and why it takes four RPCs on one session:
//   * `ProtoOAMarginCallListReq` (2167) gives the CONFIGURED thresholds. It
//     does NOT say whether one is breached.
//   * `ProtoOATraderReq` (2121) gives the balance.
//   * `ProtoOAReconcileReq` (2124) gives the open positions, each carrying its
//     own `usedMargin`.
//   * `ProtoOAGetPositionUnrealizedPnLReq` (2187) gives the broker's own
//     unrealised P&L, because `trader.unrealized_pnl` is 0.0 out of the
//     runtime loader (`ctrader_account.rs:402`) and the whole point of this
//     poll is the unrealised half.
// Margin level = equity / used_margin * 100, equity = balance + unrealised.
// A breach is `margin_level_pct <= threshold`.

/// A single margin-level reading, with everything needed to explain it.
#[derive(Debug, Clone)]
pub struct MarginStatus {
    pub account_id: i64,
    pub environment_label: &'static str,
    pub balance: f64,
    pub unrealized_pnl: f64,
    pub equity: f64,
    /// Sum of `usedMargin` across open positions, account currency.
    pub used_margin: f64,
    /// `equity / used_margin * 100`. `None` when `used_margin` is zero — with
    /// no position open there is no margin level and no margin call.
    pub margin_level_pct: Option<f64>,
    pub thresholds: CTraderMarginCallListSnapshot,
    /// The tightest configured threshold the current level has fallen to or
    /// below. `Some` means MARGIN CALL.
    pub breached_threshold_pct: Option<f64>,
    /// Positions whose `usedMargin` the broker omitted. Counted, never
    /// silently treated as zero — a missing denominator makes the computed
    /// margin level OPTIMISTIC, which is the dangerous direction.
    pub positions_missing_used_margin: usize,
    pub open_position_count: usize,
}

impl MarginStatus {
    pub fn is_margin_call(&self) -> bool {
        self.breached_threshold_pct.is_some()
    }
}

/// Marker prefix on the error returned when the broker ANSWERED but this build
/// could not read the answer — as opposed to not reaching the broker at all.
///
/// The two must be told apart because they get different treatment:
/// "unreachable" spends a failure budget before halting (network blips are
/// routine), while "we asked, it replied, and we do not understand the reply"
/// halts on the FIRST occurrence. A wire-format change is not transient, and
/// resolving it toward "keep trading" is exactly the fail-open this system
/// keeps being bitten by.
pub const MARGIN_STATUS_UNREADABLE_SENTINEL: &str = "MARGIN_STATUS_UNREADABLE";

/// Is there a broker account configured at all?
///
/// The margin-call watchdog needs this to tell "the broker is unreachable"
/// (an emergency — there may be open positions nobody can see) apart from
/// "no broker has ever been set up on this machine" (a fresh install, where
/// there is nothing to protect and halting would be nonsense). Cheap: one
/// `broker_credentials.toml` read, no network, no token refresh.
pub fn broker_credentials_configured() -> bool {
    let settings = load_broker_settings();
    let ct = &settings.ctrader;
    !ct.client_id.trim().is_empty()
        && !ct.client_secret.trim().is_empty()
        && !ct.accounts.is_empty()
}

/// Poll the broker for the account's margin-call thresholds and its current
/// margin level, in ONE authenticated session.
///
/// Blocking (sync WSS); callers must wrap in `spawn_blocking` or run on a
/// dedicated thread. Read-only — resolves credentials with `resolve_creds()`
/// (no admission decision to honour, exactly like every other read path).
///
/// **Fail-closed contract for the caller.** Two different failures come back,
/// and `app_services::margin_call` treats them differently:
///   * A transport / broker error means "we could not ask". The poller counts
///     it and halts after [`margin_call::MAX_CONSECUTIVE_POLL_FAILURES`]
///     consecutive occurrences — never treats it as "everything is fine".
///   * A parse failure AFTER the broker answered is tagged with
///     [`MARGIN_STATUS_UNREADABLE_SENTINEL`] and halts on the FIRST
///     occurrence, because a wire-format change does not heal on retry.
///
/// [`margin_call::MAX_CONSECUTIVE_POLL_FAILURES`]: crate::app_services::margin_call::MAX_CONSECUTIVE_POLL_FAILURES
pub fn fetch_margin_status_blocking() -> Result<MarginStatus> {
    let creds = resolve_creds()?;
    let transport = ProductionCTraderOpenApiTransport::new(creds.environment.endpoint_host());
    fetch_margin_status_with_transport(&transport, &creds)
}

#[path = "broker_api_entry.rs"]
mod entry;
pub(crate) use entry::{
    LiveEntryContext, fetch_bound_broker_symbol_blocking, fetch_live_entry_context_blocking,
    required_entry_conversion_symbol,
};

/// The same margin poll, with transport injected for wire-envelope regressions.
/// No additional credential read or alternate monetary implementation.
fn fetch_margin_status_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    creds: &ResolvedCreds,
) -> Result<MarginStatus> {
    let account_id: i64 = creds
        .account_id_str
        .parse()
        .map_err(|_| anyhow!("account_id '{}' is not numeric", creds.account_id_str))?;
    anyhow::ensure!(account_id > 0, "cTrader account id must be positive");

    let responses = crate::app_services::ctrader_messages::send_sequence_resilient(
        transport,
        &[
            build_application_auth_request(&creds.client_id, &creds.client_secret, "app-auth-1"),
            build_account_auth_request(account_id, &creds.access_token, "account-auth-1"),
            build_margin_call_list_request(account_id, "margin-call-list-1"),
            build_trader_request(account_id, "trader-1"),
            build_reconcile_request(account_id, false, "reconcile-1"),
            build_get_position_unrealized_pnl_request(account_id, "unrealized-pnl-1"),
        ],
        // All six are REQUIRED. `min_ok` is not "how many we would like" — a
        // margin level computed without the reconcile snapshot or without the
        // unrealised P&L would be wrong in the optimistic direction, which is
        // the one that gets an account liquidated.
        6,
        "cTrader margin-call status",
    )?;

    Ok(parse_margin_status_responses(creds, &responses)?.0)
}

// Shared by the ordinary margin poll and the entry request's same-session
// account/asset/symbol snapshot. Monetary and identity checks stay in one place.
fn parse_margin_status_responses(
    creds: &ResolvedCreds,
    responses: &[String],
) -> Result<(
    MarginStatus,
    crate::app_services::ctrader_account::CTraderTraderSnapshot,
)> {
    let account_id = creds.account_id_str.parse::<i64>()?;
    anyhow::ensure!(account_id > 0, "cTrader account id must be positive");
    if responses.len() != 6 {
        return Err(anyhow!(
            "expected 6 cTrader margin-status responses, received {}",
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
        CTRADER_OA_MARGIN_CALL_LIST_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[3], CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(&responses[4], CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(
        &responses[5],
        CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
    )?;

    // The broker answered. From here on, a failure means the WIRE FORMAT
    // changed under us — tag it so `margin_call::poll_once` halts immediately
    // instead of spending its unreachable-budget on a defect that will not
    // resolve itself.
    let unreadable = |what: &str, e: anyhow::Error| {
        anyhow!("{MARGIN_STATUS_UNREADABLE_SENTINEL}: could not read the {what} — {e}")
    };
    // Validate every account-scoped envelope before parsers discard identity,
    // including empty position/PnL arrays. Broker-error/type precedence above
    // remains unchanged; a foreign or missing identity is an unreadable reply.
    for response in &responses[1..] {
        require_response_account(response, account_id, "cTrader margin status")
            .map_err(|error| unreadable("account-scoped response", error))?;
    }
    let thresholds = parse_margin_call_list_response(&responses[2])
        .map_err(|e| unreadable("margin-call threshold list", e))?;
    let trader =
        parse_trader_response(&responses[3]).map_err(|e| unreadable("trader balance", e))?;
    let reconcile = parse_reconcile_response(&responses[4])
        .map_err(|e| unreadable("open-position reconcile snapshot", e))?;
    let pnl = parse_get_position_unrealized_pnl_response(&responses[5])
        .map_err(|e| unreadable("unrealised P&L snapshot", e))?;

    let unrealized_pnl_by_position = reconcile_broker_unrealized_pnl(&reconcile, &pnl)
        .map_err(|error| unreadable("position/PnL reconciliation", error))?;
    let unrealized_pnl: f64 = unrealized_pnl_by_position
        .values()
        .map(|position| position.net_unrealized_pnl)
        .sum();
    let equity = trader.balance + unrealized_pnl;

    let mut used_margin = 0.0_f64;
    let mut positions_missing_used_margin = 0_usize;
    for p in &reconcile.positions {
        match p.used_margin {
            Some(m) if m.is_finite() && m >= 0.0 => used_margin += m,
            _ => positions_missing_used_margin += 1,
        }
    }

    let margin_level_pct = if used_margin > 0.0 {
        Some(equity / used_margin * 100.0)
    } else {
        None
    };

    let breached_threshold_pct = match (margin_level_pct, thresholds.tightest_threshold_pct()) {
        (Some(level), Some(tightest)) if level <= tightest => Some(tightest),
        _ => None,
    };

    Ok((
        MarginStatus {
            account_id: trader.account_id,
            environment_label: creds.env_label,
            balance: trader.balance,
            unrealized_pnl,
            equity,
            used_margin,
            margin_level_pct,
            thresholds,
            breached_threshold_pct,
            positions_missing_used_margin,
            open_position_count: reconcile.positions.len(),
        },
        trader,
    ))
}

/// Maximum cTrader history window for the order-list / cash-flow RPCs.
/// The broker rejects windows wider than one week; we fail loud before the
/// round-trip instead of letting the broker bounce it (operator's
/// defensive-code rule).
const CTRADER_HISTORY_MAX_WINDOW_MS: i64 = 604_800_000; // 7 days

fn validate_history_window(from_ms: i64, to_ms: i64) -> Result<()> {
    if to_ms < from_ms {
        return Err(anyhow!(
            "history window is inverted: from={from_ms} > to={to_ms}"
        ));
    }
    if to_ms - from_ms > CTRADER_HISTORY_MAX_WINDOW_MS {
        return Err(anyhow!(
            "history window {} ms exceeds the cTrader maximum of {} ms (1 week) — narrow the range",
            to_ms - from_ms,
            CTRADER_HISTORY_MAX_WINDOW_MS
        ));
    }
    Ok(())
}

fn require_response_account(response: &str, expected_account: i64, context: &str) -> Result<()> {
    let envelope = crate::app_services::ctrader_messages::parse_open_api_envelope(response)?;
    let actual = envelope
        .payload
        .get("ctidTraderAccountId")
        .and_then(serde_json::Value::as_i64);
    anyhow::ensure!(
        actual == Some(expected_account),
        "{context} response account {actual:?} differs from requested account {expected_account}"
    );
    Ok(())
}

/// Read current symbol lot sizes on the same credential/environment snapshot.
/// This is a display conversion, not historical contract or trading authority.
/// Batch repeated symbol IDs; never make one metadata request per order.
fn fetch_current_broker_lot_sizes_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    creds: &ResolvedCreds,
    account_id: i64,
    symbol_ids: &BTreeSet<i64>,
) -> Result<BTreeMap<i64, i64>> {
    anyhow::ensure!(
        symbol_ids.iter().all(|id| *id > 0),
        "invalid broker symbol id for lotSize lookup"
    );
    let ids: Vec<i64> = symbol_ids.iter().copied().collect();
    let mut lot_sizes = BTreeMap::new();
    for chunk in ids.chunks(50) {
        let responses = transport.send_sequence(&[
            build_application_auth_request(
                &creds.client_id,
                &creds.client_secret,
                "lot-size-app-auth",
            ),
            build_account_auth_request(account_id, &creds.access_token, "lot-size-account-auth"),
            build_symbol_by_id_request(account_id, chunk, "lot-size-symbols"),
        ])?;
        anyhow::ensure!(
            responses.len() == 3,
            "expected 3 cTrader lotSize responses, received {}",
            responses.len()
        );
        ensure_success_payload_type(
            &responses[0],
            CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
        )?;
        ensure_success_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
        ensure_success_payload_type(&responses[2], CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE)?;
        require_response_account(&responses[1], account_id, "lotSize authentication")?;
        require_response_account(&responses[2], account_id, "lotSize")?;
        for symbol in parse_symbol_by_id_response(&responses[2])? {
            anyhow::ensure!(
                chunk.binary_search(&symbol.symbol_id).is_ok(),
                "lotSize response contains unrequested symbol {}",
                symbol.symbol_id
            );
            let lot_size = symbol
                .lot_size
                .ok_or_else(|| anyhow!("broker omitted lotSize for symbol {}", symbol.symbol_id))?;
            broker_lots_from_wire_volume_v1(0, lot_size)?;
            anyhow::ensure!(
                lot_sizes.insert(symbol.symbol_id, lot_size).is_none(),
                "duplicate lotSize row for symbol {}",
                symbol.symbol_id
            );
        }
        for id in chunk {
            anyhow::ensure!(
                lot_sizes.contains_key(id),
                "broker has no current lotSize for symbol {id} (possibly archived)"
            );
        }
    }
    Ok(lot_sizes)
}

/// Account-wide historical orders over `[from_ms, to_ms]` (ms).
/// `ProtoOAOrderListReq`. Blocking (sync WSS) — wrap in `spawn_blocking`.
pub fn fetch_broker_order_history_blocking(
    from_ms: i64,
    to_ms: i64,
) -> Result<CTraderOrderHistoryBundle> {
    validate_history_window(from_ms, to_ms)?;
    let creds = resolve_creds()?;
    let transport = ProductionCTraderOpenApiTransport::new(creds.environment.endpoint_host());
    fetch_broker_order_history_with_transport(&transport, &creds, from_ms, to_ms)
}

fn fetch_broker_order_history_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    creds: &ResolvedCreds,
    from_ms: i64,
    to_ms: i64,
) -> Result<CTraderOrderHistoryBundle> {
    validate_history_window(from_ms, to_ms)?;
    let account_id: i64 = creds
        .account_id_str
        .parse()
        .map_err(|_| anyhow!("account_id '{}' is not numeric", creds.account_id_str))?;
    let responses = transport.send_sequence(&[
        build_application_auth_request(&creds.client_id, &creds.client_secret, "app-auth-1"),
        build_account_auth_request(account_id, &creds.access_token, "account-auth-1"),
        build_order_list_request(account_id, from_ms, to_ms, "order-list-1"),
    ])?;
    if responses.len() != 3 {
        return Err(anyhow!(
            "expected 3 cTrader order-history responses, received {}",
            responses.len()
        ));
    }
    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(&responses[2], CTRADER_OA_ORDER_LIST_RESPONSE_PAYLOAD_TYPE)?;
    require_response_account(&responses[1], account_id, "order-history authentication")?;
    let mut bundle = parse_order_list_response(&responses[2])?;
    anyhow::ensure!(
        bundle.account_id == account_id,
        "order-history response belongs to account {}, expected {account_id}",
        bundle.account_id
    );
    let symbol_ids = bundle.orders.iter().map(|order| order.symbol_id).collect();
    let conversion = (|| -> Result<()> {
        let lot_sizes = fetch_current_broker_lot_sizes_with_transport(
            transport,
            creds,
            account_id,
            &symbol_ids,
        )?;
        // Resolve every value before publishing any lots, so one failed join
        // cannot leave a misleading partially converted history response.
        let volumes = bundle
            .orders
            .iter()
            .map(|order| -> Result<_> {
                let lot_size = lot_sizes[&order.symbol_id];
                Ok((
                    broker_lots_from_wire_volume_v1(order.volume_raw_centi_units, lot_size)?,
                    order
                        .executed_volume_raw_centi_units
                        .map(|raw| broker_lots_from_wire_volume_v1(raw, lot_size))
                        .transpose()?,
                    lot_size,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        for (order, (lots, executed_lots, lot_size)) in bundle.orders.iter_mut().zip(volumes) {
            order.volume_lots = Some(lots);
            order.executed_volume_lots = executed_lots;
            order.lot_size_raw_centi_units = Some(lot_size);
        }
        if !bundle.orders.is_empty() {
            bundle.lot_size_observed_at_unix_ms = Some(chrono::Utc::now().timestamp_millis());
        }
        Ok(())
    })();
    if let Err(error) = conversion {
        bundle.lot_size_error = Some(format!(
            "Current broker lotSize unavailable; volumes are shown in units: {error:#}"
        ));
    }
    Ok(bundle)
}

/// Cash-flow history (deposits / withdrawals / swaps / fees) over
/// `[from_ms, to_ms]` (ms). `ProtoOACashFlowHistoryListReq`. Blocking.
pub fn fetch_broker_cash_flow_history_blocking(
    from_ms: i64,
    to_ms: i64,
) -> Result<CTraderCashFlowBundle> {
    validate_history_window(from_ms, to_ms)?;
    let creds = resolve_creds()?;
    let account_id: i64 = creds
        .account_id_str
        .parse()
        .map_err(|_| anyhow!("account_id '{}' is not numeric", creds.account_id_str))?;
    let transport = ProductionCTraderOpenApiTransport::new(creds.environment.endpoint_host());
    let responses = transport.send_sequence(&[
        build_application_auth_request(&creds.client_id, &creds.client_secret, "app-auth-1"),
        build_account_auth_request(account_id, &creds.access_token, "account-auth-1"),
        build_cash_flow_history_list_request(account_id, from_ms, to_ms, "cashflow-1"),
    ])?;
    if responses.len() != 3 {
        return Err(anyhow!(
            "expected 3 cTrader cash-flow responses, received {}",
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
        CTRADER_OA_CASH_FLOW_HISTORY_LIST_RESPONSE_PAYLOAD_TYPE,
    )?;
    parse_cash_flow_history_response(&responses[2])
}

/// Account-wide deal history over `[from_ms, to_ms]`, preserving the broker's
/// `hasMore` truncation bit. Callers that need completeness or absence proof
/// must use this instead of the bounded `recent_deals` runtime field.
pub fn fetch_broker_deal_history_blocking(
    from_ms: i64,
    to_ms: i64,
    max_rows: i32,
) -> Result<CTraderDealListBundle> {
    validate_history_window(from_ms, to_ms)?;
    if max_rows <= 0 {
        return Err(anyhow!(
            "deal-history max_rows must be positive; got {max_rows}"
        ));
    }
    let creds = resolve_creds()?;
    let account_id: i64 = creds
        .account_id_str
        .parse()
        .map_err(|_| anyhow!("account_id '{}' is not numeric", creds.account_id_str))?;
    let transport = ProductionCTraderOpenApiTransport::new(creds.environment.endpoint_host());
    let responses = transport.send_sequence(&[
        build_application_auth_request(&creds.client_id, &creds.client_secret, "app-auth-1"),
        build_account_auth_request(account_id, &creds.access_token, "account-auth-1"),
        build_deal_list_request(
            &CTraderDealListRequest {
                account_id,
                from_timestamp_ms: Some(from_ms),
                to_timestamp_ms: Some(to_ms),
                max_rows: Some(max_rows),
            },
            "deal-history-1",
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
    let bundle = parse_deal_list_bundle_response(&responses[2])?;
    if bundle.account_id != account_id {
        return Err(anyhow!(
            "deal-history response belongs to account {}, expected {}",
            bundle.account_id,
            account_id
        ));
    }
    Ok(bundle)
}

/// Pre-trade margin estimate for each of `volumes` (0.01-unit wire volume) on
/// `symbol_id`. `ProtoOAExpectedMarginReq`. Blocking.
pub fn fetch_broker_expected_margin_blocking(
    symbol_id: i64,
    volumes: Vec<i64>,
) -> Result<CTraderExpectedMarginBundle> {
    validate_expected_margin_request(symbol_id, &volumes)?;
    let creds = resolve_creds()?;
    let transport = ProductionCTraderOpenApiTransport::new(creds.environment.endpoint_host());
    fetch_broker_expected_margin_with_transport(&transport, &creds, symbol_id, &volumes)
}

fn validate_expected_margin_request(symbol_id: i64, volumes: &[i64]) -> Result<()> {
    anyhow::ensure!(
        symbol_id > 0,
        "expected-margin requires a positive symbol id"
    );
    anyhow::ensure!(
        !volumes.is_empty(),
        "expected-margin requires at least one volume"
    );
    anyhow::ensure!(
        volumes
            .iter()
            .all(|v| (1..=MAX_EXACT_BROKER_VOLUME).contains(v)),
        "expected-margin requires positive, exactly representable centi-unit volumes"
    );
    anyhow::ensure!(
        volumes.iter().collect::<BTreeSet<_>>().len() == volumes.len(),
        "expected-margin request contains duplicate volumes"
    );
    Ok(())
}

fn fetch_broker_expected_margin_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    creds: &ResolvedCreds,
    symbol_id: i64,
    volumes: &[i64],
) -> Result<CTraderExpectedMarginBundle> {
    validate_expected_margin_request(symbol_id, volumes)?;
    let account_id: i64 = creds
        .account_id_str
        .parse()
        .map_err(|_| anyhow!("account_id '{}' is not numeric", creds.account_id_str))?;
    let responses = transport.send_sequence(&[
        build_application_auth_request(&creds.client_id, &creds.client_secret, "app-auth-1"),
        build_account_auth_request(account_id, &creds.access_token, "account-auth-1"),
        build_expected_margin_request(account_id, symbol_id, volumes, "exp-margin-1"),
    ])?;
    if responses.len() != 3 {
        return Err(anyhow!(
            "expected 3 cTrader expected-margin responses, received {}",
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
        CTRADER_OA_EXPECTED_MARGIN_RESPONSE_PAYLOAD_TYPE,
    )?;
    require_response_account(&responses[1], account_id, "expected-margin authentication")?;
    let mut bundle = parse_expected_margin_response(&responses[2])?;
    anyhow::ensure!(
        bundle.account_id == account_id,
        "expected-margin response belongs to account {}, expected {account_id}",
        bundle.account_id
    );
    let requested: BTreeSet<i64> = volumes.iter().copied().collect();
    let returned: BTreeSet<i64> = bundle
        .entries
        .iter()
        .map(|entry| entry.volume_raw_centi_units)
        .collect();
    anyhow::ensure!(
        returned == requested && returned.len() == bundle.entries.len(),
        "expected-margin response volumes are missing, duplicated, or differ from the request"
    );
    bundle.symbol_id = Some(symbol_id);
    match fetch_current_broker_lot_sizes_with_transport(
        transport,
        creds,
        account_id,
        &BTreeSet::from([symbol_id]),
    ) {
        Ok(lot_sizes) => {
            let lot_size = lot_sizes[&symbol_id];
            for entry in &mut bundle.entries {
                entry.volume_lots = Some(broker_lots_from_wire_volume_v1(
                    entry.volume_raw_centi_units,
                    lot_size,
                )?);
                entry.lot_size_raw_centi_units = Some(lot_size);
            }
            bundle.lot_size_observed_at_unix_ms = Some(chrono::Utc::now().timestamp_millis());
        }
        Err(error) => {
            bundle.lot_size_error = Some(format!(
                "Current broker lotSize unavailable; volumes are shown in units: {error:#}"
            ));
        }
    }
    Ok(bundle)
}

/// The cTID profile (user id) behind the saved access token.
/// `ProtoOAGetCtidProfileByTokenReq` — token-scoped, no account-auth. Blocking.
pub fn fetch_broker_ctid_profile_blocking() -> Result<CTraderCtidProfileSnapshot> {
    let creds = resolve_creds()?;
    let transport = ProductionCTraderOpenApiTransport::new(creds.environment.endpoint_host());
    let responses = transport.send_sequence(&[
        build_application_auth_request(&creds.client_id, &creds.client_secret, "app-auth-1"),
        build_get_ctid_profile_by_token_request(&creds.access_token, "ctid-profile-1"),
    ])?;
    if responses.len() != 2 {
        return Err(anyhow!(
            "expected 2 cTrader cTID-profile responses, received {}",
            responses.len()
        ));
    }
    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(
        &responses[1],
        CTRADER_OA_GET_CTID_PROFILE_BY_TOKEN_RESPONSE_PAYLOAD_TYPE,
    )?;
    parse_ctid_profile_response(&responses[1])
}

/// The broker's Open API proto version. `ProtoOAVersionReq` — app-level,
/// no account, no token. Blocking; useful as a connectivity probe.
pub fn fetch_broker_version_blocking() -> Result<CTraderServerVersionSnapshot> {
    let creds = resolve_creds()?;
    let transport = ProductionCTraderOpenApiTransport::new(creds.environment.endpoint_host());
    let responses = transport.send_sequence(&[
        build_application_auth_request(&creds.client_id, &creds.client_secret, "app-auth-1"),
        build_version_request("version-1"),
    ])?;
    if responses.len() != 2 {
        return Err(anyhow!(
            "expected 2 cTrader version responses, received {}",
            responses.len()
        ));
    }
    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[1], CTRADER_OA_VERSION_RESPONSE_PAYLOAD_TYPE)?;
    parse_version_response(&responses[1])
}

#[cfg(test)]
#[path = "broker_api_volume_tests.rs"]
mod broker_volume_tests;

#[cfg(test)]
#[path = "broker_api_account_pin_tests.rs"]
mod broker_account_pin_tests;

#[cfg(test)]
mod exact_broker_order_unit_tests {
    use super::{
        HistoricalBar, OrderSide, RecentBrokerTrendbarSnapshot, chart_fetch_from_ms,
        ensure_broker_allows_new_order, relative_distance_from_broker_symbol,
        validate_broker_bar_order, wire_volume_from_broker_lot_size,
    };
    use neoethos_core::CanonicalTimeframe;

    #[test]
    fn protection_request_binds_the_same_account_without_entry_prerequisites() {
        let creds = |account_id: &str| super::ResolvedCreds {
            client_id: "synthetic-client".into(),
            client_secret: "synthetic-secret".into(),
            access_token: "synthetic-token".into(),
            account_id_str: account_id.into(),
            environment: super::CTraderEnvironment::Demo,
            env_label: "Demo",
        };
        let request = super::amend_position_sltp_request_from_creds(
            creds("42"),
            7,
            Some(1.09),
            None,
            None,
            Some(42),
        )
        .expect("matched protection account requires no historical fills or entry quote");
        assert_eq!(request.account_id, "42");
        let super::CTraderExecutionRequest::AmendPositionSltp(amend) = request.request else {
            panic!("only protective amendment is built")
        };
        assert_eq!(
            (amend.account_id, amend.position_id, amend.stop_loss),
            (42, 7, Some(1.09))
        );
        for account in ["99", "0", "-1", "invalid"] {
            assert!(
                super::amend_position_sltp_request_from_creds(
                    creds(account),
                    7,
                    Some(1.09),
                    None,
                    None,
                    Some(42),
                )
                .is_err()
            );
        }
        assert!(
            super::amend_position_sltp_request_from_creds(
                creds("42"),
                7,
                Some(1.09),
                None,
                None,
                Some(0),
            )
            .is_err()
        );
        assert!(
            super::amend_position_sltp_request_from_creds(
                creds("99"),
                7,
                Some(1.09),
                None,
                None,
                None,
            )
            .is_ok(),
            "manual legacy wrapper remains deliberately unpinned"
        );
    }

    #[test]
    fn close_request_binds_the_account_from_the_same_resolved_credentials() {
        let creds = |account_id: &str| super::ResolvedCreds {
            client_id: "fixture-client".to_string(),
            client_secret: "fixture-secret".to_string(),
            access_token: "fixture-token".to_string(),
            account_id_str: account_id.to_string(),
            environment: super::CTraderEnvironment::Demo,
            env_label: "Demo",
        };
        let request = super::close_request_from_resolved_creds(creds("42"), 7, 37_001, Some(42))
            .expect("unchanged admitted account is allowed");
        assert_eq!(request.account_id, "42");
        let super::CTraderExecutionRequest::ClosePosition(close) = request.request else {
            panic!("must build only a close request");
        };
        assert_eq!(
            (close.account_id, close.position_id, close.volume),
            (42, 7, 37_001)
        );

        let error = super::close_request_from_resolved_creds(creds("99"), 7, 37_001, Some(42))
            .expect_err("same-environment foreign account must be refused before execution");
        assert!(
            error
                .to_string()
                .contains("admitted account 42 to resolved account 99")
        );
        assert!(super::close_request_from_resolved_creds(creds("42"), 7, 37_001, None).is_ok());
        assert!(super::close_request_from_resolved_creds(creds("99"), 7, 37_001, None).is_ok());
        assert!(
            super::close_request_from_resolved_creds(creds("invalid"), 7, 37_001, Some(42))
                .is_err()
        );
    }

    #[test]
    fn partial_chart_pages_never_gain_canonical_feature_authority() {
        use super::RecentTrendbarPurpose;
        assert!(
            RecentTrendbarPurpose::DisplayOnly
                .validate_pagination(true)
                .is_ok()
        );
        assert!(
            RecentTrendbarPurpose::DisplayOnly
                .validate_pagination(false)
                .is_ok()
        );
        assert!(
            RecentTrendbarPurpose::CanonicalEvidence
                .validate_pagination(false)
                .is_ok()
        );
        assert!(
            RecentTrendbarPurpose::CanonicalEvidence
                .validate_pagination(true)
                .is_err()
        );
    }

    #[test]
    fn chart_paging_rejects_oversized_pages_before_contacting_the_broker() {
        let error = super::fetch_chart_bars_before_blocking(
            "EURUSD",
            "M5",
            1_700_000_000_000,
            super::CTRADER_RECENT_TRENDBAR_LIMIT + 1,
        )
        .expect_err("unbounded page must fail before credentials or network access");
        assert!(error.to_string().contains("one-page ceiling"));
    }

    #[test]
    fn wire_volume_uses_the_exact_broker_lot_size() {
        assert_eq!(
            wire_volume_from_broker_lot_size(0.01, 10_000_000).expect("EURUSD lotSize"),
            100_000
        );
        assert_eq!(
            wire_volume_from_broker_lot_size(0.25, 10_000).expect("broker CFD lotSize"),
            2_500
        );
        assert!(wire_volume_from_broker_lot_size(0.01, 0).is_err());
        assert!(wire_volume_from_broker_lot_size(0.000_000_05, 10_000_000).is_err());
        assert!(wire_volume_from_broker_lot_size(f64::MAX, i64::MAX).is_err());
    }

    #[test]
    fn relative_distance_uses_broker_pip_position_and_digits() {
        assert_eq!(
            relative_distance_from_broker_symbol(20.0, 5, 4).expect("five-digit FX"),
            200
        );
        assert_eq!(
            relative_distance_from_broker_symbol(26.0, 2, 1).expect("two-digit metal"),
            260_000
        );
        assert!(relative_distance_from_broker_symbol(0.005, 2, 1).is_err());
        assert!(relative_distance_from_broker_symbol(20.0, 6, 4).is_err());
        assert!(relative_distance_from_broker_symbol(20.0, 2, 3).is_err());
    }

    #[test]
    fn new_orders_obey_full_broker_trading_permissions() {
        assert!(ensure_broker_allows_new_order("EURUSD", OrderSide::Buy, true, true).is_ok());
        assert!(ensure_broker_allows_new_order("EURUSD", OrderSide::Sell, true, true).is_ok());

        let disabled = ensure_broker_allows_new_order("EURUSD", OrderSide::Buy, false, true)
            .expect_err("close-only or disabled symbol must reject new positions");
        assert!(disabled.to_string().contains("tradingMode=ENABLED"));

        let short_disabled = ensure_broker_allows_new_order("EURUSD", OrderSide::Sell, true, false)
            .expect_err("sell-to-open must obey broker short-selling permission");
        assert!(short_disabled.to_string().contains("forbids short selling"));

        assert!(
            ensure_broker_allows_new_order("EURUSD", OrderSide::Buy, true, false).is_ok(),
            "a long order does not require the short-selling permission"
        );
    }

    #[test]
    fn every_official_timeframe_uses_the_shared_typed_contract() {
        const TO_MS: i64 = 1_700_000_040_000;
        for timeframe in CanonicalTimeframe::ALL {
            let from = chart_fetch_from_ms(timeframe.as_str(), TO_MS, 2_000)
                .expect("chart request window");
            if timeframe.fixed_duration_ms().is_some() {
                assert!(from < TO_MS);
            } else {
                assert_eq!(from, 0, "calendar frames use count, not fake duration");
            }
        }
        assert!(chart_fetch_from_ms("H2", TO_MS, 2_000).is_err());
    }

    #[test]
    fn broker_order_validation_rejects_duplicate_and_descending_rows() {
        let bar = |timestamp_ms| HistoricalBar {
            timestamp_ms,
            open: 1.0,
            high: 1.1,
            low: 0.9,
            close: 1.0,
            volume: None,
        };
        assert!(validate_broker_bar_order(&[bar(1), bar(2)], "fixture").is_ok());
        assert!(validate_broker_bar_order(&[bar(1), bar(1)], "fixture").is_err());
        assert!(validate_broker_bar_order(&[bar(2), bar(1)], "fixture").is_err());
    }

    #[test]
    fn empty_chart_response_is_valid_but_cannot_become_a_live_feature_snapshot() {
        let identity = neoethos_data::CanonicalDatasetIdentity::ctrader(
            neoethos_data::CTraderEnvironment::Demo,
            "demo.ctraderapi.com",
            42,
            1,
            "EURUSD",
            CanonicalTimeframe::M1,
            neoethos_data::BarTimestampConvention::BarOpen,
        )
        .expect("test cTrader identity");
        let snapshot = RecentBrokerTrendbarSnapshot {
            identity,
            requested_from_ms: 1,
            requested_to_ms: 2,
            retrieved_unix_ms: 1,
            bars: Vec::new(),
        };
        assert!(snapshot.validate_with_empty_policy(true).is_ok());
        assert!(snapshot.validate().is_err());
    }
}
