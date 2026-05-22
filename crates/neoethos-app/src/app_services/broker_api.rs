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

use crate::app_services::broker_config::CTraderBrokerEnvironment;
use crate::app_services::broker_persistence::load_broker_settings;
use crate::app_services::bootstrap_writer::write_bootstrap_vortex;
use crate::app_services::ctrader_bootstrap::NormalizedBar;
use crate::app_services::ctrader_data::{
    CTraderChartHistoryRequest, CTraderHistoricalBarsFetchResult,
    CTraderLightSymbolInfo, CTraderSymbolsListResult, HistoricalBar,
    load_historical_bars_only, parse_symbols_list_response,
};
use crate::app_services::ctrader_messages::{
    CTraderOpenApiTransport, ProductionCTraderOpenApiTransport,
    build_account_auth_request, build_application_auth_request,
    build_symbols_list_request,
};
use crate::app_services::trading::CTraderEnvironment;
use crate::app_services::secure_store::production_ctrader_token_store;

/// What `/broker/symbols` ultimately returns over the wire — kept here
/// so the server module just shovels it to JSON.
#[derive(Debug, Clone)]
pub struct BrokerSymbolsBundle {
    pub account_id: i64,
    pub environment: &'static str,
    pub symbols: Vec<CTraderLightSymbolInfo>,
    pub archived_symbols: Vec<String>,
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
    let responses = transport.send_sequence(&[
        build_application_auth_request(&creds.client_id, &creds.client_secret, "app-auth-1"),
        build_account_auth_request(account_id, &creds.access_token, "account-auth-1"),
        build_symbols_list_request(account_id, false, "symbols-1"),
    ])?;
    if responses.len() < 3 {
        return Err(anyhow!(
            "expected 3 cTrader symbols-list responses, received {}",
            responses.len()
        ));
    }

    let CTraderSymbolsListResult {
        account_id,
        symbols,
        archived_symbols,
    } = parse_symbols_list_response(&responses[2])?;

    Ok(BrokerSymbolsBundle {
        account_id,
        environment: creds.env_label,
        symbols,
        archived_symbols,
    })
}

/// Download historical bars for [from_ms, to_ms] and write the result
/// into the local data dir under `data/symbol=<sym>/timeframe=<tf>/`.
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
    let request = CTraderChartHistoryRequest {
        client_id: creds.client_id.clone(),
        client_secret: creds.client_secret.clone(),
        access_token: creds.access_token.clone(),
        environment: creds.environment,
        account_id: creds.account_id_str.clone(),
        symbol_name: symbol.to_string(),
        timeframe: timeframe.to_string(),
        from_timestamp_ms: from_ms,
        to_timestamp_ms: to_ms,
        // None — cTrader caps at ~5000 bars per request; the upstream
        // helper handles the cap. For very wide windows, the caller
        // should issue multiple POSTs (the UI's date-range picker
        // will encourage reasonable windows).
        count: None,
    };

    let CTraderHistoricalBarsFetchResult { bars, has_more, .. } =
        load_historical_bars_only(&request)?;

    let normalized = bars_to_normalized(&bars);
    let written_path =
        write_bootstrap_vortex(data_root, symbol, timeframe, &normalized)?;

    Ok(HistoricalDownloadOutcome {
        symbol: symbol.to_string(),
        timeframe: timeframe.to_string(),
        bar_count: bars.len(),
        has_more,
        written_path,
    })
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
