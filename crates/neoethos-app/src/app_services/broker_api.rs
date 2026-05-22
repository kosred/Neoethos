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
    CTraderLightSymbolInfo, CTraderResolvedSymbol, CTraderSymbolLookupRequest,
    CTraderSymbolsListResult, HistoricalBar, load_historical_bars_only,
    parse_symbols_list_response, resolve_symbol,
};
use crate::app_services::ctrader_execution::{
    CTraderExecutionBackend, CTraderExecutionOutcome, CTraderExecutionRequest,
    CTraderExecutionRuntimeRequest, ProductionCTraderExecutionBackend,
};
use crate::app_services::ctrader_messages::{
    CTraderNewOrderRequest, CTraderOrderType, CTraderTradeSide,
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
    // conversion. cTrader expresses volume in centi-lots: 1 lot of
    // EURUSD ≈ 100,000 units in the base currency, and the protocol
    // wants that × 100 = 10,000,000. The resolved `lot_size` field is
    // the unit count per 1 lot (typically 100,000 for FX), so we go
    // `volume_units = volume_lots * lot_size * 100` to land in
    // centi-units the way `parse_execution_event` will read them back.
    let resolved: CTraderResolvedSymbol = resolve_symbol(&CTraderSymbolLookupRequest {
        client_id: creds.client_id.clone(),
        client_secret: creds.client_secret.clone(),
        access_token: creds.access_token.clone(),
        environment: creds.environment,
        account_id: creds.account_id_str.clone(),
        symbol_name: symbol.to_string(),
    })?;
    let lot_size = resolved.symbol.lot_size.unwrap_or(100_000);
    let volume_units = (volume_lots * lot_size as f64 * 100.0).round() as i64;
    if volume_units <= 0 {
        return Err(anyhow!(
            "computed volume ({volume_units}) is not positive — \
             check lot_size ({lot_size}) and volume_lots ({volume_lots})"
        ));
    }
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
    let relative_stop_loss =
        stop_loss_pips.map(|p| (p * pip_relative_units).round() as i64);
    let relative_take_profit =
        take_profit_pips.map(|p| (p * pip_relative_units).round() as i64);

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
