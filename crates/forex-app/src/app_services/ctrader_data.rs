use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    build_account_auth_request, build_application_auth_request, build_get_trendbars_request,
    build_get_tick_data_request, build_symbol_by_id_request, build_symbols_list_request,
    build_subscribe_live_trendbar_request, build_subscribe_spots_request,
    build_unsubscribe_live_trendbar_request, build_unsubscribe_spots_request,
    parse_ctrader_error_payload, parse_open_api_envelope, trendbar_period_value,
    CTraderOpenApiJsonMessage, CTraderOpenApiTransport, ProductionCTraderOpenApiTransport, CTRADER_QUOTE_TYPE_ASK,
    CTRADER_QUOTE_TYPE_BID,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
};
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderLightSymbolInfo {
    pub symbol_id: i64,
    pub symbol_name: String,
    pub enabled: bool,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderSymbolsListResult {
    pub account_id: i64,
    pub symbols: Vec<CTraderLightSymbolInfo>,
    pub archived_symbols: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub pnl_conversion_fee_rate: Option<i32>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Deserialize)]
struct FullSymbolPayload {
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
    trading_mode: Option<Value>,
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
        .map(|symbol| CTraderSymbolInfo {
            symbol_id: symbol.symbol_id,
            symbol_name: String::new(),
            display_name: String::new(),
            digits: symbol.digits,
            pip_position: symbol.pip_position,
            is_archived: false,
            is_trading_enabled: trading_mode_enabled(symbol.trading_mode.as_ref()),
            min_volume: symbol.min_volume,
            max_volume: symbol.max_volume,
            step_volume: symbol.step_volume,
            lot_size: symbol.lot_size,
            pnl_conversion_fee_rate: symbol.pnl_conversion_fee_rate,
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

    Ok(HistoricalTicksResult {
        symbol_id: envelope.payload.symbol_id.unwrap_or(symbol.symbol_id),
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

pub fn load_chart_history(request: &CTraderChartHistoryRequest) -> Result<CTraderChartHistoryResult> {
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
    let responses = transport.send_sequence(&[
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

    if responses.len() != 1 {
        return Err(anyhow!(
            "expected 1 cTrader bars-only response, received {}",
            responses.len()
        ));
    }

    let trendbars = parse_trendbars_response(&responses[0], &resolved.symbol)?;
    Ok(CTraderHistoricalBarsFetchResult {
        symbol: resolved.symbol.clone(),
        bars: trendbars.bars,
        has_more: trendbars.has_more,
    })
}

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

    if auth_responses.len() != 3 {
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
        .ok_or_else(|| anyhow!("cTrader symbol '{}' was not found for this account", request.symbol_name))?;

    let detail_responses = transport.send_sequence(&[
        build_symbol_by_id_request(account_id, &[light_symbol.symbol_id], "symbol-by-id-1"),
    ])?;
    if detail_responses.len() != 1 {
        return Err(anyhow!(
            "expected 1 cTrader symbol-by-id response, received {}",
            detail_responses.len()
        ));
    }

    let mut symbol = parse_symbol_by_id_response(&detail_responses[0])?
        .into_iter()
        .find(|symbol| symbol.symbol_id == light_symbol.symbol_id)
        .ok_or_else(|| anyhow!("cTrader full symbol metadata missing for symbol {}", light_symbol.symbol_id))?;
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
    Ok(match period {
        1 => "M1",
        2 => "M2",
        3 => "M3",
        4 => "M4",
        5 => "M5",
        6 => "M10",
        7 => "M15",
        8 => "M30",
        9 => "H1",
        10 => "H4",
        11 => "H12",
        12 => "D1",
        13 => "W1",
        14 => "MN1",
        other => return Err(anyhow!("unsupported cTrader trendbar period {}", other)),
    }
    .to_string())
}

fn trading_mode_enabled(value: Option<&Value>) -> bool {
    match value {
        Some(Value::String(mode)) => mode == "ENABLED",
        Some(Value::Number(number)) => number.as_i64() == Some(0),
        None => false,
        _ => false,
    }
}

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
        fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> anyhow::Result<Vec<String>> {
            self.sent.lock().expect("sent lock").extend(messages.iter().cloned());
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
        let response = serde_json::json!({
            "clientMsgId": "symbol-by-id-1",
            "payloadType": 2117,
            "payload": {
                "symbol": [
                    {
                        "symbolId": 1,
                        "digits": 5,
                        "pipPosition": 4,
                        "tradingMode": "ENABLED"
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
        assert_eq!(result.ticks[0].timestamp_ms, 1_700_000_000_000);
        assert_eq!(result.ticks[1].timestamp_ms, 1_699_999_999_750);
        assert!((result.ticks[0].price - 1.10120).abs() < 1e-9);
        assert!((result.ticks[1].price - 1.10100).abs() < 1e-9);
        assert!(result.has_more);
    }

    #[test]
    fn chart_history_backend_loads_symbol_metadata_then_historical_bars_and_ticks() {
        let transport = StubTransport::with_responses(vec![
            Ok(r#"{"clientMsgId":"app-auth-1","payloadType":2101,"payload":{}}"#.to_string()),
            Ok(r#"{"clientMsgId":"account-auth-1","payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
            Ok(r#"{"clientMsgId":"symbols-1","payloadType":2115,"payload":{"ctidTraderAccountId":712345,"symbol":[{"symbolId":14,"symbolName":"EURUSD","enabled":true,"description":"Euro vs Dollar"}]}}"#.to_string()),
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
        assert_eq!(transport.sent_len(), 7);
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
        let transport = StubTransport::with_responses(vec![
            Ok(r#"{"clientMsgId":"app-auth-1","payloadType":2101,"payload":{}}"#.to_string()),
            Ok(r#"{"clientMsgId":"account-auth-1","payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
            Ok(r#"{"clientMsgId":"symbols-1","payloadType":2115,"payload":{"ctidTraderAccountId":712345,"symbol":[{"symbolId":14,"symbolName":"EURUSD","enabled":true,"description":"Euro vs Dollar"}]}}"#.to_string()),
            Ok(r#"{"clientMsgId":"symbol-by-id-1","payloadType":2117,"payload":{"symbol":[{"symbolId":14,"digits":5,"pipPosition":4,"tradingMode":"ENABLED"}]}}"#.to_string()),
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
        assert_eq!(transport.sent_len(), 5);
    }
}
