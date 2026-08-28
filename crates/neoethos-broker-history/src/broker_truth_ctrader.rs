//! Exact cTrader capture adapter for the immutable V2 broker-truth producer.
//!
//! The adapter borrows an already-authenticated session. It cannot open a
//! transport, invent reviewed replay rules, publish a bundle, or install any
//! evaluation authority.

use std::collections::{BTreeMap, VecDeque};

use anyhow::{Context, Result, anyhow, bail};
use neoethos_broker_truth::{EvidenceWindowV1, MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2, QuoteSideV1};
use serde_json::{Value, json};

use crate::broker_truth_capture::{
    BrokerEvidenceRowKindV2, BrokerFinancialTruthCaptureRequestV2, CapturedBrokerEvidencePairV2,
    CapturedBrokerEvidenceRowV2, CapturedCloseDealReconciliationV2, CapturedDealPageV2,
    CapturedQuoteSideV2, CapturedQuoteSynchronizationV2, CapturedTickPageV2, CapturedTickV2,
    ExactBrokerTruthCaptureSessionV2, ExactQuoteCaptureRequestV2, ExactQuoteInstrumentV2,
    ExactQuoteSynchronizationCaptureRequestV2,
};
use crate::ctrader_data::{
    CTraderAssetInfo, CTraderLightSymbolInfo, CTraderSymbolInfo, parse_asset_list_response,
    parse_symbol_by_id_response, parse_symbols_list_response, parse_tick_data_response,
};
use crate::ctrader_messages::{
    CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE, CTRADER_QUOTE_TYPE_ASK, CTRADER_QUOTE_TYPE_BID,
    CTraderDealListRequest, CTraderOpenApiJsonMessage, CTraderOpenApiSessionResponse,
    ProductionCTraderOpenApiSession, build_asset_list_request, build_deal_list_request,
    build_get_position_unrealized_pnl_request, build_get_tick_data_request,
    build_reconcile_request, build_symbol_by_id_request, build_symbols_list_request,
    build_trader_request, ctrader_historical_session_error_from_response,
    parse_get_position_unrealized_pnl_response,
};

const MAX_CLIENT_MESSAGE_NAMESPACE_BYTES: usize = 80;

/// Narrow exchange seam implemented by the existing authenticated production
/// session and by deterministic offline contract fixtures.
pub trait CTraderBrokerTruthSameSessionV2 {
    fn exchange_same_session(&mut self, message: &CTraderOpenApiJsonMessage) -> Result<String>;
}

impl CTraderBrokerTruthSameSessionV2 for ProductionCTraderOpenApiSession {
    fn exchange_same_session(&mut self, message: &CTraderOpenApiJsonMessage) -> Result<String> {
        match self.send_one(message, None)? {
            CTraderOpenApiSessionResponse::Expected(response) => Ok(response),
            CTraderOpenApiSessionResponse::BrokerError(response) => {
                let error = ctrader_historical_session_error_from_response(&response)?;
                Err(error.context("cTrader rejected exact V2 broker-truth capture request"))
            }
        }
    }
}

/// One externally reviewed replay-rule capture bound to the exact account,
/// instrument, and evaluated window for which it was reviewed.
#[derive(Clone, Debug)]
pub struct ReviewedCTraderQuoteSynchronizationV2 {
    account_id: i64,
    instrument: ExactQuoteInstrumentV2,
    window: EvidenceWindowV1,
    capture: CapturedQuoteSynchronizationV2,
}

impl ReviewedCTraderQuoteSynchronizationV2 {
    pub fn new(
        account_id: i64,
        instrument: ExactQuoteInstrumentV2,
        window: EvidenceWindowV1,
        capture: CapturedQuoteSynchronizationV2,
    ) -> Result<Self> {
        if account_id <= 0 {
            bail!("reviewed cTrader synchronization account id must be positive");
        }
        Ok(Self {
            account_id,
            instrument,
            window,
            capture,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ExpectedCaptureStep {
    Quote {
        account_id: i64,
        instrument: ExactQuoteInstrumentV2,
        side: QuoteSideV1,
        window: EvidenceWindowV1,
    },
    Synchronization {
        account_id: i64,
        instrument: ExactQuoteInstrumentV2,
        window: EvidenceWindowV1,
    },
    SymbolContracts,
    PositionUnrealizedPnl,
    CloseDealReconciliation,
}

#[derive(Clone, Debug)]
struct RawBrokerResponse {
    client_msg_id: String,
    raw_json: String,
    envelope: Value,
}

impl RawBrokerResponse {
    fn payload(&self) -> &Value {
        &self.envelope["payload"]
    }
}

#[derive(Clone, Debug)]
struct LightSymbolAuthority {
    raw: RawBrokerResponse,
    parsed_by_id: BTreeMap<i64, CTraderLightSymbolInfo>,
    raw_by_id: BTreeMap<i64, Value>,
}

#[derive(Clone, Debug)]
struct FullSymbolAuthority {
    raw: RawBrokerResponse,
    parsed: CTraderSymbolInfo,
    raw_symbol: Value,
}

#[derive(Clone, Debug)]
struct AssetAuthority {
    raw: RawBrokerResponse,
    parsed_by_id: BTreeMap<i64, CTraderAssetInfo>,
    raw_by_id: BTreeMap<i64, Value>,
}

#[derive(Clone, Debug)]
struct TraderAuthority {
    raw: RawBrokerResponse,
    raw_trader: Value,
}

/// Run-scoped adapter over exactly one authenticated cTrader session.
pub struct CTraderBrokerTruthAdapterV2<'a, S>
where
    S: CTraderBrokerTruthSameSessionV2,
{
    session: &'a mut S,
    exact_request: BrokerFinancialTruthCaptureRequestV2,
    client_message_namespace: String,
    next_client_message_sequence: u64,
    deal_max_rows: u32,
    return_protection_orders: bool,
    expected_steps: VecDeque<ExpectedCaptureStep>,
    reviewed_synchronizations: VecDeque<ReviewedCTraderQuoteSynchronizationV2>,
    light_symbols: Option<LightSymbolAuthority>,
    full_symbols: BTreeMap<i64, FullSymbolAuthority>,
    assets: Option<AssetAuthority>,
    trader: Option<TraderAuthority>,
}

impl<'a, S> CTraderBrokerTruthAdapterV2<'a, S>
where
    S: CTraderBrokerTruthSameSessionV2,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session: &'a mut S,
        request: &BrokerFinancialTruthCaptureRequestV2,
        client_message_namespace: impl Into<String>,
        deal_max_rows: u32,
        return_protection_orders: bool,
        reviewed_synchronizations: Vec<ReviewedCTraderQuoteSynchronizationV2>,
    ) -> Result<Self> {
        let client_message_namespace = client_message_namespace.into();
        validate_client_message_namespace(&client_message_namespace)?;
        if deal_max_rows == 0 || deal_max_rows > i32::MAX as u32 {
            bail!("cTrader DealList maxRows must be in 1..=i32::MAX");
        }
        if !return_protection_orders {
            bail!("exact cTrader reconciliation requires returnProtectionOrders=true");
        }

        let expected_instruments = capture_instruments_in_order(request);
        if reviewed_synchronizations.len() != expected_instruments.len() {
            bail!(
                "exact V2 capture requires {} reviewed quote synchronizations, received {}",
                expected_instruments.len(),
                reviewed_synchronizations.len()
            );
        }
        for (index, (reviewed, instrument)) in reviewed_synchronizations
            .iter()
            .zip(&expected_instruments)
            .enumerate()
        {
            if reviewed.account_id != request.account_id()
                || reviewed.instrument != *instrument
                || reviewed.window != request.window()
            {
                bail!(
                    "reviewed quote synchronization {index} is not bound to the exact account/instrument/window"
                );
            }
        }

        let mut expected_steps = VecDeque::new();
        for instrument in expected_instruments {
            for side in [QuoteSideV1::Bid, QuoteSideV1::Ask] {
                expected_steps.push_back(ExpectedCaptureStep::Quote {
                    account_id: request.account_id(),
                    instrument: instrument.clone(),
                    side,
                    window: request.window(),
                });
            }
            expected_steps.push_back(ExpectedCaptureStep::Synchronization {
                account_id: request.account_id(),
                instrument,
                window: request.window(),
            });
        }
        expected_steps.extend([
            ExpectedCaptureStep::SymbolContracts,
            ExpectedCaptureStep::PositionUnrealizedPnl,
            ExpectedCaptureStep::CloseDealReconciliation,
        ]);

        Ok(Self {
            session,
            exact_request: request.clone(),
            client_message_namespace,
            next_client_message_sequence: 0,
            deal_max_rows,
            return_protection_orders,
            expected_steps,
            reviewed_synchronizations: reviewed_synchronizations.into(),
            light_symbols: None,
            full_symbols: BTreeMap::new(),
            assets: None,
            trader: None,
        })
    }

    fn require_next_step(&self, actual: &ExpectedCaptureStep) -> Result<()> {
        match self.expected_steps.front() {
            Some(expected) if expected == actual => Ok(()),
            Some(expected) => bail!(
                "out-of-order exact cTrader capture step: expected {expected:?}, received {actual:?}"
            ),
            None => bail!("exact cTrader capture adapter is already exhausted"),
        }
    }

    fn complete_step(&mut self) {
        let _ = self.expected_steps.pop_front();
    }

    fn require_exact_request(&self, request: &BrokerFinancialTruthCaptureRequestV2) -> Result<()> {
        if request != &self.exact_request {
            bail!("cTrader adapter received a different broker-truth capture request");
        }
        Ok(())
    }

    fn next_client_msg_id(&mut self, label: &str) -> Result<String> {
        let sequence = self.next_client_message_sequence;
        self.next_client_message_sequence = sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("cTrader clientMsgId sequence overflow"))?;
        Ok(format!(
            "{}-{sequence:016x}-{label}",
            self.client_message_namespace
        ))
    }

    fn exchange_checked(
        &mut self,
        message: &CTraderOpenApiJsonMessage,
        expected_payload_type: u32,
    ) -> Result<RawBrokerResponse> {
        let raw_json = self
            .session
            .exchange_same_session(message)
            .with_context(|| {
                format!(
                    "same-session cTrader exchange failed for payloadType {} clientMsgId {:?}",
                    message.payload_type, message.client_msg_id
                )
            })?;
        if raw_json.trim().is_empty() {
            bail!("cTrader returned an empty same-session response");
        }
        let envelope: Value = serde_json::from_str(&raw_json)
            .context("cTrader same-session response is not valid JSON")?;
        if !envelope.is_object()
            || envelope.get("clientMsgId").and_then(Value::as_str)
                != Some(message.client_msg_id.as_str())
            || envelope.get("payloadType").and_then(Value::as_u64)
                != Some(u64::from(expected_payload_type))
            || !envelope.get("payload").is_some_and(Value::is_object)
            || envelope
                .pointer("/payload/ctidTraderAccountId")
                .and_then(Value::as_i64)
                != Some(self.exact_request.account_id())
        {
            bail!(
                "cTrader response does not exactly match clientMsgId/payloadType/account/payload object"
            );
        }
        Ok(RawBrokerResponse {
            client_msg_id: message.client_msg_id.clone(),
            raw_json,
            envelope,
        })
    }

    fn ensure_light_symbols(&mut self) -> Result<()> {
        if self.light_symbols.is_some() {
            return Ok(());
        }
        let client_msg_id = self.next_client_msg_id("light-symbols")?;
        let message =
            build_symbols_list_request(self.exact_request.account_id(), false, client_msg_id);
        let raw = self.exchange_checked(&message, CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE)?;
        let parsed = parse_symbols_list_response(&raw.raw_json)
            .context("failed to decode raw cTrader light-symbol authority")?;
        if parsed.account_id != self.exact_request.account_id() {
            bail!("decoded cTrader light-symbol account differs from exact account");
        }

        let raw_symbols = required_array(&raw.envelope, "/payload/symbol", "light symbols")?;
        let mut raw_by_id = BTreeMap::new();
        for raw_symbol in raw_symbols {
            let symbol_id = raw_symbol
                .get("symbolId")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("raw cTrader light symbol omitted integer symbolId"))?;
            if symbol_id <= 0 || raw_by_id.insert(symbol_id, raw_symbol.clone()).is_some() {
                bail!("raw cTrader light-symbol authority has invalid or duplicate symbolId");
            }
        }

        let mut parsed_by_id = BTreeMap::new();
        for symbol in parsed.symbols {
            let symbol_id = symbol.symbol_id;
            if symbol_id <= 0 || parsed_by_id.insert(symbol_id, symbol).is_some() {
                bail!("decoded cTrader light-symbol authority has invalid or duplicate symbolId");
            }
        }
        if parsed_by_id.keys().ne(raw_by_id.keys()) {
            bail!("raw and decoded cTrader light-symbol identities differ");
        }
        self.light_symbols = Some(LightSymbolAuthority {
            raw,
            parsed_by_id,
            raw_by_id,
        });
        Ok(())
    }

    fn ensure_instrument(&mut self, instrument: &ExactQuoteInstrumentV2) -> Result<()> {
        self.ensure_light_symbols()?;
        validate_light_symbol(
            self.light_symbols
                .as_ref()
                .ok_or_else(|| anyhow!("cTrader light-symbol authority was not retained"))?,
            instrument,
        )?;

        if !self.full_symbols.contains_key(&instrument.symbol_id()) {
            let client_msg_id = self.next_client_msg_id("full-symbol")?;
            let message = build_symbol_by_id_request(
                self.exact_request.account_id(),
                &[instrument.symbol_id()],
                client_msg_id,
            );
            let raw =
                self.exchange_checked(&message, CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE)?;
            let mut parsed = parse_symbol_by_id_response(&raw.raw_json)
                .context("failed to decode raw cTrader full-symbol authority")?;
            if parsed.len() != 1 || parsed[0].symbol_id != instrument.symbol_id() {
                bail!("cTrader full-symbol response did not contain exactly the requested symbol");
            }
            let parsed = parsed.remove(0);
            let raw_matches = required_array(&raw.envelope, "/payload/symbol", "full symbols")?
                .iter()
                .filter(|value| {
                    value.get("symbolId").and_then(Value::as_i64) == Some(instrument.symbol_id())
                })
                .cloned()
                .collect::<Vec<_>>();
            if raw_matches.len() != 1 {
                bail!("raw cTrader full-symbol response is missing or duplicates requested symbol");
            }
            self.full_symbols.insert(
                instrument.symbol_id(),
                FullSymbolAuthority {
                    raw,
                    parsed,
                    raw_symbol: raw_matches[0].clone(),
                },
            );
        }
        let full = self
            .full_symbols
            .get(&instrument.symbol_id())
            .ok_or_else(|| anyhow!("cTrader full-symbol authority was not retained"))?;
        if full.parsed.symbol_id != instrument.symbol_id() {
            bail!("cached cTrader full-symbol identity changed");
        }
        Ok(())
    }

    fn ensure_assets(&mut self, exact_assets: &BTreeMap<i64, String>) -> Result<()> {
        if self.assets.is_none() {
            let client_msg_id = self.next_client_msg_id("assets")?;
            let message = build_asset_list_request(self.exact_request.account_id(), client_msg_id);
            let raw =
                self.exchange_checked(&message, CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE)?;
            let parsed = parse_asset_list_response(&raw.raw_json)
                .context("failed to decode raw cTrader asset authority")?;
            let raw_assets = required_array(&raw.envelope, "/payload/asset", "account assets")?;
            let mut raw_by_id = BTreeMap::new();
            for raw_asset in raw_assets {
                let asset_id = raw_asset
                    .get("assetId")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| anyhow!("raw cTrader asset omitted integer assetId"))?;
                if asset_id <= 0 || raw_by_id.insert(asset_id, raw_asset.clone()).is_some() {
                    bail!("raw cTrader asset authority has invalid or duplicate assetId");
                }
            }
            let mut parsed_by_id = BTreeMap::new();
            for asset in parsed {
                let asset_id = asset.asset_id;
                if asset_id <= 0 || parsed_by_id.insert(asset_id, asset).is_some() {
                    bail!("decoded cTrader asset authority has invalid or duplicate assetId");
                }
            }
            if parsed_by_id.keys().ne(raw_by_id.keys()) {
                bail!("raw and decoded cTrader asset identities differ");
            }
            self.assets = Some(AssetAuthority {
                raw,
                parsed_by_id,
                raw_by_id,
            });
        }

        let assets = self
            .assets
            .as_ref()
            .ok_or_else(|| anyhow!("cTrader asset authority was not retained"))?;
        for (asset_id, expected_name) in exact_assets {
            let asset = assets.parsed_by_id.get(asset_id).ok_or_else(|| {
                anyhow!("cTrader asset authority omitted required asset id {asset_id}")
            })?;
            if asset.name != *expected_name || asset.name.trim().is_empty() {
                bail!(
                    "cTrader asset authority name mismatch for asset {asset_id}: expected {expected_name:?}, received {:?}",
                    asset.name
                );
            }
        }
        Ok(())
    }

    fn ensure_trader(&mut self) -> Result<()> {
        if self.trader.is_none() {
            let client_msg_id = self.next_client_msg_id("trader")?;
            let message = build_trader_request(self.exact_request.account_id(), client_msg_id);
            let raw = self.exchange_checked(&message, CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE)?;
            let raw_trader = raw
                .envelope
                .pointer("/payload/trader")
                .filter(|value| value.is_object())
                .cloned()
                .ok_or_else(|| anyhow!("raw cTrader trader response omitted trader object"))?;
            let expected_asset = self.exact_request.binding().account_asset_id();
            if raw_trader.get("depositAssetId").and_then(Value::as_i64) != Some(expected_asset)
                || raw_trader
                    .get("moneyDigits")
                    .and_then(Value::as_u64)
                    .is_none()
                || raw_trader.get("balance").and_then(Value::as_i64).is_none()
            {
                bail!(
                    "raw cTrader trader authority omitted exact depositAssetId/moneyDigits/balance"
                );
            }
            self.trader = Some(TraderAuthority { raw, raw_trader });
        }
        Ok(())
    }

    fn capture_quote_pages(
        &mut self,
        request: &ExactQuoteCaptureRequestV2,
    ) -> Result<CapturedQuoteSideV2> {
        self.ensure_instrument(request.instrument())?;
        let symbol = self
            .full_symbols
            .get(&request.instrument().symbol_id())
            .ok_or_else(|| anyhow!("cTrader full-symbol authority missing before tick capture"))?
            .parsed
            .clone();
        let quote_type = match request.side() {
            QuoteSideV1::Bid => CTRADER_QUOTE_TYPE_BID,
            QuoteSideV1::Ask => CTRADER_QUOTE_TYPE_ASK,
        };
        let full_window = request.window();
        let mut chunk_to_exclusive = full_window.to_unix_ms_exclusive();
        let mut chunk_sequence = 0_u64;
        let mut pages = Vec::new();

        while chunk_to_exclusive > full_window.from_unix_ms_inclusive() {
            let chunk_from_inclusive = full_window
                .from_unix_ms_inclusive()
                .max(chunk_to_exclusive.saturating_sub(MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2));
            let chunk_window = EvidenceWindowV1::new(chunk_from_inclusive, chunk_to_exclusive)
                .context("failed to form exact <=7-day cTrader tick chunk")?;
            let mut page_to_exclusive = chunk_to_exclusive;
            let mut page_sequence = 0_u64;
            loop {
                let page_window = EvidenceWindowV1::new(chunk_from_inclusive, page_to_exclusive)
                    .context("failed to form exact cTrader tick page window")?;
                let client_msg_id = self.next_client_msg_id("ticks")?;
                let message = build_get_tick_data_request(
                    request.account_id(),
                    request.instrument().symbol_id(),
                    quote_type,
                    page_window.from_unix_ms_inclusive(),
                    page_window.to_unix_ms_exclusive(),
                    client_msg_id,
                );
                let raw = self
                    .exchange_checked(&message, CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE)?;
                let decoded = parse_tick_data_response(
                    &raw.raw_json,
                    request.account_id(),
                    &raw.client_msg_id,
                    &symbol,
                )
                .context("failed to decode exact cTrader tick page")?;
                if decoded.symbol_id != request.instrument().symbol_id() || decoded.ticks.is_empty()
                {
                    bail!("exact cTrader tick page omitted requested symbol ticks");
                }
                if decoded.ticks.iter().any(|tick| {
                    tick.timestamp_ms < page_window.from_unix_ms_inclusive()
                        || tick.timestamp_ms >= page_window.to_unix_ms_exclusive()
                        || !tick.price.is_finite()
                        || tick.price <= 0.0
                }) || decoded
                    .ticks
                    .windows(2)
                    .any(|pair| pair[1].timestamp_ms <= pair[0].timestamp_ms)
                {
                    bail!("decoded cTrader tick page is out of bounds or not strictly ascending");
                }
                let oldest_timestamp = decoded
                    .ticks
                    .first()
                    .ok_or_else(|| anyhow!("cTrader tick page lost its pagination boundary"))?
                    .timestamp_ms;
                let has_more = decoded.has_more;
                pages.push(CapturedTickPageV2::new(
                    request.account_id(),
                    request.instrument().symbol_id(),
                    request.side(),
                    chunk_sequence,
                    page_sequence,
                    raw.client_msg_id,
                    chunk_window,
                    page_window,
                    raw.raw_json,
                    decoded
                        .ticks
                        .into_iter()
                        .map(|tick| CapturedTickV2::new(tick.timestamp_ms, tick.price))
                        .collect(),
                    has_more,
                ));
                if !has_more {
                    break;
                }
                if oldest_timestamp <= chunk_from_inclusive || oldest_timestamp >= page_to_exclusive
                {
                    bail!(
                        "cTrader tick hasMore page has no strictly older exclusive pagination boundary"
                    );
                }
                page_to_exclusive = oldest_timestamp;
                page_sequence = page_sequence
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("cTrader tick page sequence overflow"))?;
            }
            chunk_to_exclusive = chunk_from_inclusive;
            chunk_sequence = chunk_sequence
                .checked_add(1)
                .ok_or_else(|| anyhow!("cTrader tick chunk sequence overflow"))?;
        }
        Ok(CapturedQuoteSideV2::new(pages))
    }

    fn capture_exact_symbol_contracts(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV2,
    ) -> Result<CapturedBrokerEvidencePairV2> {
        let instruments = exact_instruments_by_id(request)?;
        for instrument in instruments.values() {
            self.ensure_instrument(instrument)?;
        }
        let exact_assets = exact_assets_by_id(request, &instruments)?;
        self.ensure_assets(&exact_assets)?;
        self.ensure_trader()?;

        let light = self
            .light_symbols
            .as_ref()
            .ok_or_else(|| anyhow!("cTrader light-symbol authority missing before encoding"))?;
        let assets = self
            .assets
            .as_ref()
            .ok_or_else(|| anyhow!("cTrader asset authority missing before encoding"))?;
        let trader = self
            .trader
            .as_ref()
            .ok_or_else(|| anyhow!("cTrader trader authority missing before encoding"))?;

        let mut raw_rows = Vec::new();
        raw_rows.push(evidence_row(
            raw_rows.len(),
            request.account_id(),
            None,
            BrokerEvidenceRowKindV2::LightSymbolResponse,
            &light.raw,
            CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
            light.raw.raw_json.clone(),
        ));
        for (symbol_id, full) in &self.full_symbols {
            if instruments.contains_key(symbol_id) {
                raw_rows.push(evidence_row(
                    raw_rows.len(),
                    request.account_id(),
                    Some(*symbol_id),
                    BrokerEvidenceRowKindV2::SymbolResponse,
                    &full.raw,
                    CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE,
                    full.raw.raw_json.clone(),
                ));
            }
        }
        raw_rows.push(evidence_row(
            raw_rows.len(),
            request.account_id(),
            None,
            BrokerEvidenceRowKindV2::AccountAssetResponse,
            &assets.raw,
            CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE,
            assets.raw.raw_json.clone(),
        ));
        raw_rows.push(evidence_row(
            raw_rows.len(),
            request.account_id(),
            None,
            BrokerEvidenceRowKindV2::TraderAccountResponse,
            &trader.raw,
            CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE,
            trader.raw.raw_json.clone(),
        ));

        let mut decoded_rows = Vec::new();
        for (symbol_id, instrument) in &instruments {
            let raw_light_symbol = light.raw_by_id.get(symbol_id).ok_or_else(|| {
                anyhow!("raw light-symbol authority omitted required symbol {symbol_id}")
            })?;
            decoded_rows.push(CapturedBrokerEvidenceRowV2::new(
                decoded_rows.len() as u64,
                request.account_id(),
                Some(*symbol_id),
                None,
                BrokerEvidenceRowKindV2::LightSymbolContract,
                None,
                light.raw.client_msg_id.clone(),
                CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
                canonical_json(json!({
                    "authority": "ProtoOALightSymbol",
                    "exactInstrument": exact_instrument_json(instrument),
                    "rawLightSymbol": raw_light_symbol.clone(),
                }))?,
            ));
            let full = self.full_symbols.get(symbol_id).ok_or_else(|| {
                anyhow!("cTrader full-symbol authority omitted required symbol {symbol_id}")
            })?;
            decoded_rows.push(CapturedBrokerEvidenceRowV2::new(
                decoded_rows.len() as u64,
                request.account_id(),
                Some(*symbol_id),
                None,
                BrokerEvidenceRowKindV2::SymbolContract,
                None,
                full.raw.client_msg_id.clone(),
                CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE,
                canonical_json(json!({
                    "authority": "ProtoOASymbol",
                    "rawSymbol": full.raw_symbol.clone(),
                }))?,
            ));
        }
        let required_raw_assets =
            exact_assets
                .keys()
                .map(|asset_id| {
                    assets.raw_by_id.get(asset_id).cloned().ok_or_else(|| {
                        anyhow!("raw asset authority omitted required asset {asset_id}")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
        decoded_rows.push(CapturedBrokerEvidenceRowV2::new(
            decoded_rows.len() as u64,
            request.account_id(),
            None,
            None,
            BrokerEvidenceRowKindV2::AccountAssetContract,
            None,
            assets.raw.client_msg_id.clone(),
            CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE,
            canonical_json(json!({
                "accountAssetId": request.binding().account_asset_id(),
                "accountAssetName": request.binding().account_asset_name(),
                "requiredRawAssets": required_raw_assets,
            }))?,
        ));
        decoded_rows.push(CapturedBrokerEvidenceRowV2::new(
            decoded_rows.len() as u64,
            request.account_id(),
            None,
            None,
            BrokerEvidenceRowKindV2::TraderAccountContract,
            None,
            trader.raw.client_msg_id.clone(),
            CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE,
            canonical_json(json!({
                "accountAssetId": request.binding().account_asset_id(),
                "rawTrader": trader.raw_trader.clone(),
            }))?,
        ));
        Ok(CapturedBrokerEvidencePairV2::new(raw_rows, decoded_rows))
    }

    fn capture_exact_unrealized_pnl(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV2,
    ) -> Result<CapturedBrokerEvidencePairV2> {
        let client_msg_id = self.next_client_msg_id("unrealized-pnl")?;
        let message =
            build_get_position_unrealized_pnl_request(request.account_id(), client_msg_id);
        let raw = self.exchange_checked(
            &message,
            CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
        )?;
        let decoded = parse_get_position_unrealized_pnl_response(&raw.raw_json)
            .context("failed to decode exact cTrader unrealized PnL authority")?;
        if decoded.account_id != request.account_id()
            || decoded.positions.iter().any(|position| {
                !position.gross_unrealized_pnl.is_finite()
                    || !position.net_unrealized_pnl.is_finite()
            })
        {
            bail!("decoded cTrader unrealized PnL account/value is invalid");
        }
        let positions = decoded
            .positions
            .iter()
            .map(|position| {
                json!({
                    "positionId": position.position_id,
                    "grossUnrealizedPnL": position.gross_unrealized_pnl,
                    "netUnrealizedPnL": position.net_unrealized_pnl,
                })
            })
            .collect::<Vec<_>>();
        let raw_row = evidence_row(
            0,
            request.account_id(),
            None,
            BrokerEvidenceRowKindV2::PositionUnrealizedPnlResponse,
            &raw,
            CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
            raw.raw_json.clone(),
        );
        let decoded_row = CapturedBrokerEvidenceRowV2::new(
            0,
            request.account_id(),
            None,
            None,
            BrokerEvidenceRowKindV2::PositionUnrealizedPnl,
            None,
            raw.client_msg_id,
            CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
            canonical_json(json!({
                "accountId": decoded.account_id,
                "moneyDigits": decoded.money_digits,
                "positions": positions,
            }))?,
        );
        Ok(CapturedBrokerEvidencePairV2::new(
            vec![raw_row],
            vec![decoded_row],
        ))
    }

    fn capture_exact_close_deals(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV2,
    ) -> Result<CapturedCloseDealReconciliationV2> {
        let reconcile_client_msg_id = self.next_client_msg_id("reconcile")?;
        let reconcile_message = build_reconcile_request(
            request.account_id(),
            self.return_protection_orders,
            reconcile_client_msg_id,
        );
        let reconcile = self.exchange_checked(
            &reconcile_message,
            CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE,
        )?;
        for field in ["position", "order"] {
            if reconcile
                .payload()
                .get(field)
                .is_some_and(|value| !value.is_array())
            {
                bail!("raw cTrader reconcile {field} field is not an array");
            }
        }
        let reconcile_payload = reconcile.payload().clone();
        let reconcile_raw = CapturedBrokerEvidenceRowV2::new(
            0,
            request.account_id(),
            None,
            None,
            BrokerEvidenceRowKindV2::OpenPositionReconcileResponse,
            Some(request.window()),
            reconcile.client_msg_id,
            CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE,
            reconcile.raw_json,
        );

        let mut page_to_exclusive = request.window().to_unix_ms_exclusive();
        let mut page_sequence = 0_u64;
        let mut pages = Vec::new();
        let mut decoded_rows = Vec::new();
        loop {
            let page_window =
                EvidenceWindowV1::new(request.window().from_unix_ms_inclusive(), page_to_exclusive)
                    .context("failed to form exact cTrader DealList page window")?;
            let client_msg_id = self.next_client_msg_id("deals")?;
            let deal_request = CTraderDealListRequest {
                account_id: request.account_id(),
                from_timestamp_ms: Some(page_window.from_unix_ms_inclusive()),
                to_timestamp_ms: Some(page_window.to_unix_ms_exclusive()),
                max_rows: Some(self.deal_max_rows as i32),
            };
            let message = build_deal_list_request(&deal_request, client_msg_id);
            let raw =
                self.exchange_checked(&message, CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE)?;
            let payload = raw.payload();
            let has_more = payload
                .get("hasMore")
                .and_then(Value::as_bool)
                .ok_or_else(|| anyhow!("raw cTrader DealList response omitted boolean hasMore"))?;
            let raw_deals: &[Value] = match payload.get("deal") {
                Some(value) => value
                    .as_array()
                    .ok_or_else(|| anyhow!("raw cTrader DealList deal field is not an array"))?,
                None if !has_more => &[],
                None => bail!("cTrader DealList hasMore response omitted deal rows"),
            };
            if has_more && raw_deals.is_empty() {
                bail!("cTrader DealList hasMore response contains no pagination boundary");
            }
            let mut timestamps = Vec::with_capacity(raw_deals.len());
            for raw_deal in raw_deals {
                let timestamp = raw_deal
                    .get("executionTimestamp")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| {
                        anyhow!("raw cTrader DealList row omitted integer executionTimestamp")
                    })?;
                if timestamp < page_window.from_unix_ms_inclusive()
                    || timestamp >= page_window.to_unix_ms_exclusive()
                    || timestamps
                        .last()
                        .is_some_and(|previous| timestamp <= *previous)
                {
                    bail!("raw cTrader DealList timestamps are out of bounds or not ascending");
                }
                timestamps.push(timestamp);
            }
            let raw_payload = payload.clone();
            let next_boundary = timestamps.first().copied();
            pages.push(CapturedDealPageV2::new(
                request.account_id(),
                page_sequence,
                raw.client_msg_id.clone(),
                page_window,
                self.deal_max_rows,
                raw.raw_json,
                timestamps,
                has_more,
            ));
            decoded_rows.push(CapturedBrokerEvidenceRowV2::new(
                decoded_rows.len() as u64,
                request.account_id(),
                None,
                None,
                BrokerEvidenceRowKindV2::CloseDealReconciliation,
                Some(request.window()),
                raw.client_msg_id,
                CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE,
                canonical_json(json!({
                    "dealPageRequest": {
                        "fromTimestamp": page_window.from_unix_ms_inclusive(),
                        "toTimestamp": page_window.to_unix_ms_exclusive(),
                        "maxRows": self.deal_max_rows,
                    },
                    "rawDealPayload": raw_payload,
                    "rawReconcilePayload": reconcile_payload.clone(),
                    "returnProtectionOrders": self.return_protection_orders,
                }))?,
            ));
            if !has_more {
                break;
            }
            let next_boundary = next_boundary
                .ok_or_else(|| anyhow!("cTrader DealList hasMore page omitted oldest boundary"))?;
            if next_boundary <= request.window().from_unix_ms_inclusive()
                || next_boundary >= page_to_exclusive
            {
                bail!(
                    "cTrader DealList hasMore page has no strictly older exclusive pagination boundary"
                );
            }
            page_to_exclusive = next_boundary;
            page_sequence = page_sequence
                .checked_add(1)
                .ok_or_else(|| anyhow!("cTrader DealList page sequence overflow"))?;
        }
        Ok(CapturedCloseDealReconciliationV2::new(
            reconcile_raw,
            pages,
            decoded_rows,
        ))
    }
}

impl<S> ExactBrokerTruthCaptureSessionV2 for CTraderBrokerTruthAdapterV2<'_, S>
where
    S: CTraderBrokerTruthSameSessionV2,
{
    fn capture_quote_side(
        &mut self,
        request: &ExactQuoteCaptureRequestV2,
    ) -> Result<CapturedQuoteSideV2> {
        let step = ExpectedCaptureStep::Quote {
            account_id: request.account_id(),
            instrument: request.instrument().clone(),
            side: request.side(),
            window: request.window(),
        };
        self.require_next_step(&step)?;
        let capture = self.capture_quote_pages(request)?;
        self.complete_step();
        Ok(capture)
    }

    fn capture_quote_synchronization(
        &mut self,
        request: &ExactQuoteSynchronizationCaptureRequestV2,
    ) -> Result<CapturedQuoteSynchronizationV2> {
        let step = ExpectedCaptureStep::Synchronization {
            account_id: request.account_id(),
            instrument: request.instrument().clone(),
            window: request.window(),
        };
        self.require_next_step(&step)?;
        let reviewed = self
            .reviewed_synchronizations
            .pop_front()
            .ok_or_else(|| anyhow!("required reviewed quote synchronization is exhausted"))?;
        if reviewed.account_id != request.account_id()
            || reviewed.instrument != *request.instrument()
            || reviewed.window != request.window()
        {
            bail!("reviewed quote synchronization does not match exact capture request");
        }
        self.complete_step();
        Ok(reviewed.capture)
    }

    fn capture_symbol_contracts(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV2,
    ) -> Result<CapturedBrokerEvidencePairV2> {
        self.require_exact_request(request)?;
        self.require_next_step(&ExpectedCaptureStep::SymbolContracts)?;
        let capture = self.capture_exact_symbol_contracts(request)?;
        self.complete_step();
        Ok(capture)
    }

    fn capture_position_unrealized_pnl(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV2,
    ) -> Result<CapturedBrokerEvidencePairV2> {
        self.require_exact_request(request)?;
        self.require_next_step(&ExpectedCaptureStep::PositionUnrealizedPnl)?;
        let capture = self.capture_exact_unrealized_pnl(request)?;
        self.complete_step();
        Ok(capture)
    }

    fn capture_close_deal_reconciliation(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV2,
    ) -> Result<CapturedCloseDealReconciliationV2> {
        self.require_exact_request(request)?;
        self.require_next_step(&ExpectedCaptureStep::CloseDealReconciliation)?;
        let capture = self.capture_exact_close_deals(request)?;
        self.complete_step();
        Ok(capture)
    }
}

fn validate_client_message_namespace(namespace: &str) -> Result<()> {
    if namespace.is_empty()
        || namespace.len() > MAX_CLIENT_MESSAGE_NAMESPACE_BYTES
        || namespace != namespace.trim()
        || !namespace
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!("cTrader clientMsgId namespace must be a bounded non-empty ASCII token");
    }
    Ok(())
}

fn capture_instruments_in_order(
    request: &BrokerFinancialTruthCaptureRequestV2,
) -> Vec<ExactQuoteInstrumentV2> {
    std::iter::once(request.primary_instrument().clone())
        .chain(
            request
                .conversion_routes()
                .iter()
                .flat_map(|route| route.legs().iter().map(|leg| leg.instrument().clone())),
        )
        .collect()
}

fn exact_instruments_by_id(
    request: &BrokerFinancialTruthCaptureRequestV2,
) -> Result<BTreeMap<i64, ExactQuoteInstrumentV2>> {
    let mut instruments = BTreeMap::new();
    for instrument in capture_instruments_in_order(request) {
        if let Some(previous) = instruments.insert(instrument.symbol_id(), instrument.clone())
            && previous != instrument
        {
            bail!("one cTrader symbol id was assigned conflicting exact instruments");
        }
    }
    Ok(instruments)
}

fn exact_assets_by_id(
    request: &BrokerFinancialTruthCaptureRequestV2,
    instruments: &BTreeMap<i64, ExactQuoteInstrumentV2>,
) -> Result<BTreeMap<i64, String>> {
    let mut assets = BTreeMap::new();
    for instrument in instruments.values() {
        insert_exact_asset(
            &mut assets,
            instrument.base_asset_id(),
            instrument.base_asset_name(),
        )?;
        insert_exact_asset(
            &mut assets,
            instrument.quote_asset_id(),
            instrument.quote_asset_name(),
        )?;
    }
    insert_exact_asset(
        &mut assets,
        request.binding().account_asset_id(),
        request.binding().account_asset_name(),
    )?;
    Ok(assets)
}

fn insert_exact_asset(assets: &mut BTreeMap<i64, String>, id: i64, name: &str) -> Result<()> {
    if let Some(previous) = assets.insert(id, name.to_owned())
        && previous != name
    {
        bail!("one cTrader asset id was assigned conflicting exact names");
    }
    Ok(())
}

fn validate_light_symbol(
    authority: &LightSymbolAuthority,
    instrument: &ExactQuoteInstrumentV2,
) -> Result<()> {
    let light = authority
        .parsed_by_id
        .get(&instrument.symbol_id())
        .ok_or_else(|| {
            anyhow!(
                "raw light-symbol authority omitted exact symbol {}",
                instrument.symbol_id()
            )
        })?;
    if light.symbol_name != instrument.symbol_name()
        || light.base_asset_id != Some(instrument.base_asset_id())
        || light.quote_asset_id != Some(instrument.quote_asset_id())
    {
        bail!("raw cTrader light-symbol name/base/quote authority differs from exact instrument");
    }
    Ok(())
}

fn required_array<'a>(root: &'a Value, pointer: &str, label: &str) -> Result<&'a Vec<Value>> {
    root.pointer(pointer)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("raw cTrader {label} response omitted required array at {pointer}"))
}

fn evidence_row(
    sequence: usize,
    account_id: i64,
    symbol_id: Option<i64>,
    kind: BrokerEvidenceRowKindV2,
    raw: &RawBrokerResponse,
    payload_type: u32,
    payload_json: String,
) -> CapturedBrokerEvidenceRowV2 {
    CapturedBrokerEvidenceRowV2::new(
        sequence as u64,
        account_id,
        symbol_id,
        None,
        kind,
        None,
        raw.client_msg_id.clone(),
        payload_type,
        payload_json,
    )
}

fn exact_instrument_json(instrument: &ExactQuoteInstrumentV2) -> Value {
    json!({
        "symbolId": instrument.symbol_id(),
        "symbolName": instrument.symbol_name(),
        "baseAssetId": instrument.base_asset_id(),
        "baseAssetName": instrument.base_asset_name(),
        "quoteAssetId": instrument.quote_asset_id(),
        "quoteAssetName": instrument.quote_asset_name(),
    })
}

fn canonical_json(value: Value) -> Result<String> {
    serde_json::to_string(&value).context("failed to encode canonical decoded broker evidence")
}
