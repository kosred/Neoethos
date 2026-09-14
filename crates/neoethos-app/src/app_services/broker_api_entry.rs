//! Current broker inputs for one entry decision; not a historical or promotion permit.
//! The ordinary order backend, volume/bracket math and margin parser remain shared.
use super::*;
use crate::app_services::ctrader_data::{CTraderAssetInfo, parse_asset_list_response};
use crate::app_services::ctrader_messages::{
    CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
    build_asset_list_request,
};
use crate::app_services::live_spots::{self, FreshSpotQuote, SpotSessionId};
use neoethos_core::symbol_metadata::SymbolMetadata;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
enum LossConversion {
    Identity,
    Direct(i64),
    Inverse(i64),
}

/// Kept private-fielded and local to this request/decision. No serialization,
// global cache, historical receipt, completed-deal requirement or blanket permit.
pub(crate) struct LiveEntryContext {
    environment: CTraderEnvironment,
    resolved: CTraderResolvedSymbol,
    margin: MarginStatus,
    metadata: SymbolMetadata,
    account_currency: String,
    trader_money_digits: u32,
    loss_conversion: LossConversion,
    request_started: Instant,
    client_order_id: String,
}

pub(crate) struct LiveEntryValuation {
    pub(crate) bid: f64,
    pub(crate) ask: f64,
    pub(crate) price_for_risk: f64,
    pub(crate) quote_to_account_rate: f64,
    // Original primary/conversion observations, not a refreshed risk valuation.
    quote_valid_until: Instant,
}

impl LiveEntryContext {
    pub(crate) fn client_order_id(&self) -> &str {
        &self.client_order_id
    }
    pub(crate) fn metadata(&self) -> &SymbolMetadata {
        &self.metadata
    }
    pub(crate) fn contract(&self) -> &CTraderResolvedSymbol {
        &self.resolved
    }
    pub(crate) fn margin(&self) -> &MarginStatus {
        &self.margin
    }
    pub(crate) fn account_currency(&self) -> &str {
        &self.account_currency
    }
    pub(crate) fn trader_money_digits(&self) -> u32 {
        self.trader_money_digits
    }

    fn require_age(&self, max_age_ms: i64) -> Result<()> {
        self.require_age_at(max_age_ms, Instant::now())
    }

    fn require_age_at(&self, max_age_ms: i64, now: Instant) -> Result<()> {
        let max_age_ms =
            u64::try_from(max_age_ms).context("negative entry observation age bound")?;
        anyhow::ensure!(
            max_age_ms > 0,
            "entry observation age bound must be positive"
        );
        anyhow::ensure!(
            now.checked_duration_since(self.request_started)
                .is_some_and(|age| age <= Duration::from_millis(max_age_ms)),
            "entry account/symbol observation expired; acquire a new decision context"
        );
        Ok(())
    }

    pub(crate) fn valuation(
        &self,
        session: SpotSessionId,
        now_ms: i64,
        max_age_ms: i64,
    ) -> Result<LiveEntryValuation> {
        self.valuation_with(session, now_ms, max_age_ms, |symbol_id| {
            live_spots::get_fresh_tick(
                self.resolved.account_id,
                self.environment,
                session,
                symbol_id,
                now_ms,
                max_age_ms,
            )
            .map_err(|reason| anyhow!("entry quote {symbol_id} refused: {reason:?}"))
        })
    }

    fn valuation_with(
        &self,
        session: SpotSessionId,
        now_ms: i64,
        max_age_ms: i64,
        mut quote: impl FnMut(i64) -> Result<FreshSpotQuote>,
    ) -> Result<LiveEntryValuation> {
        let valuation_started = Instant::now();
        self.require_age_at(max_age_ms, valuation_started)?;
        anyhow::ensure!(now_ms > 0, "invalid entry quote observation time");
        let mut oldest_quote_timestamp_ms = now_ms;
        let primary = quote(self.resolved.symbol.symbol_id)?;
        let mut validate = |q: &FreshSpotQuote, symbol_id| -> Result<()> {
            anyhow::ensure!(
                q.account_id == self.resolved.account_id
                    && q.environment == self.environment
                    && q.session_id == session
                    && q.symbol_id == symbol_id,
                "entry quote identity/session differs from the pinned broker context"
            );
            anyhow::ensure!(
                q.bid.is_finite() && q.ask.is_finite() && q.bid > 0.0 && q.ask >= q.bid,
                "entry quote has invalid prices"
            );
            for (broker, received) in [
                (q.bid_broker_timestamp_ms, q.bid_received_at_unix_ms),
                (q.ask_broker_timestamp_ms, q.ask_received_at_unix_ms),
            ] {
                anyhow::ensure!(
                    broker > 0
                        && broker <= received
                        && received <= now_ms
                        && now_ms - broker <= max_age_ms,
                    "entry quote side timestamp is invalid or expired"
                );
                oldest_quote_timestamp_ms = oldest_quote_timestamp_ms.min(broker).min(received);
            }
            Ok(())
        };
        validate(&primary, self.resolved.symbol.symbol_id)?;
        let quote_to_account_rate = match self.loss_conversion {
            LossConversion::Identity => 1.0,
            LossConversion::Direct(symbol_id) => {
                let leg = if symbol_id == primary.symbol_id {
                    primary.clone()
                } else {
                    quote(symbol_id)?
                };
                validate(&leg, symbol_id)?;
                loss_rate_from_prices(leg.bid, leg.ask, false)?
            }
            LossConversion::Inverse(symbol_id) => {
                let leg = if symbol_id == primary.symbol_id {
                    primary.clone()
                } else {
                    quote(symbol_id)?
                };
                validate(&leg, symbol_id)?;
                loss_rate_from_prices(leg.bid, leg.ask, true)?
            }
        };
        anyhow::ensure!(
            quote_to_account_rate.is_finite() && quote_to_account_rate > 0.0,
            "entry loss conversion is not finite and positive"
        );
        // This is stop-loss liability valuation, not realized PnL conversion.
        // For a base-currency account the existing pip helper divides by price:
        // use bid (cost of funding quote-currency loss), never a closed bar.
        // Otherwise ask conservatively values the existing notional ceiling.
        let price_for_risk = if self.metadata.base == self.account_currency {
            primary.bid
        } else {
            primary.ask
        };
        let remaining_ms = max_age_ms - (now_ms - oldest_quote_timestamp_ms);
        let quote_valid_until = valuation_started
            .checked_add(Duration::from_millis(u64::try_from(remaining_ms)?))
            .context("entry quote expiry overflow")?;
        Ok(LiveEntryValuation {
            bid: primary.bid,
            ask: primary.ask,
            price_for_risk,
            quote_to_account_rate,
            quote_valid_until,
        })
    }

    /// Re-resolve the actual routing credentials/full symbol, then check the
    /// same decision's session/age/budget immediately before the shared execute.
    /// No atomic quote-to-fill guarantee is implied.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn submit_market_order_blocking(
        self,
        session: SpotSessionId,
        side: OrderSide,
        lots: f64,
        stop_loss_pips: Option<f64>,
        take_profit_pips: Option<f64>,
        comment: Option<String>,
        max_age_ms: i64,
        submission_may_have_started: &std::sync::atomic::AtomicBool,
        persist_intent: impl FnOnce(&CTraderExecutionRuntimeRequest) -> Result<()>,
        final_check: impl FnOnce(&SymbolMetadata, &LiveEntryValuation) -> Result<()>,
    ) -> Result<CTraderExecutionOutcome> {
        ensure_entry_margin_halt_clear()?;
        let creds = resolve_creds_expecting(Some(self.environment == CTraderEnvironment::Live))?;
        self.submit_with(
            creds,
            side,
            lots,
            stop_loss_pips,
            take_profit_pips,
            comment,
            max_age_ms,
            resolve_symbol,
            |context| context.valuation(session, current_unix_time_ms_i64()?, max_age_ms),
            final_check,
            submission_may_have_started,
            persist_intent,
            Instant::now,
            |request| ProductionCTraderExecutionBackend::default().execute(request),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_with(
        self,
        creds: ResolvedCreds,
        side: OrderSide,
        lots: f64,
        stop_loss_pips: Option<f64>,
        take_profit_pips: Option<f64>,
        comment: Option<String>,
        max_age_ms: i64,
        resolve: impl FnOnce(&CTraderSymbolLookupRequest) -> Result<CTraderResolvedSymbol>,
        valuation: impl FnOnce(&Self) -> Result<LiveEntryValuation>,
        final_check: impl FnOnce(&SymbolMetadata, &LiveEntryValuation) -> Result<()>,
        submission_may_have_started: &std::sync::atomic::AtomicBool,
        persist_intent: impl FnOnce(&CTraderExecutionRuntimeRequest) -> Result<()>,
        recheck_now: impl FnOnce() -> Instant,
        execute: impl FnOnce(&CTraderExecutionRuntimeRequest) -> Result<CTraderExecutionOutcome>,
    ) -> Result<CTraderExecutionOutcome> {
        anyhow::ensure!(
            creds.environment == self.environment,
            "entry broker environment changed"
        );
        validate_order_volume_and_brackets(Some(lots), stop_loss_pips, take_profit_pips)?;
        let resolved = resolve_order_symbol_from_creds(
            &creds,
            &self.resolved.light_symbol.symbol_name,
            Some(self.resolved.account_id),
            resolve,
        )?;
        anyhow::ensure!(
            resolved == self.resolved,
            "entry broker symbol/asset/lot/pip contract changed since sizing; no order is sent"
        );
        let prep = prepare_new_order_from_resolved_creds(
            creds,
            &self.metadata.symbol,
            lots,
            stop_loss_pips,
            take_profit_pips,
            Some(self.resolved.account_id),
            |_| Ok(resolved),
        )?;
        ensure_broker_allows_new_order(
            &self.metadata.symbol,
            side,
            prep.broker_allows_new_positions,
            prep.broker_allows_short_selling,
        )?;
        self.require_age(max_age_ms)?;
        let current = valuation(&self)?;
        final_check(&self.metadata, &current)?;
        execute_prepared_market_order(
            prep,
            side,
            comment,
            Some(self.client_order_id.clone()),
            |request| {
                persist_intent(request)?;
                // Check the original observations after the potentially blocking
                // durable write. Later backend auth/mutex/socket waits are separate.
                let now = recheck_now();
                self.require_age_at(max_age_ms, now)?;
                anyhow::ensure!(
                    now <= current.quote_valid_until,
                    "entry primary/conversion quote expired during intent persistence; no order is sent"
                );
                submission_may_have_started.store(true, std::sync::atomic::Ordering::Release);
                execute(request)
            },
        )
    }
}

/// Read-only full-symbol identity/pricing lookup for diagnostics and entry setup.
/// It does not require account equity, closed deals or a historical research permit.
pub(crate) fn fetch_bound_broker_symbol_blocking(
    symbol: &str,
    expected_environment: CTraderEnvironment,
    expected_account_id: i64,
    expected_symbol_id: i64,
) -> Result<CTraderResolvedSymbol> {
    let creds = resolve_creds_expecting(Some(expected_environment == CTraderEnvironment::Live))?;
    resolve_bound_symbol_from_creds(
        &creds,
        symbol,
        expected_account_id,
        expected_symbol_id,
        resolve_symbol,
    )
}

fn resolve_bound_symbol_from_creds(
    creds: &ResolvedCreds,
    symbol: &str,
    expected_account_id: i64,
    expected_symbol_id: i64,
    resolve: impl FnOnce(&CTraderSymbolLookupRequest) -> Result<CTraderResolvedSymbol>,
) -> Result<CTraderResolvedSymbol> {
    let resolved =
        resolve_order_symbol_from_creds(creds, symbol, Some(expected_account_id), resolve)?;
    anyhow::ensure!(
        expected_symbol_id > 0
            && resolved.symbol.symbol_id == expected_symbol_id
            && resolved.light_symbol.symbol_id == expected_symbol_id,
        "full broker symbol differs from the admitted broker symbol"
    );
    anyhow::ensure!(
        (0..=5).contains(&resolved.symbol.digits)
            && (0..=resolved.symbol.digits).contains(&resolved.symbol.pip_position),
        "full broker symbol has unsupported pip/digit precision"
    );
    Ok(resolved)
}

pub(crate) fn fetch_live_entry_context_blocking(
    symbol: &str,
    expected_environment: CTraderEnvironment,
    expected_account_id: i64,
    expected_symbol_id: i64,
) -> Result<LiveEntryContext> {
    let creds = resolve_creds_expecting(Some(expected_environment == CTraderEnvironment::Live))?;
    let transport = ProductionCTraderOpenApiTransport::new(creds.environment.endpoint_host());
    fetch_with_transport(
        &transport,
        &creds,
        symbol,
        expected_account_id,
        expected_symbol_id,
        resolve_symbol,
    )
}

fn fetch_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    creds: &ResolvedCreds,
    symbol: &str,
    expected_account_id: i64,
    expected_symbol_id: i64,
    resolve: impl FnOnce(&CTraderSymbolLookupRequest) -> Result<CTraderResolvedSymbol>,
) -> Result<LiveEntryContext> {
    let request_started = Instant::now();
    let resolved = resolve_bound_symbol_from_creds(
        creds,
        symbol,
        expected_account_id,
        expected_symbol_id,
        resolve,
    )?;
    let account_id = resolved.account_id;
    let responses = crate::app_services::ctrader_messages::send_sequence_resilient(
        transport,
        &[
            build_application_auth_request(&creds.client_id, &creds.client_secret, "entry-app"),
            build_account_auth_request(account_id, &creds.access_token, "entry-account"),
            build_margin_call_list_request(account_id, "entry-margin"),
            build_trader_request(account_id, "entry-trader"),
            build_reconcile_request(account_id, false, "entry-reconcile"),
            build_get_position_unrealized_pnl_request(account_id, "entry-pnl"),
            build_asset_list_request(account_id, "entry-assets"),
            build_symbols_list_request(account_id, false, "entry-symbols"),
        ],
        8,
        "cTrader entry account and symbol context",
    )?;
    anyhow::ensure!(
        responses.len() == 8,
        "entry context requires all eight response envelopes"
    );
    let (margin, trader) = parse_margin_status_responses(creds, &responses[..6])?;
    ensure_success_payload_type(&responses[6], CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(&responses[7], CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE)?;
    require_response_account(&responses[6], account_id, "entry assets")?;
    require_response_account(&responses[7], account_id, "entry symbols")?;
    let assets = parse_asset_list_response(&responses[6])?;
    let symbols = parse_symbols_list_response(&responses[7])?;
    let deposit = trader
        .deposit_asset_id
        .context("entry trader omitted depositAssetId")?;
    let account_currency = asset_name(&assets, deposit)?;
    let (metadata, loss_conversion) =
        project_entry_metadata(&resolved, deposit, &assets, &symbols.symbols)?;
    anyhow::ensure!(
        margin.account_id == account_id
            && margin.balance.is_finite()
            && margin.balance > 0.0
            && margin.equity.is_finite()
            && margin.equity > 0.0
            && margin.used_margin.is_finite()
            && margin.used_margin >= 0.0
            && margin.positions_missing_used_margin == 0
            && !margin.is_margin_call(),
        "entry account equity/margin snapshot is incomplete or does not permit new exposure"
    );
    Ok(LiveEntryContext {
        environment: creds.environment,
        resolved,
        margin,
        metadata,
        account_currency,
        trader_money_digits: trader.money_digits,
        loss_conversion,
        request_started,
        // One ID per decision; never regenerated during submit or an ambiguous retry.
        // This separates execution-cache fingerprints, not a broker deduplication guarantee.
        client_order_id: format_entry_client_order_id(rand::random::<u128>()),
    })
}

fn format_entry_client_order_id(nonce: u128) -> String {
    format!("neo-{nonce:032x}")
}

fn asset_name(assets: &[CTraderAssetInfo], id: i64) -> Result<String> {
    anyhow::ensure!(id > 0, "entry currency asset id must be positive");
    let mut matches = assets.iter().filter(|asset| asset.asset_id == id);
    let asset = matches
        .next()
        .context("entry currency is absent from the same-request asset registry")?;
    anyhow::ensure!(
        matches.next().is_none()
            && !asset.name.trim().is_empty()
            && asset.name == asset.name.trim(),
        "entry currency asset registry is ambiguous or malformed"
    );
    let name = asset.name.to_ascii_uppercase();
    anyhow::ensure!(
        assets
            .iter()
            .filter(|other| other.name.to_ascii_uppercase() == name)
            .count()
            == 1,
        "entry currency name maps to multiple broker asset ids"
    );
    Ok(name)
}

fn project_entry_metadata(
    resolved: &CTraderResolvedSymbol,
    deposit: i64,
    assets: &[CTraderAssetInfo],
    symbols: &[CTraderLightSymbolInfo],
) -> Result<(SymbolMetadata, LossConversion)> {
    let symbol = &resolved.symbol;
    let light = &resolved.light_symbol;
    anyhow::ensure!(
        symbol.symbol_id > 0
            && symbol.symbol_id == light.symbol_id
            && symbol.symbol_name == light.symbol_name
            && light.enabled
            && !symbol.is_archived
            && symbol.is_trading_enabled,
        "entry full/light symbol identity or availability disagrees"
    );
    let mut recorded = symbols.iter().filter(|s| s.symbol_id == light.symbol_id);
    anyhow::ensure!(
        recorded.next() == Some(light) && recorded.next().is_none(),
        "entry same-request light symbol differs from resolved full-symbol binding"
    );
    let base = light
        .base_asset_id
        .context("entry symbol omitted baseAssetId")?;
    let quote = light
        .quote_asset_id
        .context("entry symbol omitted quoteAssetId")?;
    anyhow::ensure!(
        base != quote,
        "entry symbol has identical base and quote assets"
    );
    let base_name = asset_name(assets, base)?;
    let quote_name = asset_name(assets, quote)?;
    let _ = asset_name(assets, deposit)?;
    anyhow::ensure!(
        (0..=5).contains(&symbol.digits) && (0..=symbol.digits).contains(&symbol.pip_position),
        "entry symbol has unsupported pip/digit precision"
    );
    let lot_size = symbol.lot_size.context("entry symbol omitted lotSize")?;
    let min = symbol
        .min_volume
        .context("entry symbol omitted minVolume")?;
    let max = symbol
        .max_volume
        .context("entry symbol omitted maxVolume")?;
    let step = symbol
        .step_volume
        .context("entry symbol omitted stepVolume")?;
    anyhow::ensure!(
        [lot_size, min, max, step]
            .into_iter()
            .all(|v| (1..=MAX_EXACT_BROKER_VOLUME).contains(&v))
            && min <= max,
        "entry symbol has invalid or inexact volume-grid metadata"
    );
    symbol
        .financials
        .as_ref()
        .context("entry symbol omitted full trading metadata")?;
    let additional_conversion = required_entry_conversion_symbol(light, deposit, symbols)?;
    let conversion = if quote == deposit {
        LossConversion::Identity
    } else if base == deposit {
        LossConversion::Inverse(symbol.symbol_id)
    } else {
        let leg = additional_conversion
            .context("cross-currency entry unexpectedly has no conversion dependency")?;
        if leg.base_asset_id == Some(quote) {
            LossConversion::Direct(leg.symbol_id)
        } else {
            LossConversion::Inverse(leg.symbol_id)
        }
    };
    let pip_size = 10.0_f64.powi(-symbol.pip_position);
    let contract_size = lot_size as f64 / 100.0;
    Ok((
        SymbolMetadata {
            symbol: light.symbol_name.clone(),
            base: base_name,
            quote: quote_name,
            pip_size,
            contract_size,
            pip_value_quote: pip_size * contract_size,
            digits: symbol.digits as u32,
            min_lot: min as f64 / lot_size as f64,
            max_lot: max as f64 / lot_size as f64,
            lot_step: step as f64 / lot_size as f64,
            typical_price: None,
            typical_spread_pips: None,
            commission_per_lot: None,
            daily_swap_long_pips: None,
            daily_swap_short_pips: None,
            pnl_conversion_fee_rate: None,
            commission_type: None,
            commission_rate_decimal: None,
        },
        conversion,
    ))
}

/// Single direct/inverse asset-ID selection shared by entry sizing and the
/// stream's subscription dependency list. None means the primary quote alone
/// suffices (quote or base account currency); never a missing-rate fallback.
pub(crate) fn required_entry_conversion_symbol<'a>(
    primary: &CTraderLightSymbolInfo,
    deposit: i64,
    symbols: &'a [CTraderLightSymbolInfo],
) -> Result<Option<&'a CTraderLightSymbolInfo>> {
    let base = primary
        .base_asset_id
        .context("entry symbol omitted baseAssetId")?;
    let quote = primary
        .quote_asset_id
        .context("entry symbol omitted quoteAssetId")?;
    anyhow::ensure!(
        primary.symbol_id > 0 && base > 0 && quote > 0 && deposit > 0 && base != quote,
        "entry conversion has invalid symbol/asset identity"
    );
    if quote == deposit || base == deposit {
        return Ok(None);
    }
    let mut candidates = symbols
        .iter()
        .filter(|symbol| symbol.enabled && symbol.symbol_id > 0)
        .filter(|symbol| {
            (symbol.base_asset_id == Some(quote) && symbol.quote_asset_id == Some(deposit))
                || (symbol.base_asset_id == Some(deposit) && symbol.quote_asset_id == Some(quote))
        });
    let leg = candidates.next().context(
        "no direct broker asset-linked live conversion symbol; historical/name-derived FX is not permitted"
    )?;
    anyhow::ensure!(
        candidates.next().is_none(),
        "ambiguous broker live conversion symbols"
    );
    anyhow::ensure!(
        !leg.symbol_name.trim().is_empty()
            && leg.symbol_name == leg.symbol_name.trim()
            && symbols
                .iter()
                .filter(|symbol| symbol.symbol_id == leg.symbol_id)
                .count()
                == 1,
        "conversion symbol catalog identity is ambiguous or malformed"
    );
    Ok(Some(leg))
}

fn loss_rate_from_prices(bid: f64, ask: f64, inverse: bool) -> Result<f64> {
    anyhow::ensure!(
        bid.is_finite() && ask.is_finite() && bid > 0.0 && ask >= bid,
        "invalid loss-conversion bid/ask"
    );
    let rate = if inverse { 1.0 / bid } else { ask };
    anyhow::ensure!(
        rate.is_finite() && rate > 0.0,
        "invalid loss-conversion rate"
    );
    Ok(rate)
}

#[cfg(test)]
#[path = "broker_api_entry_tests.rs"]
mod tests;
